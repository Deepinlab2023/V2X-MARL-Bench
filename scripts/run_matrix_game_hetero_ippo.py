"""
run_matrix_game_hetero_ippo.py — IPPO (parameter sharing, PS) on a FIXED
heterogeneous C-V2X matrix game (one SUMO channel realization).

Same networks, loss, and ALL hyperparameters / training budget as
run_matrix_game_ippo.py (the homogeneous V-E study, PS variant): ppo_params
defaults (episodes=50000, batch=256, epochs=5, entropy_coef=0.01,
actor_lr=4e-4, critic_lr=6e-4, eps_clip=0.2).  ONLY change: the reward is
computed from a fixed heterogeneous channel realization G loaded from a
game-cache npz built by analysis/hetero_game.py.  The realization is held
fixed throughout training and evaluation.

Action space:
  default (single power): {0..3 = subchannel, 4 = NT};
  --three_power: {0..11 = sc*3 + power, 12 = NT}, powers [23, 10, 5] dBm.

The final deterministic policy a^L is the per-agent argmax of the logits
(same readout rule as the homogeneous study).  Reference optimum = the
SINGLE-POWER exact R*(G) (subset DP) from the cache; the 3P ratio
R_3P(a^L)/R*_1P is "reward relative to the single-power optimum" (may
exceed 1), not an efficiency w.r.t. the 3-power optimum (not computed).

Usage:
  python3 run_matrix_game_hetero_ippo.py --game_cache Results/Hetero_batch/games/low_snap7092.npz --seed 9
  python3 run_matrix_game_hetero_ippo.py --game_cache ... --seed 9 --three_power
"""

import argparse
import csv
import os
import random
import sys
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Networks.Agents.ppo_actor import PPOSharedActor
from Networks.Critics.ppo_critic import PPOSharedCritic
from analysis.hetero_game import load_game_cache
from Configuration.ppo_params import PPOparameters

N_AGENTS = 16
STATE_DIM = 1
HIDDEN_DIM = 128

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def _constant_state() -> th.Tensor:
    return th.tensor([1.0], dtype=th.float32, device=device)


def _agent_id(a: int) -> th.Tensor:
    return F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)


# ── Data collection ────────────────────────────────────────────────────────────

def _collect_batch(actor, critic, game, batch_size):
    """Collect batch_size single-step episodes under the current policy."""
    state = _constant_state()
    all_actions, all_log_probs, all_values, all_rewards = [], [], [], []

    with th.no_grad():
        for _ in range(batch_size):
            actions, log_probs, values = [], [], []
            for a in range(N_AGENTS):
                logits = actor(state, _agent_id(a))
                v = critic(state, _agent_id(a)).squeeze()
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                values.append(v)
                actions.append(action.item())

            reward = game.reward(actions)
            all_actions.append(actions)
            all_log_probs.append(th.stack(log_probs))
            all_values.append(th.stack(values))
            all_rewards.append(reward)

    return (
        th.tensor(all_actions, dtype=th.long, device=device),      # [B, N]
        th.stack(all_log_probs).to(device),                        # [B, N]
        th.stack(all_values).to(device),                           # [B, N]
        th.tensor(all_rewards, dtype=th.float32, device=device),   # [B]
    )


# ── PPO update (identical to run_matrix_game_ippo.py) ─────────────────────────

def _ppo_update(actor, critic, actor_opt, critic_opt,
                actions, log_probs_old, values_old, rewards,
                eps_clip, entropy_coef, epochs, num_mini_batches):
    B = rewards.shape[0]
    state = _constant_state()

    advantages = rewards.unsqueeze(1) - values_old.detach()        # [B, N]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns = rewards.detach()                                     # [B]

    mb_size = max(1, B // num_mini_batches)

    for _ in range(epochs):
        perm = th.randperm(B, device=device)
        for start in range(0, B, mb_size):
            idx = perm[start:start + mb_size]
            n = idx.shape[0]
            mb_act = actions[idx]
            mb_lp_old = log_probs_old[idx].detach()
            mb_adv = advantages[idx]
            mb_ret = returns[idx]

            state_mb = state.unsqueeze(0).expand(n, -1)

            total_actor_loss = th.zeros(1, device=device)
            total_critic_loss = th.zeros(1, device=device)

            for a in range(N_AGENTS):
                aid_mb = _agent_id(a).unsqueeze(0).expand(n, -1)
                logits = actor(state_mb, aid_mb)
                dist = Categorical(logits=logits)
                new_lp = dist.log_prob(mb_act[:, a])
                entropy = dist.entropy().mean()

                ratio = th.exp(new_lp - mb_lp_old[:, a])
                adv = mb_adv[:, a]
                surr1 = ratio * adv
                surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
                actor_loss = -th.min(surr1, surr2).mean() - entropy_coef * entropy
                total_actor_loss = total_actor_loss + actor_loss

                v_pred = critic(state_mb, aid_mb).squeeze(-1)
                critic_loss = (v_pred - mb_ret).pow(2).mean()
                total_critic_loss = total_critic_loss + critic_loss

            actor_opt.zero_grad()
            (total_actor_loss / N_AGENTS).backward()
            actor_opt.step()

            critic_opt.zero_grad()
            (total_critic_loss / N_AGENTS).backward()
            critic_opt.step()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _evaluate(actor, game, n_trials: int = 9):
    state = _constant_state()
    total, last_loads = 0.0, None
    with th.no_grad():
        for _ in range(n_trials):
            actions = []
            for a in range(N_AGENTS):
                logits = actor(state, _agent_id(a))
                action, _, _ = actor.action_sampler(logits)
                actions.append(action.item())
            total += game.reward(actions)
            last_loads = game.loads(actions)
    return total / n_trials, last_loads


def _deterministic_readout(actor, game):
    """a^L: per-agent argmax of logits (same rule as the homogeneous study)."""
    state = _constant_state()
    actions = []
    with th.no_grad():
        for a in range(N_AGENTS):
            logits = actor(state, _agent_id(a))
            actions.append(int(th.argmax(logits).item()))
    return actions, game.reward(actions), game.loads(actions)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IPPO (PS) on a fixed heterogeneous C-V2X matrix game."
    )
    parser.add_argument("--game_cache", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Total env episodes (default: ppo_params.training_episodes)")
    parser.add_argument("--entropy_coef", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=None, help="Actor lr")
    parser.add_argument("--beta", type=float, default=None, help="Critic lr")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--eps_clip", type=float, default=None)
    parser.add_argument("--three_power", action="store_true",
                        help="3 power levels x 4 SC + NT = 13 actions")
    parser.add_argument("--out_dir", type=str, default="Results/Hetero_batch/curves")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    game, meta = load_game_cache(args.game_cache, three_power=args.three_power)
    density, snapshot_id = meta["density"], meta["snapshot_id"]
    R_star = meta["R_star"]
    mode = "3P" if args.three_power else "1P"

    pp = PPOparameters()
    episodes     = args.episodes     if args.episodes     is not None else pp.training_episodes
    entropy_coef = args.entropy_coef if args.entropy_coef is not None else pp.entropy_coef
    alpha        = args.alpha        if args.alpha        is not None else pp.alpha
    beta         = args.beta         if args.beta         is not None else pp.beta
    batch_size   = args.batch_size   if args.batch_size   is not None else pp.batch_size
    epochs       = args.epochs       if args.epochs       is not None else pp.epochs
    eps_clip     = args.eps_clip     if args.eps_clip     is not None else pp.eps_clip

    print(f"=== Hetero matrix game (IPPO-PS, {mode})  density={density} "
          f"snapshot={snapshot_id}  seed={args.seed} ===")
    print(f"  snapshot proxy: beta={meta['beta']:.3f} dB  gamma={meta['gamma']:.3f} dB  "
          f"(ensemble mean {meta['beta_mean']:.3f}, {meta['gamma_mean']:.3f})")
    print(f"  n_actions={game.n_actions}  powers={game.pw_dbm} dBm")
    print(f"  R*_1P = {R_star:.4f}  (subset DP, exact)")
    print(f"  entropy_coef={entropy_coef}  episodes={episodes}  batch={batch_size}  "
          f"epochs={epochs}  seed={args.seed}")
    print(f"  actor_lr={alpha}  critic_lr={beta}  eps_clip={eps_clip}\n")

    p_ns = SimpleNamespace(n_agent=N_AGENTS, critic_hidden_dim=128, value_dim=1)
    actor = PPOSharedActor(STATE_DIM, game.n_actions, HIDDEN_DIM, N_AGENTS).to(device)
    critic = PPOSharedCritic(STATE_DIM, p_ns).to(device)
    actor_opt = th.optim.Adam(actor.parameters(), lr=alpha)
    critic_opt = th.optim.Adam(critic.parameters(), lr=beta)

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = (f"IPPO_HET_{density}_snap{snapshot_id}_{mode}"
                f"_ep{episodes}_seed{args.seed}_{ts}.csv")
    csv_path = os.path.join(args.out_dir, csv_name)

    num_updates = episodes // batch_size
    test_interval = max(1, num_updates // 100)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "test_reward", "pct_Rstar_1P", "loads"])
        f.flush()

        episode = 0
        for upd in range(num_updates):
            actions, log_probs_old, values_old, rewards = _collect_batch(
                actor, critic, game, batch_size)
            _ppo_update(actor, critic, actor_opt, critic_opt,
                        actions, log_probs_old, values_old, rewards,
                        eps_clip, entropy_coef, epochs, 4)
            episode += batch_size

            if upd % test_interval == 0 or upd == num_updates - 1:
                test_r, loads = _evaluate(actor, game)
                pct = test_r / R_star * 100 if R_star > 0 else float("nan")
                writer.writerow([episode, f"{test_r:.4f}", f"{pct:.1f}",
                                 loads.tolist()])
                f.flush()
                print(f"  ep {episode:>6}/{episodes}: reward={test_r:.3f} "
                      f"({pct:.1f}% R*_1P)  loads={loads.tolist()}")

    # ── Final: stochastic Phi^L AND deterministic readout a^L ─────────────────
    final_r, _ = _evaluate(actor, game)
    actions, phi_rd, loads_rd = _deterministic_readout(actor, game)
    eps, is_pne, _, best_dev = game.epsilon_pne(actions)
    ratio = phi_rd / R_star if R_star > 0 else float("nan")
    eps_norm = eps / R_star if R_star > 0 else float("nan")

    print(f"\n=== Final (hetero IPPO-PS, {mode}, density={density}, "
          f"snap={snapshot_id}, seed={args.seed}) ===")
    print(f"  [stochastic] Phi = {final_r:.6f}")
    print(f"  READOUT_ACTIONS={actions}")
    print(f"  Phi={phi_rd:.6f}  R_star_1P={R_star:.6f}  ratio={ratio:.6f}")
    print(f"  PNE?={is_pne}  eps={eps:.9f}  eps_norm={eps_norm:.9f}")
    if best_dev is not None:
        print(f"  best_dev=agent{best_dev[0]}:{best_dev[1]}->{best_dev[2]}")
    print(f"  loads={loads_rd.tolist()}")
    print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
