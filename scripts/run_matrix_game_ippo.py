"""
run_matrix_game_ippo.py — IPPO on homogeneous C-V2X matrix games (no dataset).

Game: N=16 agents, M=4 subchannels, action ∈ {0,1,2,3=SC, 4=NT}.
Global reward: total SE across all active agents (shared by all).
Exploration via entropy bonus (entropy_coef) instead of ε-greedy.

Supports parameter sharing (default) and independent networks (--no_sharing).

Usage:
  python run_matrix_game_ippo.py --op_beta 13.2932 --op_gamma 74.5837 --seed 9 --episodes 50000
  python run_matrix_game_ippo.py --op_beta 13.2932 --op_gamma 74.5837 --seed 9 --no_sharing
"""

import argparse
import csv
import math
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

from Networks.Agents.ppo_actor import PPOSharedActor, PPOActorNS
from Networks.Critics.ppo_critic import PPOSharedCritic, PPOCriticNS
from analysis.paper_utils import SIGMA2_MW, classify_4, find_opt_and_ne, n_plus_exact, R_ch
from Configuration.ppo_params import PPOparameters

OPERATING_POINTS = {
     35: (13.50, 74.45),
     62: (13.30, 76.33),
    123: (12.71, 79.03),
    167: (11.94, 79.96),
    333: (10.89, 82.06),
    500: (10.19, 83.43),
}

N_AGENTS   = 16
N_SC       = 4
ACTION_DIM = N_SC + 1
STATE_DIM  = 1
HIDDEN_DIM = 128

# Canonical Regime-III PNE load vectors (sorted desc), N=16 M=4
CANON = {
    (1, 1, 1, 1):  'v0',
    (13, 1, 1, 1): 'v1',
    (7, 7, 1, 1):  'v2',
    (5, 5, 5, 1):  'v3',
    (4, 4, 4, 4):  'v4',
}

device = th.device("cuda" if th.cuda.is_available() else "cpu")


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_SI(beta_db: float, gamma_db: float):
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    return S, I


def _constant_state() -> th.Tensor:
    return th.tensor([1.0], dtype=th.float32, device=device)


def _compute_reward(actions, S: float, I: float):
    loads = np.zeros(N_SC, dtype=int)
    for a in actions:
        if a < N_SC:
            loads[a] += 1
    total = 0.0
    for a in actions:
        if a < N_SC:
            n_m = loads[a]
            total += math.log2(1.0 + S / (SIGMA2_MW + (n_m - 1) * I))
    return total, loads


# ── Data collection ────────────────────────────────────────────────────────────

def _collect_batch(actor_or_list, critic_or_list, S, I, batch_size, no_sharing):
    """Collect batch_size single-step episodes under current stochastic policy."""
    state = _constant_state()
    all_actions   = []
    all_log_probs = []
    all_values    = []
    all_rewards   = []

    with th.no_grad():
        for _ in range(batch_size):
            actions, log_probs, values = [], [], []
            for a in range(N_AGENTS):
                if no_sharing:
                    logits = actor_or_list[a](state)
                    v = critic_or_list[a](state).squeeze()
                else:
                    agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
                    logits = actor_or_list(state, agent_id)
                    v = critic_or_list(state, agent_id).squeeze()

                dist = Categorical(logits=logits)
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                values.append(v)
                actions.append(action.item())

            reward, _ = _compute_reward(actions, S, I)
            all_actions.append(actions)
            all_log_probs.append(th.stack(log_probs))
            all_values.append(th.stack(values))
            all_rewards.append(reward)

    return (
        th.tensor(all_actions,  dtype=th.long,    device=device),   # [B, N]
        th.stack(all_log_probs).to(device),                          # [B, N]
        th.stack(all_values).to(device),                             # [B, N]
        th.tensor(all_rewards,  dtype=th.float32, device=device),    # [B]
    )


# ── PPO update ─────────────────────────────────────────────────────────────────

def _ppo_update(actor_or_list, critic_or_list, actor_opt_or_list, critic_opt_or_list,
                actions, log_probs_old, values_old, rewards,
                eps_clip, entropy_coef, epochs, num_mini_batches, no_sharing):
    B = rewards.shape[0]
    state = _constant_state()

    # Single-step GAE: A_a = r - V_a(s),  return = r  (done=True → no bootstrap)
    advantages = rewards.unsqueeze(1) - values_old.detach()        # [B, N]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns    = rewards.detach()                                   # [B]

    mb_size = max(1, B // num_mini_batches)

    for _ in range(epochs):
        perm = th.randperm(B, device=device)
        for start in range(0, B, mb_size):
            idx = perm[start:start + mb_size]
            n   = idx.shape[0]
            mb_act     = actions[idx]                    # [mb, N]
            mb_lp_old  = log_probs_old[idx].detach()    # [mb, N]
            mb_adv     = advantages[idx]                 # [mb, N]
            mb_ret     = returns[idx]                    # [mb]

            state_mb = state.unsqueeze(0).expand(n, -1)  # [mb, STATE_DIM]

            if no_sharing:
                # Independent: each agent has own network and optimizer
                for a in range(N_AGENTS):
                    # Actor
                    logits  = actor_or_list[a](state_mb)
                    dist    = Categorical(logits=logits)
                    new_lp  = dist.log_prob(mb_act[:, a])
                    entropy = dist.entropy().mean()

                    ratio = th.exp(new_lp - mb_lp_old[:, a])
                    adv   = mb_adv[:, a]
                    surr1 = ratio * adv
                    surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
                    actor_loss = -th.min(surr1, surr2).mean() - entropy_coef * entropy

                    actor_opt_or_list[a].zero_grad()
                    actor_loss.backward()
                    actor_opt_or_list[a].step()

                    # Critic
                    v_pred      = critic_or_list[a](state_mb).squeeze(-1)
                    critic_loss = (v_pred - mb_ret).pow(2).mean()

                    critic_opt_or_list[a].zero_grad()
                    critic_loss.backward()
                    critic_opt_or_list[a].step()

            else:
                # Shared: single network, accumulate loss across agents
                total_actor_loss  = th.zeros(1, device=device)
                total_critic_loss = th.zeros(1, device=device)

                for a in range(N_AGENTS):
                    agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
                    aid_mb   = agent_id.unsqueeze(0).expand(n, -1)

                    # Actor loss (PPO-clip + entropy)
                    logits  = actor_or_list(state_mb, aid_mb)
                    dist    = Categorical(logits=logits)
                    new_lp  = dist.log_prob(mb_act[:, a])
                    entropy = dist.entropy().mean()

                    ratio = th.exp(new_lp - mb_lp_old[:, a])
                    adv   = mb_adv[:, a]
                    surr1 = ratio * adv
                    surr2 = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
                    actor_loss = -th.min(surr1, surr2).mean() - entropy_coef * entropy
                    total_actor_loss = total_actor_loss + actor_loss

                    # Critic loss (MSE vs single-step return)
                    v_pred      = critic_or_list(state_mb, aid_mb).squeeze(-1)
                    critic_loss = (v_pred - mb_ret).pow(2).mean()
                    total_critic_loss = total_critic_loss + critic_loss

                actor_opt_or_list.zero_grad()
                (total_actor_loss / N_AGENTS).backward()
                actor_opt_or_list.step()

                critic_opt_or_list.zero_grad()
                (total_critic_loss / N_AGENTS).backward()
                critic_opt_or_list.step()


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _evaluate(actor_or_list, S, I, n_trials: int = 9, no_sharing: bool = False):
    """Stochastic evaluation averaged over n_trials."""
    state = _constant_state()
    total = 0.0
    last_loads = None
    with th.no_grad():
        for _ in range(n_trials):
            actions = []
            for a in range(N_AGENTS):
                if no_sharing:
                    logits = actor_or_list[a](state)
                else:
                    agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
                    logits = actor_or_list(state, agent_id)
                action, _, _ = actor_or_list[a].action_sampler(logits) if no_sharing \
                    else actor_or_list.action_sampler(logits)
                actions.append(action.item())
            r, loads = _compute_reward(actions, S, I)
            total += r
            last_loads = loads
    return total / n_trials, actions, last_loads


def _deterministic_readout(actor_or_list, S, I, no_sharing: bool = False):
    """Deterministic readout: argmax logits per agent."""
    state = _constant_state()
    loads = np.zeros(N_SC, dtype=int)
    actions = []
    with th.no_grad():
        for a in range(N_AGENTS):
            if no_sharing:
                logits = actor_or_list[a](state)
            else:
                agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
                logits = actor_or_list(state, agent_id)
            actions.append(int(th.argmax(logits).item()))
    for a in actions:
        if a < N_SC:
            loads[a] += 1
    phi = sum(R_ch(n, S, I, SIGMA2_MW) for n in loads)
    return phi, actions, loads


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IPPO on homogeneous C-V2X matrix game (no dataset)."
    )
    parser.add_argument("--density",      type=int,   default=None,
                        choices=[35, 62, 123, 167, 333, 500],
                        help="ETSI vehicle density [veh/km] (or use --op_beta/--op_gamma)")
    parser.add_argument("--op_beta",      type=float, default=None,
                        help="Operating-point beta [dB] (full precision); overrides --density")
    parser.add_argument("--op_gamma",     type=float, default=None,
                        help="Operating-point gamma [dB] (full precision); overrides --density")
    parser.add_argument("--seed",         type=int, default=0)
    parser.add_argument("--episodes",     type=int,   default=None,
                        help="Total env episodes (default: ppo_params.training_episodes)")
    parser.add_argument("--entropy_coef", type=float, default=None,
                        help="Entropy bonus (default: ppo_params.entropy_coef)")
    parser.add_argument("--alpha",        type=float, default=None, help="Actor lr")
    parser.add_argument("--beta",         type=float, default=None, help="Critic lr")
    parser.add_argument("--batch_size",   type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--eps_clip",     type=float, default=None)
    parser.add_argument("--no_sharing",   action="store_true",
                        help="Use independent networks per agent (no parameter sharing)")
    args = parser.parse_args()

    if args.op_beta is not None and args.op_gamma is not None:
        beta_db, gamma_db = args.op_beta, args.op_gamma
    elif args.density is not None:
        beta_db, gamma_db = OPERATING_POINTS[args.density]
    else:
        parser.error("must give either --density or both --op_beta and --op_gamma")

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    # ── Load PPO params, CLI overrides ─────────────────────────────────────────
    pp = PPOparameters()
    episodes     = args.episodes     if args.episodes     is not None else pp.training_episodes
    entropy_coef = args.entropy_coef if args.entropy_coef is not None else pp.entropy_coef
    alpha        = args.alpha        if args.alpha        is not None else pp.alpha
    beta         = args.beta         if args.beta         is not None else pp.beta
    batch_size   = args.batch_size   if args.batch_size   is not None else pp.batch_size
    epochs       = args.epochs       if args.epochs       is not None else pp.epochs
    eps_clip     = args.eps_clip     if args.eps_clip     is not None else pp.eps_clip
    no_sharing   = args.no_sharing

    S, I = _get_SI(beta_db, gamma_db)

    # ── Theory bounds ──────────────────────────────────────────────────────────
    opt_dist, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
    worst_NE_R = ne_list[0][1]  if ne_list else float("nan")
    best_NE_R  = ne_list[-1][1] if ne_list else float("nan")
    poa        = opt_R / worst_NE_R if (ne_list and worst_NE_R > 0) else float("nan")
    regime     = classify_4(S, I, SIGMA2_MW)
    np_val     = n_plus_exact(S, I, SIGMA2_MW)

    sharing_tag = "NS" if no_sharing else "PS"
    print(f"=== IPPO Matrix game ({sharing_tag})  λ={args.density} veh/km ===")
    print(f"  β={beta_db:.2f} dB   γ={gamma_db:.2f} dB   regime={regime}   n⁺={np_val}")
    print(f"  OPT={opt_R:.4f}  best_NE={best_NE_R:.4f}  worst_NE={worst_NE_R:.4f}  PoA={poa:.4f}")
    print(f"  entropy_coef={entropy_coef}  episodes={episodes}  batch={batch_size}  epochs={epochs}  seed={args.seed}")
    print(f"  actor_lr={alpha}  critic_lr={beta}  eps_clip={eps_clip}  no_sharing={no_sharing}\n")

    # ── Networks ───────────────────────────────────────────────────────────────
    p_ns = SimpleNamespace(n_agent=N_AGENTS, critic_hidden_dim=128, value_dim=1)

    if no_sharing:
        actors  = [PPOActorNS(STATE_DIM, ACTION_DIM, HIDDEN_DIM).to(device) for _ in range(N_AGENTS)]
        critics = [PPOCriticNS(STATE_DIM, HIDDEN_DIM, 1).to(device) for _ in range(N_AGENTS)]
        actor_opts  = [th.optim.Adam(actors[a].parameters(),  lr=alpha) for a in range(N_AGENTS)]
        critic_opts = [th.optim.Adam(critics[a].parameters(), lr=beta)  for a in range(N_AGENTS)]
    else:
        actor  = PPOSharedActor(STATE_DIM, ACTION_DIM, HIDDEN_DIM, N_AGENTS).to(device)
        critic = PPOSharedCritic(STATE_DIM, p_ns).to(device)
        actor_opt  = th.optim.Adam(actor.parameters(),  lr=alpha)
        critic_opt = th.optim.Adam(critic.parameters(), lr=beta)

    # ── Output CSV ─────────────────────────────────────────────────────────────
    out_dir  = os.path.join("Results", "IPPO_MatrixGame")
    os.makedirs(out_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"lam{args.density}" if args.density is not None else f"b{beta_db:.4f}_g{gamma_db:.4f}"
    csv_name = (f"IPPO_MG_{tag}_ep{episodes}"
                f"_ec{entropy_coef}_{sharing_tag}_seed{args.seed}_{ts}.csv")
    csv_path = os.path.join(out_dir, csv_name)

    num_updates   = episodes // batch_size
    test_interval = max(1, num_updates // 100)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "test_reward", "pct_opt", "pct_best_ne", "loads"])
        f.flush()

        episode = 0
        for upd in range(num_updates):
            if no_sharing:
                actions, log_probs_old, values_old, rewards = _collect_batch(
                    actors, critics, S, I, batch_size, no_sharing=True)
                _ppo_update(
                    actors, critics, actor_opts, critic_opts,
                    actions, log_probs_old, values_old, rewards,
                    eps_clip, entropy_coef, epochs, 4, no_sharing=True)
            else:
                actions, log_probs_old, values_old, rewards = _collect_batch(
                    actor, critic, S, I, batch_size, no_sharing=False)
                _ppo_update(
                    actor, critic, actor_opt, critic_opt,
                    actions, log_probs_old, values_old, rewards,
                    eps_clip, entropy_coef, epochs, 4, no_sharing=False)

            episode += batch_size

            if upd % test_interval == 0:
                if no_sharing:
                    test_r, _, loads = _evaluate(actors, S, I, no_sharing=True)
                else:
                    test_r, _, loads = _evaluate(actor, S, I, no_sharing=False)
                pct_opt     = test_r / opt_R     * 100 if opt_R     > 0 else float("nan")
                pct_best_ne = test_r / best_NE_R * 100 if best_NE_R > 0 else float("nan")
                writer.writerow([episode, f"{test_r:.4f}", f"{pct_opt:.1f}",
                                 f"{pct_best_ne:.1f}", loads.tolist()])
                f.flush()
                print(f"  ep {episode:>6}/{episodes}: "
                      f"reward={test_r:.3f} ({pct_opt:.1f}% OPT, {pct_best_ne:.1f}% best_NE)  "
                      f"loads={loads.tolist()}")

    # ── Final: stochastic Phi^L  AND  deterministic readout Phi(n^L) ────────────
    if no_sharing:
        final_r, _, _ = _evaluate(actors, S, I, no_sharing=True)
        phi_rd, actions_rd, loads_rd = _deterministic_readout(actors, S, I, no_sharing=True)
    else:
        final_r, _, _ = _evaluate(actor, S, I, no_sharing=False)
        phi_rd, actions_rd, loads_rd = _deterministic_readout(actor, S, I, no_sharing=False)

    pct_opt     = final_r / opt_R     * 100 if opt_R     > 0 else float("nan")
    pct_best_ne = final_r / best_NE_R * 100 if best_NE_R > 0 else float("nan")

    readout = tuple(sorted(loads_rd.tolist(), reverse=True))
    pne_set = {d for d, _ in ne_list}
    is_pne  = readout in pne_set
    rho     = ((opt_R - phi_rd) / (opt_R - worst_NE_R)
               if is_pne and abs(opt_R - worst_NE_R) > 1e-15 else float("nan"))
    label   = CANON.get(readout, "non-PNE")

    print(f"\n=== Final ({sharing_tag}) ===")
    print(f"  [stochastic]  Phi^L     = {final_r:.4f}   %OPT={pct_opt:.1f}%  %best_NE={pct_best_ne:.1f}%")
    print(f"  [readout]     Phi(n^L) = {phi_rd:.4f}   loads={loads_rd.tolist()}   sorted={readout}")
    print(f"  [readout]     PNE?={is_pne}   label={label}   rho={rho:.4f}" if is_pne
          else f"  [readout]     PNE?={is_pne}   (readout not in equilibrium set)")
    print(f"  OPT={opt_R:.4f}  best_NE={best_NE_R:.4f}  worst_NE={worst_NE_R:.4f}  PoA={poa:.4f}")
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
