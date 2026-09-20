"""
run_matrix_game_ippo_diag.py — IPPO matrix game with lean diagnostic logs.
Training logic identical to run_matrix_game_ippo.py; adds I/O only.

Outputs (Results/IPPO_MatrixGame_Diag/<run_tag>/):
  main.csv         — episode, test_reward, pct_opt, pct_best_ne, loads,
                     n_explore, n_multi_explore   (78 rows)
  flips_ippo.csv   — update_step, episode, agent_id, old_action, new_action, tie
  values_ippo.csv  — episode, agent_id, v_ch1..v_silence, l_ch1..l_silence
                     (ep<8000: every ~2 updates; ep≥8000: every update; + flip-triggered)
  eval_ippo.csv    — episode, sample_idx, a_1..a_16, reward  (last 10 checkpoints × 9)
  meta_ippo.json   — formula params, implementation answers

Usage:
  python run_matrix_game_ippo_diag.py --density 333 --seed 1 --episodes 20000 --entropy_coef 0.001
"""

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from Networks.Agents.ppo_actor import PPOSharedActor
from Networks.Critics.ppo_critic import PPOSharedCritic
from analysis.paper_utils import SIGMA2_MW, classify_4, find_opt_and_ne, n_plus_exact

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

device = th.device("cuda" if th.cuda.is_available() else "cpu")


def _get_SI(beta_db, gamma_db):
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    return S, I


def _constant_state():
    return th.tensor([1.0], dtype=th.float32, device=device)


def _compute_reward(actions, S, I):
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


def _collect_batch(actor, critic, S, I, batch_size):
    """Returns (actions[B,N], log_probs[B,N], values[B,N], rewards[B])."""
    state = _constant_state()
    all_actions, all_log_probs, all_values, all_rewards = [], [], [], []
    with th.no_grad():
        for _ in range(batch_size):
            actions, log_probs, values = [], [], []
            for a in range(N_AGENTS):
                agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
                logits   = actor(state, agent_id)
                dist     = Categorical(logits=logits)
                action   = dist.sample()
                log_probs.append(dist.log_prob(action))
                values.append(critic(state, agent_id).squeeze())
                actions.append(action.item())
            reward, _ = _compute_reward(actions, S, I)
            all_actions.append(actions)
            all_log_probs.append(th.stack(log_probs))
            all_values.append(th.stack(values))
            all_rewards.append(reward)
    return (
        th.tensor(all_actions,  dtype=th.long,    device=device),
        th.stack(all_log_probs).to(device),
        th.stack(all_values).to(device),
        th.tensor(all_rewards,  dtype=th.float32, device=device),
    )


def _ppo_update(actor, critic, actor_opt, critic_opt,
                actions, log_probs_old, values_old, rewards,
                eps_clip, entropy_coef, epochs, num_mini_batches):
    B     = rewards.shape[0]
    state = _constant_state()

    advantages = rewards.unsqueeze(1) - values_old.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns    = rewards.detach()
    mb_size    = max(1, B // num_mini_batches)

    for _ in range(epochs):
        perm = th.randperm(B, device=device)
        for start in range(0, B, mb_size):
            idx       = perm[start:start + mb_size]
            n         = idx.shape[0]
            mb_act    = actions[idx]
            mb_lp_old = log_probs_old[idx].detach()
            mb_adv    = advantages[idx]
            mb_ret    = returns[idx]
            state_mb  = state.unsqueeze(0).expand(n, -1)

            total_actor_loss  = th.zeros(1, device=device)
            total_critic_loss = th.zeros(1, device=device)

            for a in range(N_AGENTS):
                agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
                aid_mb   = agent_id.unsqueeze(0).expand(n, -1)
                logits   = actor(state_mb, aid_mb)
                dist     = Categorical(logits=logits)
                new_lp   = dist.log_prob(mb_act[:, a])
                entropy  = dist.entropy().mean()
                ratio    = th.exp(new_lp - mb_lp_old[:, a])
                adv      = mb_adv[:, a]
                surr1    = ratio * adv
                surr2    = th.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * adv
                actor_loss  = -th.min(surr1, surr2).mean() - entropy_coef * entropy
                total_actor_loss = total_actor_loss + actor_loss

                v_pred      = critic(state_mb, aid_mb).squeeze(-1)
                critic_loss = (v_pred - mb_ret).pow(2).mean()
                total_critic_loss = total_critic_loss + critic_loss

            actor_opt.zero_grad()
            (total_actor_loss / N_AGENTS).backward()
            actor_opt.step()

            critic_opt.zero_grad()
            (total_critic_loss / N_AGENTS).backward()
            critic_opt.step()


def _evaluate_diag(actor, S, I, n_trials=9, eval_writer=None, episode=None,
                   write_eval=False):
    """
    Stochastic evaluation (matching original NFIG IPPOtester).
    If write_eval=True, logs all n_trials rows to eval_writer.
    Returns (avg_reward, last_actions, last_loads).
    """
    state = _constant_state()
    total = 0.0
    last_loads   = None
    last_actions = None
    with th.no_grad():
        for trial in range(n_trials):
            actions = []
            for a in range(N_AGENTS):
                agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
                logits   = actor(state, agent_id)
                action, _, _ = actor.action_sampler(logits)
                actions.append(action.item())
            r, loads = _compute_reward(actions, S, I)
            total += r
            last_loads   = loads
            last_actions = actions
            if write_eval and eval_writer is not None and episode is not None:
                eval_writer.writerow([episode, trial] + actions + [f"{r:.4f}"])
    return total / n_trials, last_actions, last_loads


def _get_all_greedy_ippo(actor):
    """
    Returns:
      greedy:     list[int]         argmax logit per agent
      ties:       list[int]         1 if two or more logits share max
      logit_vecs: list[np.ndarray]  raw logits, shape [5]
      prob_vecs:  list[np.ndarray]  softmax probs, shape [5]
    """
    state = _constant_state()
    greedy, ties, logit_vecs, prob_vecs = [], [], [], []
    with th.no_grad():
        for a in range(N_AGENTS):
            agent_id = F.one_hot(th.tensor(a), num_classes=N_AGENTS).float().to(device)
            logits   = actor(state, agent_id)
            l_np     = logits.cpu().numpy()
            p_np     = th.softmax(logits, dim=-1).cpu().numpy()
            best     = int(np.argmax(l_np))
            tied     = int((l_np == l_np[best]).sum() > 1)
            greedy.append(best)
            ties.append(tied)
            logit_vecs.append(l_np.copy())
            prob_vecs.append(p_np.copy())
    return greedy, ties, logit_vecs, prob_vecs


def _greedy_load(greedy):
    load = np.zeros(N_SC, dtype=int)
    for g in greedy:
        if g < N_SC:
            load[g] += 1
    return load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density",      type=int,   required=True,
                        choices=[35, 62, 123, 167, 333, 500])
    parser.add_argument("--seed",         type=int,   default=1)
    parser.add_argument("--episodes",     type=int,   default=20000)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--alpha",        type=float, default=4e-4)
    parser.add_argument("--beta",         type=float, default=6e-4)
    parser.add_argument("--batch_size",   type=int,   default=256)
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--eps_clip",     type=float, default=0.2)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    beta_db, gamma_db = OPERATING_POINTS[args.density]
    S, I = _get_SI(beta_db, gamma_db)

    opt_dist, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
    worst_NE_R = ne_list[0][1]  if ne_list else float("nan")
    best_NE_R  = ne_list[-1][1] if ne_list else float("nan")
    poa        = opt_R / worst_NE_R if (ne_list and worst_NE_R > 0) else float("nan")

    print(f"=== IPPO Diag  λ={args.density}  seed={args.seed}  "
          f"ec={args.entropy_coef}  episodes={args.episodes} ===")
    print(f"  OPT={opt_R:.4f}  best_NE={best_NE_R:.4f}  "
          f"worst_NE={worst_NE_R:.4f}  PoA={poa:.4f}\n")

    p_ns   = SimpleNamespace(n_agent=N_AGENTS, critic_hidden_dim=128, value_dim=1)
    actor  = PPOSharedActor(STATE_DIM, ACTION_DIM, 128, N_AGENTS).to(device)
    critic = PPOSharedCritic(STATE_DIM, p_ns).to(device)
    actor_opt  = th.optim.Adam(actor.parameters(),  lr=args.alpha)
    critic_opt = th.optim.Adam(critic.parameters(), lr=args.beta)

    num_updates       = args.episodes // args.batch_size
    test_interval     = max(1, num_updates // 100)
    total_checkpoints = num_updates  # test every update (test_interval=1 for 78 updates)

    # ── Output directory ────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = (f"lam{args.density}_ep{args.episodes}"
               f"_ec{args.entropy_coef}_seed{args.seed}_{ts}")
    out_dir = os.path.join("Results", "IPPO_MatrixGame_Diag", run_tag)
    os.makedirs(out_dir, exist_ok=True)

    # ── Open files ──────────────────────────────────────────────────────────────
    main_f   = open(os.path.join(out_dir, "main.csv"),          "w", newline="")
    flips_f  = open(os.path.join(out_dir, "flips_ippo.csv"),    "w", newline="")
    values_f = open(os.path.join(out_dir, "values_ippo.csv"),   "w", newline="")
    eval_f   = open(os.path.join(out_dir, "eval_ippo.csv"),     "w", newline="")

    main_w   = csv.writer(main_f)
    flips_w  = csv.writer(flips_f)
    values_w = csv.writer(values_f)
    eval_w   = csv.writer(eval_f)

    main_w.writerow(["episode", "test_reward", "pct_opt", "pct_best_ne",
                     "loads", "n_explore", "n_multi_explore"])
    flips_w.writerow(["update_step", "episode", "agent_id",
                      "old_action", "new_action", "tie"])
    values_w.writerow(
        ["episode", "agent_id",
         "v_ch1", "v_ch2", "v_ch3", "v_ch4", "v_silence",
         "l_ch1", "l_ch2", "l_ch3", "l_ch4", "l_silence"]
    )
    eval_w.writerow(
        ["episode", "sample_idx"] + [f"a_{i+1}" for i in range(N_AGENTS)] + ["reward"]
    )

    for f in [main_f, flips_f, values_f, eval_f]:
        f.flush()

    # ── Initial greedy snapshot ─────────────────────────────────────────────────
    prev_greedy, _, init_logit_vecs, init_prob_vecs = _get_all_greedy_ippo(actor)
    initial_greedy = prev_greedy[:]

    # Log initial values snapshot (ep=-1 / before training)
    for i in range(N_AGENTS):
        values_w.writerow([-1, i]
                          + init_prob_vecs[i].tolist()
                          + init_logit_vecs[i].tolist())
    values_f.flush()

    # ── Bookkeeping ─────────────────────────────────────────────────────────────
    all_flip_counts = []
    verify_rows     = []   # (episode, greedy_load, eval_last_load)
    first_flip_upd  = None

    # ── Training loop ───────────────────────────────────────────────────────────
    episode = 0
    for upd in range(num_updates):
        # Greedy reference before this batch's update
        greedy_ref = prev_greedy[:]

        batch_actions, log_probs_old, values_old, rewards = _collect_batch(
            actor, critic, S, I, args.batch_size
        )
        episode += args.batch_size

        # Aggregate explore stats for this batch
        batch_np = batch_actions.cpu().numpy()  # [B, N]
        n_explore     = 0
        n_multi_explore = 0
        for b in range(args.batch_size):
            dev = sum(1 for i in range(N_AGENTS) if batch_np[b, i] != greedy_ref[i])
            n_explore += dev
            if dev >= 2:
                n_multi_explore += 1

        _ppo_update(
            actor, critic, actor_opt, critic_opt,
            batch_actions, log_probs_old, values_old, rewards,
            args.eps_clip, args.entropy_coef, args.epochs, 4,
        )

        curr_greedy, curr_ties, curr_logit_vecs, curr_prob_vecs = _get_all_greedy_ippo(actor)

        c = 0
        for i in range(N_AGENTS):
            if curr_greedy[i] != prev_greedy[i]:
                flips_w.writerow([upd, episode, i,
                                  prev_greedy[i], curr_greedy[i], curr_ties[i]])
                c += 1
        all_flip_counts.append(c)
        if c:
            flips_f.flush()
            if first_flip_upd is None:
                first_flip_upd = upd

        prev_greedy = curr_greedy[:]

        # Values: two-phase periodic + flip-triggered
        # ep<8000 → every ~2 updates; ep≥8000 → every update
        if episode < 8000:
            log_vals = (upd % 2 == 0) or (c > 0)
        else:
            log_vals = True  # every update in late phase
        if c > 0:
            log_vals = True  # always log on flip

        if log_vals:
            for i in range(N_AGENTS):
                values_w.writerow(
                    [episode, i]
                    + curr_prob_vecs[i].tolist()
                    + curr_logit_vecs[i].tolist()
                )
            values_f.flush()

        if upd % test_interval == 0:
            write_eval = (upd >= num_updates - 10)
            test_r, _, eval_loads = _evaluate_diag(
                actor, S, I,
                eval_writer=eval_w, episode=episode,
                write_eval=write_eval,
            )
            if write_eval:
                eval_f.flush()

            pct_opt     = test_r / opt_R     * 100 if opt_R     > 0 else float("nan")
            pct_best_ne = test_r / best_NE_R * 100 if best_NE_R > 0 else float("nan")
            gl = _greedy_load(curr_greedy)
            main_w.writerow([episode, f"{test_r:.4f}", f"{pct_opt:.1f}",
                             f"{pct_best_ne:.1f}", eval_loads.tolist(),
                             n_explore, n_multi_explore])
            main_f.flush()

            verify_rows.append((episode, gl.tolist(), eval_loads.tolist()))

            print(f"  ep {episode:>6}/{args.episodes}: "
                  f"reward={test_r:.3f} ({pct_opt:.1f}% OPT)  "
                  f"loads={eval_loads.tolist()}  "
                  f"greedy={gl.tolist()}  "
                  f"n_exp={n_explore}  n_multi={n_multi_explore}")

    for f in [main_f, flips_f, values_f, eval_f]:
        f.close()

    # ── Meta JSON ───────────────────────────────────────────────────────────────
    meta = {
        "density":      args.density,
        "seed":         args.seed,
        "episodes":     args.episodes,
        "entropy_coef": args.entropy_coef,
        "alpha":        args.alpha,
        "beta":         args.beta,
        "batch_size":   args.batch_size,
        "epochs":       args.epochs,
        "eps_clip":     args.eps_clip,
        "num_updates":  num_updates,
        "initial_greedy_actions": initial_greedy,
        "first_flip_update": first_flip_upd,
        "first_flip_episode": (first_flip_upd + 1) * args.batch_size if first_flip_upd is not None else None,
        "Q_answer_a_sync_update": (
            "All 16 agents share one actor and one critic network. "
            "Within each PPO update step, all 16 agents are optimized jointly: "
            "the inner loop iterates over N_AGENTS inside each mini-batch forward pass. "
            "One _ppo_update() call per batch_size=256 episodes."
        ),
        "Q_answer_b_warmup": (
            "No warmup: PPO has no replay buffer, so the first batch of 256 episodes "
            "immediately triggers a PPO update. First observed flip: "
            f"update_step={first_flip_upd}, "
            f"episode={(first_flip_upd + 1) * args.batch_size if first_flip_upd is not None else None}."
        ),
        "Q_answer_c_eval": (
            "Evaluation: stochastic sampling via actor.action_sampler (Categorical.sample). "
            "9 independent trials. reward = mean over 9 trials. "
            "loads column in main.csv = loads from the 9th (last) trial, NOT the mode. "
            "eval_ippo.csv stores all 9 trials for the last 10 checkpoints only. "
            "Greedy reference (flips) = argmax logit per agent, separate from eval."
        ),
    }
    with open(os.path.join(out_dir, "meta_ippo.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Verification & statistics ───────────────────────────────────────────────
    CONVERGENCE_UPD = int(9728 / args.batch_size)  # ≈ update 38
    post_flips = [c for upd, c in enumerate(all_flip_counts)
                  if upd * args.batch_size >= 9728]

    n_steps = len(all_flip_counts)
    p0 = sum(1 for c in all_flip_counts if c == 0) / n_steps
    p1 = sum(1 for c in all_flip_counts if c == 1) / n_steps
    p2 = sum(1 for c in all_flip_counts if c == 2) / n_steps
    p3 = sum(1 for c in all_flip_counts if c >= 3) / n_steps
    max_c_post = max(post_flips) if post_flips else 0

    print(f"\n=== Verification & Statistics ===")
    print(f"  Flip-count distribution (over {n_steps} PPO updates):")
    print(f"    P(c=0) = {p0:.4f}   P(c=1) = {p1:.4f}   "
          f"P(c=2) = {p2:.4f}   P(c≥3) = {p3:.4f}")
    print(f"  Max c after ep≥9728 (update≥{CONVERGENCE_UPD}): {max_c_post}")
    print(f"  First flip: update_step={first_flip_upd}")

    # Greedy-load vs eval-last-trial-load
    n_agree = sum(1 for (ep, gl, el) in verify_rows if gl == el)
    n_total = len(verify_rows)
    print(f"\n  Greedy-load (argmax logit) == eval-last-trial-load: "
          f"{n_agree}/{n_total} checkpoints")
    if n_agree < n_total:
        print("  → 'loads' column = last of 9 stochastic trials, not argmax logit.")
    for ep, gl, el in [(ep, gl, el) for (ep, gl, el) in verify_rows if gl != el][:3]:
        print(f"    ep={ep}: greedy={gl}  eval_last={el}")

    # ── Load-2 simultaneous departure detection ─────────────────────────────────
    flips_path = os.path.join(out_dir, "flips_ippo.csv")
    flip_by_upd = {}
    with open(flips_path) as f:
        for row in csv.DictReader(f):
            upd_idx = int(row["update_step"])
            if upd_idx not in flip_by_upd:
                flip_by_upd[upd_idx] = []
            flip_by_upd[upd_idx].append(
                (int(row["agent_id"]), int(row["old_action"]), int(row["new_action"]))
            )

    curr_recon = initial_greedy[:]
    load2_events = []

    for upd_idx in range(num_updates):
        flips_here = flip_by_upd.get(upd_idx, [])
        if flips_here:
            pre_load = np.zeros(N_SC, dtype=int)
            for g in curr_recon:
                if g < N_SC:
                    pre_load[g] += 1
            load2_scs = {sc for sc in range(N_SC) if pre_load[sc] == 2}
            depart_by_sc = {sc: [] for sc in load2_scs}
            for (aid, old_a, new_a) in flips_here:
                if old_a in load2_scs:
                    depart_by_sc[old_a].append(aid)
            for sc, agents in depart_by_sc.items():
                if len(agents) >= 2:
                    load2_events.append({
                        "update_step": upd_idx,
                        "episode":     (upd_idx + 1) * args.batch_size,
                        "sc":          sc,
                        "agents":      agents,
                        "pre_load":    pre_load.tolist(),
                    })
            for (aid, old_a, new_a) in flips_here:
                curr_recon[aid] = new_a

    print(f"\n  Load-2 simultaneous departure events: {len(load2_events)}")
    for ev in load2_events:
        print(f"    upd={ev['update_step']}  ep={ev['episode']}  "
              f"sc={ev['sc']}  agents={ev['agents']}  pre_load={ev['pre_load']}")
    if not load2_events:
        print("    None found — ratchet invariance holds empirically.")

    # ── Flips-reconstruction vs main.csv cross-check ────────────────────────────
    main_csv_rows = {}
    with open(os.path.join(out_dir, "main.csv")) as f:
        for row in csv.DictReader(f):
            main_csv_rows[int(row["episode"])] = row["loads"]

    curr_recon = initial_greedy[:]
    recon_match = 0
    recon_mismatch = 0
    for upd_idx in range(num_updates):
        for (aid, old_a, new_a) in flip_by_upd.get(upd_idx, []):
            curr_recon[aid] = new_a
        ep = (upd_idx + 1) * args.batch_size
        if ep in main_csv_rows:
            recon_load = _greedy_load(curr_recon).tolist()
            if str(recon_load) == main_csv_rows[ep]:
                recon_match += 1
            else:
                recon_mismatch += 1

    print(f"\n  Flips-reconstruction (argmax logit) == main.csv loads (last trial):")
    print(f"    Match: {recon_match}  Mismatch: {recon_mismatch}")
    if recon_mismatch > 0:
        print("    Confirmed: 'loads' = single stochastic trial, not argmax logit.")

    # ── Final ───────────────────────────────────────────────────────────────────
    final_r, _, final_loads = _evaluate_diag(actor, S, I)
    pct_opt = final_r / opt_R * 100 if opt_R > 0 else float("nan")
    print(f"\n=== Final stochastic eval ===  reward={final_r:.4f}  "
          f"({pct_opt:.1f}% OPT)  loads={final_loads.tolist()}")
    print(f"Saved → {out_dir}/")


if __name__ == "__main__":
    main()
