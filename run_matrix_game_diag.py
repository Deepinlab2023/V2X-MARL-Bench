"""
run_matrix_game_diag.py — IDQL matrix game with lean diagnostic logs.
Training logic identical to run_matrix_game.py; adds I/O only.

Outputs (Results/IDQL_MatrixGame_Diag/<run_tag>/):
  main.csv        — episode, test_reward, pct_opt, pct_best_ne, loads,
                    n_explore, n_multi_explore   (100 rows)
  flips_idql.csv  — update_step, episode, agent_id, old_action, new_action, tie
  values_idql.csv — episode, agent_id, v_ch1..v_silence
                    (ep<8000: every 500ep; ep≥8000: every 200ep; + flip-triggered)
  eval_idql.csv   — episode, a_1..a_16, reward   (last 10 checkpoints only)
  meta_idql.json  — formula params, implementation answers

Usage:
  python run_matrix_game_diag.py --density 333 --seed 1 --episodes 20000 --epsi_final 0.08
"""

import argparse
import csv
import json
import math
import os
import random
from datetime import datetime

import numpy as np
import torch as th

from Configuration.idql_params import IDQLparameters
from Networks.Agents.idql_agent import DQNAgent
from analysis.paper_utils import SIGMA2_MW, classify_4, find_opt_and_ne, n_plus_exact

OPERATING_POINTS = {
     35: (13.50, 74.45),
     62: (13.30, 76.33),
    123: (12.71, 79.03),
    167: (11.94, 79.96),
    333: (10.89, 82.06),
    500: (10.19, 83.43),
}

N_AGENTS  = 16
N_SC      = 4
STATE_DIM = 1


def _get_SI(beta_db, gamma_db):
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    return S, I


def _constant_state():
    return np.array([[1.0]], dtype=np.float32)


def _play_episode(S, I, agent_list, epsi):
    """Single training episode. Returns (total_reward, loads, actions)."""
    state = _constant_state()
    for ag in agent_list:
        ag.eps_threshold = epsi
    actions = [ag.select_action(state, None).item() for ag in agent_list]
    loads = np.zeros(N_SC, dtype=int)
    for a in actions:
        if a < N_SC:
            loads[a] += 1
    total = 0.0
    for a in actions:
        if a < N_SC:
            n_m = loads[a]
            total += math.log2(1.0 + S / (SIGMA2_MW + (n_m - 1) * I))
    global_r = np.array([[total]], dtype=np.float32)
    for i, ag in enumerate(agent_list):
        ag.store_transition(state, np.array([[actions[i]]]), state, True, global_r)
    return total, loads, actions


def _evaluate(S, I, agent_list):
    """Greedy (eps=0) evaluation. Returns (reward, actions, loads)."""
    state = _constant_state()
    for ag in agent_list:
        ag.eps_threshold = 0.0
    actions = [ag.select_action(state, None).item() for ag in agent_list]
    loads = np.zeros(N_SC, dtype=int)
    for a in actions:
        if a < N_SC:
            loads[a] += 1
    total = 0.0
    for a in actions:
        if a < N_SC:
            n_m = loads[a]
            total += math.log2(1.0 + S / (SIGMA2_MW + (n_m - 1) * I))
    return total, actions, loads


def _get_all_greedy(agent_list, state_np):
    """
    Returns:
      greedy: list[int]         argmax Q_i, ties → lowest index
      ties:   list[int]         1 if two or more actions share max Q
      qvecs:  list[np.ndarray]  full Q-vector shape [5]
    """
    state_t = th.tensor(state_np, dtype=th.float32)
    greedy, ties, qvecs = [], [], []
    for ag in agent_list:
        with th.no_grad():
            q = ag.q_net(state_t.to(ag.device)).squeeze(0).cpu().numpy()
        best = int(np.argmax(q))
        tied = int((q == q[best]).sum() > 1)
        greedy.append(best)
        ties.append(tied)
        qvecs.append(q.copy())
    return greedy, ties, qvecs


def _greedy_load(greedy):
    load = np.zeros(N_SC, dtype=int)
    for g in greedy:
        if g < N_SC:
            load[g] += 1
    return load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--density",    type=int,   required=True,
                        choices=[35, 62, 123, 167, 333, 500])
    parser.add_argument("--seed",       type=int,   default=1)
    parser.add_argument("--episodes",   type=int,   default=20000)
    parser.add_argument("--epsi_final", type=float, default=0.08)
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

    print(f"=== IDQL Diag  λ={args.density}  seed={args.seed}  "
          f"ef={args.epsi_final}  episodes={args.episodes} ===")
    print(f"  OPT={opt_R:.4f}  best_NE={best_NE_R:.4f}  "
          f"worst_NE={worst_NE_R:.4f}  PoA={poa:.4f}\n")

    algo = IDQLparameters()
    agent_list = [
        DQNAgent(
            ag_idx=i, num_agents=N_AGENTS, state_dim=STATE_DIM, action_dim=N_SC + 1,
            is_hysteretic_q=False, memory_capacity=algo.memory_capacity,
            batch_size=algo.batch_size, gamma=algo.gamma, tau=algo.tau, lr=algo.lr,
            hidden_dim=algo.hidden_dim, hysteretic_high_lr=algo.hysteretic_high_lr,
            hysteretic_low_lr=algo.hysteretic_low_lr, force_nt_when_empty=False,
        )
        for i in range(N_AGENTS)
    ]

    epsi_anneal   = int(0.8 * args.episodes)
    test_interval = max(1, args.episodes // 100)
    total_checkpoints = args.episodes // test_interval  # = 100

    # ── Output directory ────────────────────────────────────────────────────────
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_tag = (f"lam{args.density}_ep{args.episodes}"
               f"_ef{args.epsi_final}_seed{args.seed}_{ts}")
    out_dir = os.path.join("Results", "IDQL_MatrixGame_Diag", run_tag)
    os.makedirs(out_dir, exist_ok=True)

    # ── Open files ──────────────────────────────────────────────────────────────
    main_f   = open(os.path.join(out_dir, "main.csv"),         "w", newline="")
    flips_f  = open(os.path.join(out_dir, "flips_idql.csv"),   "w", newline="")
    values_f = open(os.path.join(out_dir, "values_idql.csv"),  "w", newline="")
    eval_f   = open(os.path.join(out_dir, "eval_idql.csv"),    "w", newline="")

    main_w   = csv.writer(main_f)
    flips_w  = csv.writer(flips_f)
    values_w = csv.writer(values_f)
    eval_w   = csv.writer(eval_f)

    main_w.writerow(["episode", "test_reward", "pct_opt", "pct_best_ne",
                     "loads", "n_explore", "n_multi_explore"])
    flips_w.writerow(["update_step", "episode", "agent_id",
                      "old_action", "new_action", "tie"])
    values_w.writerow(["episode", "agent_id",
                       "v_ch1", "v_ch2", "v_ch3", "v_ch4", "v_silence"])
    eval_w.writerow(["episode"] + [f"a_{i+1}" for i in range(N_AGENTS)] + ["reward"])

    for f in [main_f, flips_f, values_f, eval_f]:
        f.flush()

    # ── Initial greedy snapshot ─────────────────────────────────────────────────
    prev_greedy, _, init_qvecs = _get_all_greedy(agent_list, _constant_state())
    initial_greedy = prev_greedy[:]

    # Log initial values snapshot (ep=0, before training)
    for i in range(N_AGENTS):
        values_w.writerow([-1, i] + init_qvecs[i].tolist())
    values_f.flush()

    # ── Bookkeeping ─────────────────────────────────────────────────────────────
    all_flip_counts  = []
    verify_mismatches = []
    first_flip_ep    = None
    checkpoint_num   = 0

    # Window accumulators for n_explore / n_multi_explore (reset each checkpoint)
    window_explore = 0
    window_multi   = 0

    # ── Training loop ───────────────────────────────────────────────────────────
    for te in range(args.episodes):
        epsi = (1.0 - te * (1.0 - args.epsi_final) / max(epsi_anneal - 1, 1)
                if te < epsi_anneal else args.epsi_final)

        # Greedy reference before this episode's update
        greedy_ref = prev_greedy[:]

        _, _, ep_actions = _play_episode(S, I, agent_list, epsi)

        # Count deviations of actual actions from pre-episode greedy
        deviations = sum(1 for i in range(N_AGENTS)
                         if ep_actions[i] != greedy_ref[i])
        window_explore += deviations
        if deviations >= 2:
            window_multi += 1

        for ag in agent_list:
            ag.optimize_model()
            ag.soft_update_target_net()

        curr_greedy, curr_ties, curr_qvecs = _get_all_greedy(agent_list, _constant_state())

        c = 0
        for i in range(N_AGENTS):
            if curr_greedy[i] != prev_greedy[i]:
                flips_w.writerow([te, te, i,
                                  prev_greedy[i], curr_greedy[i], curr_ties[i]])
                c += 1
        all_flip_counts.append(c)
        if c:
            flips_f.flush()
            if first_flip_ep is None:
                first_flip_ep = te

        prev_greedy = curr_greedy[:]

        # Values: two-phase periodic + flip-triggered
        if te < 8000:
            log_vals = (te % 500 == 0) or (c > 0)
        else:
            log_vals = (te % 200 == 0) or (c > 0)
        if log_vals:
            for i in range(N_AGENTS):
                values_w.writerow([te, i] + curr_qvecs[i].tolist())
            values_f.flush()

        if te % test_interval == 0:
            test_r, eval_acts, eval_loads = _evaluate(S, I, agent_list)
            pct_opt     = test_r / opt_R     * 100 if opt_R     > 0 else float("nan")
            pct_best_ne = test_r / best_NE_R * 100 if best_NE_R > 0 else float("nan")
            main_w.writerow([te, f"{test_r:.4f}", f"{pct_opt:.1f}",
                             f"{pct_best_ne:.1f}", eval_loads.tolist(),
                             window_explore, window_multi])
            main_f.flush()

            # Last-10 checkpoints: write eval actions
            if checkpoint_num >= total_checkpoints - 10:
                eval_w.writerow([te] + eval_acts + [f"{test_r:.4f}"])
                eval_f.flush()
            checkpoint_num += 1

            # Verify greedy-load == eval-load
            gl = _greedy_load(curr_greedy)
            if not np.array_equal(gl, eval_loads):
                verify_mismatches.append((te, gl.tolist(), eval_loads.tolist()))

            print(f"  ep {te:>6}/{args.episodes}: "
                  f"reward={test_r:.3f} ({pct_opt:.1f}% OPT)  "
                  f"loads={eval_loads.tolist()}  "
                  f"n_exp={window_explore}  n_multi={window_multi}")

            # Reset window accumulators
            window_explore = 0
            window_multi   = 0

    for f in [main_f, flips_f, values_f, eval_f]:
        f.close()

    # ── Meta JSON ───────────────────────────────────────────────────────────────
    meta = {
        "density":         args.density,
        "seed":            args.seed,
        "episodes":        args.episodes,
        "lr":              algo.lr,
        "tau":             algo.tau,
        "batch_size":      algo.batch_size,
        "gamma":           algo.gamma,
        "memory_capacity": algo.memory_capacity,
        "hidden_dim":      algo.hidden_dim,
        "epsilon": {
            "type":       "linear_then_constant",
            "start":      1.0,
            "final":      args.epsi_final,
            "anneal_from_ep": 0,
            "anneal_to_ep":   epsi_anneal,
            "formula":    "eps(t) = 1.0 - t*(1-epsi_final)/(epsi_anneal-1)  if t<epsi_anneal  else epsi_final",
        },
        "initial_greedy_actions": initial_greedy,
        "first_flip_episode": first_flip_ep,
        "Q_answer_a_sync_update": (
            "NOT fully synchronous. All 16 agents update sequentially within "
            "each episode: for i in 0..15: optimize_model(i); soft_update_target_net(i). "
            "Agent i's update sees a slightly different target net than agent i+1's, "
            "because soft_update_target_net(i) runs before optimize_model(i+1)."
        ),
        "Q_answer_b_warmup": (
            f"optimize_model() is a no-op until buffer has >= batch_size={algo.batch_size} "
            f"transitions. So first {algo.batch_size} episodes produce no weight update. "
            f"First observed flip: ep={first_flip_ep}."
        ),
        "Q_answer_c_eval": (
            "Evaluation: set eps=0 for all agents, call select_action once per agent "
            "(argmax Q, deterministic). Single trial. "
            "loads column = load from this single greedy trial. "
            "eval_idql.csv stores raw joint actions for last 10 checkpoints."
        ),
    }
    with open(os.path.join(out_dir, "meta_idql.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ── Verification & statistics ───────────────────────────────────────────────
    CONVERGENCE_EP = 12400
    post_flips = [c for te, c in enumerate(all_flip_counts) if te >= CONVERGENCE_EP]

    n_steps = len(all_flip_counts)
    p0 = sum(1 for c in all_flip_counts if c == 0) / n_steps
    p1 = sum(1 for c in all_flip_counts if c == 1) / n_steps
    p2 = sum(1 for c in all_flip_counts if c == 2) / n_steps
    p3 = sum(1 for c in all_flip_counts if c >= 3) / n_steps
    max_c_post = max(post_flips) if post_flips else 0

    print(f"\n=== Verification & Statistics ===")
    print(f"  Greedy-load vs eval-load mismatches (should be 0): "
          f"{len(verify_mismatches)}")
    for m in verify_mismatches[:5]:
        print(f"    te={m[0]}: greedy_load={m[1]}  eval_load={m[2]}")

    print(f"  Flip-count distribution (over {n_steps} episodes):")
    print(f"    P(c=0) = {p0:.4f}   P(c=1) = {p1:.4f}   "
          f"P(c=2) = {p2:.4f}   P(c≥3) = {p3:.4f}")
    print(f"  Max c after ep≥{CONVERGENCE_EP}: {max_c_post}")
    print(f"  First flip episode: {first_flip_ep}")

    # ── Load-2 simultaneous departure detection ─────────────────────────────────
    flips_path = os.path.join(out_dir, "flips_idql.csv")
    flip_by_step = {}
    with open(flips_path) as f:
        for row in csv.DictReader(f):
            step = int(row["update_step"])
            if step not in flip_by_step:
                flip_by_step[step] = []
            flip_by_step[step].append(
                (int(row["agent_id"]), int(row["old_action"]), int(row["new_action"]))
            )

    curr_recon = initial_greedy[:]
    load2_events = []

    for step in range(args.episodes):
        flips_here = flip_by_step.get(step, [])
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
                        "episode":  step,
                        "sc":       sc,
                        "agents":   agents,
                        "pre_load": pre_load.tolist(),
                    })
            for (aid, old_a, new_a) in flips_here:
                curr_recon[aid] = new_a

    print(f"\n  Load-2 simultaneous departure events: {len(load2_events)}")
    for ev in load2_events:
        print(f"    ep={ev['episode']}  sc={ev['sc']}  "
              f"agents={ev['agents']}  pre_load={ev['pre_load']}")
    if not load2_events:
        print("    None found — ratchet invariance holds empirically.")

    # ── Flips-reconstruction vs main.csv cross-check ────────────────────────────
    main_csv_rows = {}
    with open(os.path.join(out_dir, "main.csv")) as f:
        for row in csv.DictReader(f):
            main_csv_rows[int(row["episode"])] = row["loads"]

    curr_recon = initial_greedy[:]
    recon_mismatches = []
    for step in range(args.episodes):
        for (aid, old_a, new_a) in flip_by_step.get(step, []):
            curr_recon[aid] = new_a
        if step in main_csv_rows:
            recon_load = _greedy_load(curr_recon).tolist()
            if str(recon_load) != main_csv_rows[step]:
                recon_mismatches.append((step, recon_load, main_csv_rows[step]))

    print(f"\n  Flips-reconstruction vs main.csv loads mismatches "
          f"(should be 0): {len(recon_mismatches)}")
    for m in recon_mismatches[:5]:
        print(f"    ep={m[0]}: recon={m[1]}  csv={m[2]}")

    # ── Final ───────────────────────────────────────────────────────────────────
    final_r, _, final_loads = _evaluate(S, I, agent_list)
    pct_opt = final_r / opt_R * 100 if opt_R > 0 else float("nan")
    print(f"\n=== Final greedy ===  reward={final_r:.4f}  "
          f"({pct_opt:.1f}% OPT)  loads={final_loads.tolist()}")
    print(f"Saved → {out_dir}/")


if __name__ == "__main__":
    main()
