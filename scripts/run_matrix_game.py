"""
run_matrix_game.py — IDQL on 6 homogeneous C-V2X matrix games (no dataset).

Game: N=16 agents, M=4 subchannels, action ∈ {0,1,2,3=SC, 4=NT}.
Per-agent reward: log2(1 + S/(σ² + (n_m−1)·I))  [individual, selfish].
σ² = −114 dBm (fixed), no V2I. S and I derived from ETSI diamond operating
points (ratio-of-means from etsi_real_topo_overlay.py).

Theory bounds (OPT, worst/best NE, PoA) printed before training and compared
at test intervals so we can see whether IDQL finds a good NE.

Usage:
  python run_matrix_game.py --density 333 --seed 0 --episodes 10000
  python run_matrix_game.py --density 333 --seed 0 --episodes 50000
"""

import argparse
import csv
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Configuration.idql_params import IDQLparameters
from Networks.Agents.idql_agent import DQNAgent
from analysis.paper_utils import SIGMA2_MW, classify_4, find_opt_and_ne, n_plus_exact, R_ch

# ── Operating points (β_diamond, γ_diamond) from etsi_real_topo_stats.csv ─────
# Computed as 10·log10(mean_Sbar / mean_Ibar) and 10·log10(mean_Sbar / σ²).
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
NT_ACT    = N_SC      # action index 4 = No Transmission
STATE_DIM = 1         # stateless game — constant input [1.0]

# Canonical Regime-III PNE load vectors (sorted desc), N=16 M=4
CANON = {
    (1, 1, 1, 1):  'v0',
    (13, 1, 1, 1): 'v1',
    (7, 7, 1, 1):  'v2',
    (5, 5, 5, 1):  'v3',
    (4, 4, 4, 4):  'v4',
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_SI(beta_db: float, gamma_db: float):
    """Back-calculate (S, I) [mW] from operating-point (β, γ) [dB]."""
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    return S, I


def _constant_state() -> np.ndarray:
    return np.array([[1.0]], dtype=np.float32)   # shape (1, STATE_DIM)


def _play_episode(S: float, I: float, agent_list: list, epsi: float):
    """
    One training episode (single-step game).
    Sets ε for all agents, collects actions, computes individual rewards,
    stores transitions in each agent's replay buffer.
    Returns total (global) reward and per-SC load vector.
    """
    state = _constant_state()

    for ag in agent_list:
        ag.eps_threshold = epsi

    actions = [ag.select_action(state, None).item() for ag in agent_list]

    loads = np.zeros(N_SC, dtype=int)
    for a in actions:
        if a < N_SC:
            loads[a] += 1

    # Global reward = total SE across all active agents (shared by all, same as existing trainer)
    total = 0.0
    for a in actions:
        if a < N_SC:
            n_m = loads[a]
            total += math.log2(1.0 + S / (SIGMA2_MW + (n_m - 1) * I))

    global_r = np.array([[total]], dtype=np.float32)
    for ag_idx, ag in enumerate(agent_list):
        ag.store_transition(state, np.array([[actions[ag_idx]]]), state, True, global_r)

    return total, loads


def _evaluate(S: float, I: float, agent_list: list):
    """
    Greedy (ε=0) evaluation.
    Returns total reward, per-agent action list, per-SC load vector.
    """
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


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="IDQL on homogeneous C-V2X matrix game (no dataset)."
    )
    parser.add_argument("--density",  type=int, default=None,
                        choices=[35, 62, 123, 167, 333, 500],
                        help="ETSI vehicle density [veh/km] (or use --op_beta/--op_gamma)")
    parser.add_argument("--op_beta",  type=float, default=None,
                        help="Operating-point beta [dB] (full precision); overrides --density")
    parser.add_argument("--op_gamma", type=float, default=None,
                        help="Operating-point gamma [dB] (full precision); overrides --density")
    parser.add_argument("--seed",     type=int, default=0,
                        help="Random seed")
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes (default: idql_params.training_episodes)")
    parser.add_argument("--epsi_final", type=float, default=None,
                        help="Final epsilon after annealing (default: idql_params.epsi_final)")
    args = parser.parse_args()

    if args.op_beta is not None and args.op_gamma is not None:
        beta_db, gamma_db = args.op_beta, args.op_gamma
    elif args.density is not None:
        beta_db, gamma_db = OPERATING_POINTS[args.density]
    else:
        parser.error("must give either --density or both --op_beta and --op_gamma")

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    S, I = _get_SI(beta_db, gamma_db)

    # ── Load training params (CLI overrides) ───────────────────────────────────
    algo = IDQLparameters()
    episodes    = args.episodes if args.episodes is not None else algo.training_episodes
    epsi_final  = args.epsi_final if args.epsi_final is not None else algo.epsi_final
    epsi_start  = algo.epsi_start
    epsi_anneal = int(algo.epsi_anneal_frac * episodes)
    test_interval = max(1, episodes // 100)

    # ── Theory bounds ──────────────────────────────────────────────────────────
    opt_dist, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
    worst_NE_R = ne_list[0][1]  if ne_list else float("nan")
    best_NE_R  = ne_list[-1][1] if ne_list else float("nan")
    worst_dist = ne_list[0][0]  if ne_list else None
    best_dist  = ne_list[-1][0] if ne_list else None
    poa        = opt_R / worst_NE_R if (ne_list and worst_NE_R > 0) else float("nan")
    regime     = classify_4(S, I, SIGMA2_MW)
    np_val     = n_plus_exact(S, I, SIGMA2_MW)

    print(f"=== Matrix game  λ={args.density} veh/km ===")
    print(f"  β={beta_db:.2f} dB   γ={gamma_db:.2f} dB   regime={regime}   n⁺={np_val}")
    print(f"  S={S:.3e} mW   I={I:.3e} mW   σ²={SIGMA2_MW:.3e} mW")
    print(f"  OPT    = {opt_R:.4f}  dist={opt_dist}")
    print(f"  best_NE= {best_NE_R:.4f}  dist={best_dist}")
    print(f"  worst_NE={worst_NE_R:.4f}  dist={worst_dist}")
    print(f"  PoA    = {poa:.4f}   |NE|={len(ne_list)}")
    print(f"  episodes={episodes}  epsi={epsi_start}->{epsi_final} (anneal@{epsi_anneal})  seed={args.seed}")
    print(f"  lr={algo.lr}  batch={algo.batch_size}  gamma={algo.gamma}  tau={algo.tau}  buf={algo.memory_capacity}  hidden={algo.hidden_dim}\n")

    # ── Agents ─────────────────────────────────────────────────────────────────
    agent_list = [
        DQNAgent(
            ag_idx=i, num_agents=N_AGENTS,
            state_dim=STATE_DIM, action_dim=N_SC + 1,
            is_hysteretic_q=False,
            memory_capacity=algo.memory_capacity,
            batch_size=algo.batch_size,
            gamma=algo.gamma, tau=algo.tau, lr=algo.lr,
            hidden_dim=algo.hidden_dim,
            hysteretic_high_lr=algo.hysteretic_high_lr,
            hysteretic_low_lr=algo.hysteretic_low_lr,
            force_nt_when_empty=False,
        )
        for i in range(N_AGENTS)
    ]

    # ── Output CSV ─────────────────────────────────────────────────────────────
    out_dir = os.path.join("Results", "IDQL_MatrixGame")
    os.makedirs(out_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"lam{args.density}" if args.density is not None else f"b{beta_db:.4f}_g{gamma_db:.4f}"
    csv_name = f"IDQL_MG_{tag}_ep{episodes}_ef{epsi_final}_seed{args.seed}_{ts}.csv"
    csv_path = os.path.join(out_dir, csv_name)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "test_reward", "pct_opt", "pct_best_ne", "loads"])
        f.flush()

        # ── Training loop ──────────────────────────────────────────────────────
        for te in range(episodes):
            if te < epsi_anneal:
                epsi = epsi_start - te * (epsi_start - epsi_final) / max(epsi_anneal - 1, 1)
            else:
                epsi = epsi_final

            _play_episode(S, I, agent_list, epsi)

            for ag in agent_list:
                ag.optimize_model()
                ag.soft_update_target_net()

            if te % test_interval == 0:
                test_r, _, loads = _evaluate(S, I, agent_list)
                pct_opt     = test_r / opt_R      * 100 if opt_R      > 0 else float("nan")
                pct_best_ne = test_r / best_NE_R  * 100 if best_NE_R  > 0 else float("nan")
                writer.writerow([te, f"{test_r:.4f}", f"{pct_opt:.1f}",
                                 f"{pct_best_ne:.1f}", loads.tolist()])
                f.flush()
                print(f"  ep {te:>6}/{episodes}: "
                      f"reward={test_r:.3f} ({pct_opt:.1f}% OPT, {pct_best_ne:.1f}% best_NE)  "
                      f"loads={loads.tolist()}")

    # ── Final greedy evaluation ────────────────────────────────────────────────
    final_r, actions, loads = _evaluate(S, I, agent_list)
    pct_opt     = final_r / opt_R     * 100 if opt_R     > 0 else float("nan")
    pct_best_ne = final_r / best_NE_R * 100 if best_NE_R > 0 else float("nan")

    # Deterministic readout load vector → PNE status + rho
    readout = tuple(sorted(loads.tolist(), reverse=True))
    pne_set = {d for d, _ in ne_list}
    is_pne  = readout in pne_set
    phi_r   = sum(R_ch(n, S, I, SIGMA2_MW) for n in readout)
    rho     = ((opt_R - phi_r) / (opt_R - worst_NE_R)
               if is_pne and abs(opt_R - worst_NE_R) > 1e-15 else float("nan"))
    label   = CANON.get(readout, "non-PNE")

    print(f"\n=== Final (greedy readout) ===")
    print(f"  Reward   = {final_r:.4f}   (recomputed Phi(n^L) = {phi_r:.4f})")
    print(f"  % OPT    = {pct_opt:.1f}%")
    print(f"  % best_NE= {pct_best_ne:.1f}%")
    print(f"  Loads    = {loads.tolist()}   sorted={readout}")
    print(f"  PNE?     = {is_pne}   label={label}   rho={rho:.4f}" if is_pne
          else f"  PNE?     = {is_pne}   (readout not in equilibrium set)")
    print(f"  OPT={opt_R:.4f}  best_NE={best_NE_R:.4f}  worst_NE={worst_NE_R:.4f}  PoA={poa:.4f}")
    print(f"Saved → {csv_path}")


if __name__ == "__main__":
    main()
