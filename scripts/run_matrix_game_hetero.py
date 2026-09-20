"""
run_matrix_game_hetero.py — IDQN on a FIXED heterogeneous C-V2X matrix game
(one SUMO channel realization).

Same network, loss, and ALL hyperparameters / training budget as
run_matrix_game.py (the homogeneous V-E study): idql_params defaults
(episodes=30000, epsi 1.0 -> 0.01 annealed over 80%, lr=1e-3, batch=64,
buffer=1000, hidden=128).  ONLY change: the reward is computed from a fixed
heterogeneous channel realization G loaded from a game-cache npz built by
analysis/hetero_game.py (per-link S_i, interferer I_ji from one SUMO
snapshot; sigma2 = thermal, no V2I).  The realization is held fixed
throughout training and evaluation.

Action space:
  default (single power): {0..3 = subchannel, 4 = NT} — identical to the
      homogeneous study (only channel heterogeneity changes);
  --three_power: {0..11 = sc*3 + power, 12 = NT} with powers [23, 10, 5] dBm
      (env_params.V2V_POWER_LEVELS_DBM, same encoding as the full benchmark).

The reference optimum for % curves and the final ratio is the SINGLE-POWER
exact optimum R*(G) (subset DP) stored in the cache.  For three-power runs
the ratio R_3P(a)/R*_1P is "reward relative to the single-power optimum"
(it can exceed 1) — NOT an efficiency w.r.t. the 3-power optimum, which is
not computed.

Usage:
  python3 run_matrix_game_hetero.py --game_cache Results/Hetero_batch/games/low_snap7092.npz --seed 9
  python3 run_matrix_game_hetero.py --game_cache ... --seed 9 --three_power
  python3 run_matrix_game_hetero.py --game_cache ... --seed 9 --episodes 3000   # pilot
"""

import argparse
import csv
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Configuration.idql_params import IDQLparameters
from Networks.Agents.idql_agent import DQNAgent
from analysis.hetero_game import load_game_cache

N_AGENTS = 16
STATE_DIM = 1          # stateless game — constant input [1.0]


def _constant_state() -> np.ndarray:
    return np.array([[1.0]], dtype=np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="IDQN on a fixed heterogeneous C-V2X matrix game (SUMO realization)."
    )
    parser.add_argument("--game_cache", type=str, required=True,
                        help="Path to the game-cache npz (analysis/hetero_game.py)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=None,
                        help="Training episodes (default: idql_params.training_episodes)")
    parser.add_argument("--epsi_final", type=float, default=None,
                        help="Final epsilon (default: idql_params.epsi_final)")
    parser.add_argument("--three_power", action="store_true",
                        help="3 power levels x 4 SC + NT = 13 actions "
                             "(default: single power, 5 actions)")
    parser.add_argument("--out_dir", type=str, default="Results/Hetero_batch/curves",
                        help="Directory for the training-curve CSV")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

    game, meta = load_game_cache(args.game_cache, three_power=args.three_power)
    density, snapshot_id = meta["density"], meta["snapshot_id"]
    R_star = meta["R_star"]                      # single-power exact optimum
    mode = "3P" if args.three_power else "1P"

    algo = IDQLparameters()
    episodes   = args.episodes   if args.episodes   is not None else algo.training_episodes
    epsi_final = args.epsi_final if args.epsi_final is not None else algo.epsi_final
    epsi_start = algo.epsi_start
    epsi_anneal = int(algo.epsi_anneal_frac * episodes)
    test_interval = max(1, episodes // 100)

    print(f"=== Hetero matrix game (IDQN, {mode})  density={density} "
          f"snapshot={snapshot_id}  seed={args.seed} ===")
    print(f"  snapshot proxy: beta={meta['beta']:.3f} dB  gamma={meta['gamma']:.3f} dB  "
          f"(ensemble mean {meta['beta_mean']:.3f}, {meta['gamma_mean']:.3f})")
    print(f"  n_actions={game.n_actions}  powers={game.pw_dbm} dBm  "
          f"sigma2={game.sigma2:.3e} mW")
    print(f"  R*_1P = {R_star:.4f}  (subset DP, exact)")
    print(f"  episodes={episodes}  epsi={epsi_start}->{epsi_final} "
          f"(anneal@{epsi_anneal})")
    print(f"  lr={algo.lr}  batch={algo.batch_size}  gamma={algo.gamma}  "
          f"tau={algo.tau}  buf={algo.memory_capacity}  hidden={algo.hidden_dim}\n")

    agent_list = [
        DQNAgent(
            ag_idx=i, num_agents=N_AGENTS,
            state_dim=STATE_DIM, action_dim=game.n_actions,
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

    os.makedirs(args.out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_name = (f"IDQN_HET_{density}_snap{snapshot_id}_{mode}"
                f"_ep{episodes}_seed{args.seed}_{ts}.csv")
    csv_path = os.path.join(args.out_dir, csv_name)

    def _evaluate():
        state = _constant_state()
        for ag in agent_list:
            ag.eps_threshold = 0.0
        actions = [ag.select_action(state, None).item() for ag in agent_list]
        return actions, game.reward(actions), game.loads(actions)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "test_reward", "pct_Rstar_1P", "loads"])
        f.flush()

        for te in range(episodes):
            if te < epsi_anneal:
                epsi = epsi_start - te * (epsi_start - epsi_final) / max(epsi_anneal - 1, 1)
            else:
                epsi = epsi_final

            state = _constant_state()
            for ag in agent_list:
                ag.eps_threshold = epsi
            actions = [ag.select_action(state, None).item() for ag in agent_list]
            global_r = np.array([[game.reward(actions)]], dtype=np.float32)
            for ag_idx, ag in enumerate(agent_list):
                ag.store_transition(state, np.array([[actions[ag_idx]]]),
                                    state, True, global_r)

            for ag in agent_list:
                ag.optimize_model()
                ag.soft_update_target_net()

            if te % test_interval == 0 or te == episodes - 1:
                _, r, loads = _evaluate()
                pct = r / R_star * 100 if R_star > 0 else float("nan")
                writer.writerow([te, f"{r:.4f}", f"{pct:.1f}", loads.tolist()])
                f.flush()
                print(f"  ep {te:>6}/{episodes}: reward={r:.3f} "
                      f"({pct:.1f}% R*_1P)  loads={loads.tolist()}")

    # ── Final greedy readout + exact stability metrics ────────────────────────
    actions, phi, loads = _evaluate()
    eps, is_pne, _, best_dev = game.epsilon_pne(actions)
    ratio = phi / R_star if R_star > 0 else float("nan")
    eps_norm = eps / R_star if R_star > 0 else float("nan")

    print(f"\n=== Final (hetero IDQN, {mode}, density={density}, "
          f"snap={snapshot_id}, seed={args.seed}) ===")
    print(f"  READOUT_ACTIONS={actions}")
    print(f"  Phi={phi:.6f}  R_star_1P={R_star:.6f}  ratio={ratio:.6f}")
    print(f"  PNE?={is_pne}  eps={eps:.9f}  eps_norm={eps_norm:.9f}")
    if best_dev is not None:
        print(f"  best_dev=agent{best_dev[0]}:{best_dev[1]}->{best_dev[2]}")
    print(f"  loads={loads.tolist()}")
    print(f"Saved -> {csv_path}")


if __name__ == "__main__":
    main()
