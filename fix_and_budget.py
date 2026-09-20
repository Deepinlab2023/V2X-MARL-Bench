"""
fix_and_budget.py — Fix Phi^L/readout inconsistency + compute env-interaction
and gradient-update budgets for IDQN vs IPPO.

Two tasks:
  (1) Check every row in summary_all_runs.csv: is Phi^L == Phi(readout)?
      Report any mismatches. Fix the known high/ippo/seed91 issue.
  (2) Compute total env interactions and gradient updates for both algorithms
      under their actual configs, to confirm the 3/15 vs 15/15 comparison
      uses comparable training budgets.

Output: Results/VE_batch/budget_and_consistency.csv

Usage:
  python3 fix_and_budget.py
"""
import csv
import glob
import math
import os
import re

import numpy as np

from analysis.paper_utils import R_ch, find_opt_and_ne, SIGMA2_MW

REPO = os.path.dirname(os.path.abspath(__file__))

N_AGENTS = 16
N_SC = 4
NT_ACT = N_SC

POINTS = {
    "low":  (13.293186199612448, 74.58369474788802),
    "mid":  (11.976708878135344, 79.92478010397355),
    "high": (10.141001516147227, 83.40988888212391),
}

CANON = {
    (1, 1, 1, 1):   "v0",
    (13, 1, 1, 1):  "v1",
    (7, 7, 1, 1):   "v2",
    (5, 5, 5, 1):   "v3",
    (4, 4, 4, 4):   "v4",
}

_RE_SEED = re.compile(r"seed(\d+)")


def _get_SI(beta_db, gamma_db):
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    return S, I


def _phi_from_readout(readout, S, I):
    """Compute Phi(load vector) = sum_m R_ch(n_m)."""
    return sum(R_ch(n, S, I, SIGMA2_MW) for n in readout)


def _parse_readout(loads_str):
    loads_str = loads_str.strip().strip("[]").strip("()").strip('"').strip("'")
    parts = [int(x.strip()) for x in loads_str.split(",")]
    return tuple(sorted(parts, reverse=True))


def check_consistency():
    """Task 1: Check Phi^L == Phi(readout) for all rows."""
    summary_csv = os.path.join(REPO, "Results", "VE_batch", "summary_all_runs.csv")
    if not os.path.exists(summary_csv):
        print(f"[ERROR] {summary_csv} not found")
        return []

    results = []
    with open(summary_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            point = row["point"]
            algo = row["algo"]
            seed = row["seed"]
            phi_L_reported = float(row["Phi_L"])
            readout_str = row["load_vector"]
            readout = _parse_readout(readout_str)

            beta, gamma = POINTS[point]
            S, I = _get_SI(beta, gamma)
            phi_computed = _phi_from_readout(readout, S, I)

            diff = abs(phi_L_reported - phi_computed)
            consistent = diff < 0.01  # tolerance for float rounding in CSV

            results.append({
                "point": point,
                "algo": algo,
                "seed": seed,
                "readout": str(readout),
                "phi_L_reported": f"{phi_L_reported:.4f}",
                "phi_L_computed": f"{phi_computed:.4f}",
                "diff": f"{diff:.4f}",
                "consistent": consistent,
            })

    return results


def compute_budget():
    """Task 2: Compute env-interaction and gradient-update budgets."""
    # IDQN config (from idql_params.py)
    idqn_episodes = 30000
    idqn_batch_size = 64
    idqn_memory_capacity = 1000
    idqn_n_agents = 16

    # IPPO config (from ppo_params.py + run_matrix_game_ippo.py)
    ippo_episodes = 50000
    ippo_batch_size = 256
    ippo_epochs = 10
    ippo_num_mini_batches = 4
    ippo_n_agents = 16

    # ── IDQN ─────────────────────────────────────────────────────────────
    # Each episode: 16 agents act simultaneously → 16 env interactions
    # Each episode: 16 agents each do 1 optimize_model() → 16 gradient updates
    #   (each operates on a batch of 64 from replay buffer)
    idqn_env_interactions = idqn_episodes * idqn_n_agents
    idqn_grad_updates = idqn_episodes * idqn_n_agents
    idqn_samples_per_update = idqn_batch_size
    idqn_total_samples_seen = idqn_grad_updates * idqn_samples_per_update

    # ── IPPO ─────────────────────────────────────────────────────────────
    # Each episode: 16 agents act simultaneously → 16 env interactions
    # Each update (batch_size episodes): 10 epochs × 4 mini-batches × 16 agents
    #   = 640 gradient updates per PPO update
    ippo_num_updates = ippo_episodes // ippo_batch_size
    ippo_env_interactions = ippo_episodes * ippo_n_agents
    ippo_grad_updates_per_update = ippo_epochs * ippo_num_mini_batches * ippo_n_agents
    ippo_grad_updates = ippo_num_updates * ippo_grad_updates_per_update
    ippo_mini_batch_size = ippo_batch_size // ippo_num_mini_batches
    ippo_total_samples_seen = ippo_grad_updates * ippo_mini_batch_size

    budget = {
        "IDQN": {
            "episodes": idqn_episodes,
            "n_agents": idqn_n_agents,
            "env_interactions": idqn_env_interactions,
            "grad_updates": idqn_grad_updates,
            "batch_size": idqn_batch_size,
            "total_samples_seen": idqn_total_samples_seen,
            "notes": f"{idqn_episodes} ep × {idqn_n_agents} agents = "
                     f"{idqn_env_interactions} env steps; "
                     f"1 optimize per agent per episode = "
                     f"{idqn_grad_updates} updates (each on {idqn_batch_size} from replay)",
        },
        "IPPO": {
            "episodes": ippo_episodes,
            "n_agents": ippo_n_agents,
            "env_interactions": ippo_env_interactions,
            "grad_updates": ippo_grad_updates,
            "batch_size": ippo_batch_size,
            "total_samples_seen": ippo_total_samples_seen,
            "notes": f"{ippo_episodes} ep × {ippo_n_agents} agents = "
                     f"{ippo_env_interactions} env steps; "
                     f"{ippo_num_updates} PPO updates × "
                     f"{ippo_epochs} epochs × {ippo_num_mini_batches} mini-batches × "
                     f"{ippo_n_agents} agents = {ippo_grad_updates} updates",
        },
    }
    return budget


def main():
    print("=" * 80)
    print("  TASK 1: Phi^L / readout consistency check")
    print("=" * 80)

    results = check_consistency()
    n_ok = sum(1 for r in results if r["consistent"])
    n_bad = sum(1 for r in results if not r["consistent"])

    print(f"\n  Total rows: {len(results)}")
    print(f"  Consistent: {n_ok}")
    print(f"  Inconsistent: {n_bad}")

    if n_bad > 0:
        print("\n  *** INCONSISTENT ROWS ***")
        print(f"  {'point':>6} {'algo':>5} {'seed':>4} {'readout':>16} "
              f"{'Phi^L(rpt)':>12} {'Phi(readout)':>14} {'diff':>8}")
        print("  " + "-" * 70)
        for r in results:
            if not r["consistent"]:
                print(f"  {r['point']:>6} {r['algo']:>5} {r['seed']:>4} "
                      f"{r['readout']:>16} {r['phi_L_reported']:>12} "
                      f"{r['phi_L_computed']:>14} {r['diff']:>8}")

    # Write consistency CSV
    out_dir = os.path.join(REPO, "Results", "VE_batch")
    cons_csv = os.path.join(out_dir, "phi_consistency_check.csv")
    cols = ["point", "algo", "seed", "readout",
            "phi_L_reported", "phi_L_computed", "diff", "consistent"]
    with open(cons_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results:
            w.writerow(r)
    print(f"\n  Consistency CSV -> {cons_csv}")

    # ── Task 2 ──────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  TASK 2: Training budget comparison (IDQN vs IPPO)")
    print("=" * 80)

    budget = compute_budget()

    print(f"\n  {'Metric':<30} {'IDQN':>15} {'IPPO':>15}")
    print("  " + "-" * 62)
    for key in ["episodes", "n_agents", "env_interactions",
                "grad_updates", "batch_size", "total_samples_seen"]:
        idqn_val = budget["IDQN"][key]
        ippo_val = budget["IPPO"][key]
        print(f"  {key:<30} {idqn_val:>15} {ippo_val:>15}")

    print(f"\n  IDQN notes: {budget['IDQN']['notes']}")
    print(f"  IPPO notes: {budget['IPPO']['notes']}")

    # Budget comparison verdict
    idqn_env = budget["IDQN"]["env_interactions"]
    ippo_env = budget["IPPO"]["env_interactions"]
    ratio = ippo_env / idqn_env if idqn_env > 0 else float("nan")
    print(f"\n  Env interaction ratio (IPPO/IDQN): {ratio:.2f}")
    if 0.5 < ratio < 2.0:
        print("  VERDICT: Budgets are comparable.")
    else:
        print(f"  VERDICT: Budgets differ by {ratio:.1f}x — may need adjustment.")

    # Write budget CSV
    budget_csv = os.path.join(out_dir, "budget_comparison.csv")
    with open(budget_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "IDQN", "IPPO"])
        for key in ["episodes", "n_agents", "env_interactions",
                    "grad_updates", "batch_size", "total_samples_seen"]:
            w.writerow([key, budget["IDQN"][key], budget["IPPO"][key]])
        w.writerow([])
        w.writerow(["IDQN notes", budget["IDQN"]["notes"], ""])
        w.writerow(["IPPO notes", budget["IPPO"]["notes"], ""])
    print(f"  Budget CSV -> {budget_csv}")


if __name__ == "__main__":
    main()
