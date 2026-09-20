"""
summarize_all_results.py — Consolidate IDQN + IPPO results into one CSV.

Reads directly from the CSV files (not logs):
  IDQN: Results/IDQL_MatrixGame/IDQN {point}/*.csv
  IPPO: Results/IPPO_MatrixGame/{point}/*.csv

Output: Results/VE_batch/summary_all_runs.csv

Usage:
  python3 summarize_all_results.py
"""
import csv
import glob
import math
import os
import re
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.paper_utils import R_ch, find_opt_and_ne, SIGMA2_MW

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(REPO, "Results", "VE_batch", "summary_all_runs.csv")

N_AGENTS = 16
N_SC = 4

CANON = {
    (1, 1, 1, 1):   "v0",
    (13, 1, 1, 1):  "v1",
    (7, 7, 1, 1):   "v2",
    (5, 5, 5, 1):   "v3",
    (4, 4, 4, 4):   "v4",
}

POINTS = {
    "low":  (13.293186199612448, 74.58369474788802),
    "mid":  (11.976708878135344, 79.92478010397355),
    "high": (10.141001516147227, 83.40988888212391),
}

ALGO_DIRS = {
    "idqn": "Results/IDQL_MatrixGame/IDQN {point}",
    "ippo": "Results/IPPO_MatrixGame/{point}",
}

_RE_SEED = re.compile(r"seed(\d+)")


def compute_ne_set(beta_db, gamma_db):
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    _, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
    ne_set = {dist for dist, _ in ne_list}
    ne_rewards = {CANON.get(d, str(d)): r for d, r in ne_list}
    return ne_set, ne_rewards, opt_R


def l1_dist(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def find_nearest_pne(readout, ne_set, ne_rewards_map):
    if not readout:
        return "?", -1, 0.0
    best_d, best_dist, best_r = 999, None, 0.0
    for dist_tuple in ne_set:
        d = l1_dist(readout, dist_tuple)
        r = ne_rewards_map.get(dist_tuple, 0.0)
        if d < best_d:
            best_d = d
            best_dist = dist_tuple
            best_r = r
    return CANON.get(best_dist, str(best_dist)), best_d, best_r


def parse_csv_final(filepath):
    """Read last row of CSV: returns (seed, final_reward, final_loads_tuple)."""
    seed = None
    m = _RE_SEED.search(os.path.basename(filepath))
    if m:
        seed = int(m.group(1))

    last_row = None
    with open(filepath) as f:
        for row in csv.DictReader(f):
            last_row = row

    if not last_row:
        return seed, float("nan"), None

    reward = float(last_row["test_reward"])
    loads_str = last_row.get("loads", "")
    loads_str = loads_str.strip().strip("[]")
    loads = tuple(sorted([int(x) for x in loads_str.split(",")], reverse=True))

    return seed, reward, loads


def main():
    S_global = None
    I_global = None

    rows = []
    for algo in ["idqn", "ippo"]:
        for point in ["low", "mid", "high"]:
            beta, gamma = POINTS[point]
            S = SIGMA2_MW * 10.0 ** (gamma / 10.0)
            I = SIGMA2_MW * 10.0 ** ((gamma - beta) / 10.0)
            ne_set, ne_rewards, opt_R = compute_ne_set(beta, gamma)
            ne_rewards_map = {d: r for d, r in find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)[2]}
            worst_ne = ne_rewards.get("v4", 0)
            poa = opt_R / worst_ne if worst_ne > 0 else float("nan")

            dir_path = os.path.join(REPO, ALGO_DIRS[algo].format(point=point))
            csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))

            for fpath in csv_files:
                seed, reward, readout = parse_csv_final(fpath)
                if readout is None:
                    continue

                is_pne = readout in ne_set
                label = CANON.get(readout, "non-PNE")

                if is_pne:
                    nearest_label = label
                    dist_to_nearest = 0
                    nearest_reward = reward
                else:
                    nearest_label, dist_to_nearest, nearest_reward = \
                        find_nearest_pne(readout, ne_set, ne_rewards_map)

                pct_opt = reward / opt_R * 100 if opt_R > 0 else float("nan")
                pct_nearest = reward / nearest_reward * 100 if nearest_reward > 0 else float("nan")
                emp_poa = opt_R / reward if reward > 0 else float("nan")

                if is_pne:
                    conv = f"exact {label}"
                elif dist_to_nearest <= 2:
                    conv = f"near {nearest_label} (off by {dist_to_nearest})"
                elif dist_to_nearest <= 5:
                    conv = f"approx {nearest_label} (off by {dist_to_nearest})"
                else:
                    conv = f"far (off by {dist_to_nearest})"

                rows.append({
                    "point": point,
                    "algo": algo,
                    "seed": seed,
                    "PoA": f"{poa:.4f}",
                    "OPT": f"{opt_R:.4f}",
                    "worst_NE": f"{worst_ne:.4f}",
                    "Phi_L": f"{reward:.4f}",
                    "pct_OPT": f"{pct_opt:.1f}",
                    "load_vector": str(readout),
                    "is_pne": is_pne,
                    "label": label,
                    "nearest_pne": nearest_label,
                    "dist_to_nearest": dist_to_nearest,
                    "pct_of_nearest": f"{pct_nearest:.1f}",
                    "empirical_PoA": f"{emp_poa:.4f}",
                    "convergence_status": conv,
                })

    cols = [
        "point", "algo", "seed",
        "PoA", "OPT", "worst_NE",
        "Phi_L", "pct_OPT",
        "load_vector", "is_pne", "label",
        "nearest_pne", "dist_to_nearest", "pct_of_nearest",
        "empirical_PoA",
        "convergence_status",
    ]

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: (x["point"], x["algo"], x["seed"])):
            w.writerow(r)

    print(f"{len(rows)} rows -> {OUT_CSV}\n")

    # Print summary table
    print(f"{'point':>5} {'algo':>5} {'seed':>4} {'PoA':>6} {'Phi^L':>8} {'%OPT':>6} "
          f"{'load_vector':>16} {'PNE?':>4} {'label':>7} {'nearest':>8} {'dist':>4} "
          f"{'emp_PoA':>8}  {'status'}")
    print("-" * 110)
    for r in sorted(rows, key=lambda x: (x["point"], x["algo"], x["seed"])):
        pne = "Y" if r["is_pne"] else "N"
        print(f"{r['point']:>5} {r['algo']:>5} {r['seed']:>4} {r['PoA']:>6} "
              f"{r['Phi_L']:>8} {r['pct_OPT']:>5}% {r['load_vector']:>16} "
              f"{pne:>4} {r['label']:>7} {r['nearest_pne']:>8} "
              f"{r['dist_to_nearest']:>4} {r['empirical_PoA']:>8}  "
              f"{r['convergence_status']}")


if __name__ == "__main__":
    main()
