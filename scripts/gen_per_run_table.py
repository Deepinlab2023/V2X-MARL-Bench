"""
gen_per_run_table.py — Generate enhanced per-run results table for 30 new runs.

Output: Results/VE_batch/per_run_enhanced.csv

Columns:
  point, algo, seed, PoA, OPT, worst_NE,
  Phi_L, pct_OPT,
  load_vector, is_pne, label,
  nearest_pne, dist_to_nearest, nearest_pne_reward, pct_nearest,
  empirical_PoA, gap_theo_emp

The "nearest_pne" columns let you see that IDQN's non-PNE readouts
(e.g. (10,1,1,1)) are close to v1=(13,1,1,1) but off by a few agents
— i.e. IDQN approximately converges to a NE but cannot hit it exactly
due to decentralised greedy argmax.

Usage:
  python3 gen_per_run_table.py
"""
import csv
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.paper_utils import R_ch, find_opt_and_ne, SIGMA2_MW

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(REPO, "Results", "VE_batch", "logs")
OUT_CSV = os.path.join(REPO, "Results", "VE_batch", "per_run_enhanced.csv")

POINTS = {
    "low":  (13.293186199612448, 74.58369474788802),
    "mid":  (11.976708878135344, 79.92478010397355),
    "high": (10.141001516147227, 83.40988888212391),
}

N_AGENTS = 16
N_SC = 4
SEEDS = [9, 10, 12, 87, 91]

CANON = {
    (1, 1, 1, 1):   "v0",
    (13, 1, 1, 1):  "v1",
    (7, 7, 1, 1):   "v2",
    (5, 5, 5, 1):   "v3",
    (4, 4, 4, 4):   "v4",
}

# ── log parsers ──────────────────────────────────────────────────────────────
_RE_HEADER = re.compile(
    r"OPT=([\d.]+)\s+best_NE=([\d.]+)\s+worst_NE=([\d.]+)\s+PoA=([\d.]+)")
_RE_IDQN_REWARD  = re.compile(r"Reward\s+=\s+([\d.]+)")
_RE_IDQN_READOUT = re.compile(r"sorted=\(([^)]+)\)")
_RE_PNE_TRUE  = re.compile(r"PNE\?\s*=\s*True\s+label=(\S+)\s+rho=([\d.]+)")
_RE_PNE_FALSE = re.compile(r"PNE\?\s*=\s*False")
_RE_IPPO_STCH  = re.compile(r"\[stochastic\]\s+Phi\^L\s+=\s+([\d.]+)")
_RE_IPPO_RDTV  = re.compile(
    r"\[readout\]\s+Phi\(n\^L\)\s+=\s+([\d.]+)\s+loads=\[[^\]]*\]\s+sorted=\(([^)]+)\)")
_RE_IPPO_PNE_T = re.compile(r"\[readout\]\s+PNE\?=True\s+label=(\S+)\s+rho=([\d.]+)")
_RE_IPPO_PNE_F = re.compile(r"\[readout\]\s+PNE\?=False")


def parse_log(text, algo):
    out = {}
    hdrs = _RE_HEADER.findall(text)
    if not hdrs:
        return None
    o, bn, wn, p = hdrs[-1]
    out["opt_R"] = float(o)
    out["best_ne"] = float(bn)
    out["worst_ne"] = float(wn)
    out["poa"] = float(p)

    if algo == "idqn":
        m = _RE_IDQN_REWARD.search(text)
        out["phi_L"] = float(m.group(1)) if m else float("nan")
        m = _RE_IDQN_READOUT.search(text)
        out["readout"] = tuple(int(x) for x in m.group(1).split(",")) if m else None
        m = _RE_PNE_TRUE.search(text)
        if m:
            out["is_pne"] = True
            out["label"] = m.group(1)
            out["rho"] = float(m.group(2))
        elif _RE_PNE_FALSE.search(text):
            out["is_pne"] = False
            out["label"] = "non-PNE"
            out["rho"] = float("nan")
        else:
            out["is_pne"] = None
            out["label"] = "?"
            out["rho"] = float("nan")
    else:
        m = _RE_IPPO_STCH.search(text)
        out["phi_L"] = float(m.group(1)) if m else float("nan")
        m = _RE_IPPO_RDTV.search(text)
        out["readout"] = tuple(int(x) for x in m.group(2).split(",")) if m else None
        m = _RE_IPPO_PNE_T.search(text)
        if m:
            out["is_pne"] = True
            out["label"] = m.group(1)
            out["rho"] = float(m.group(2))
        elif _RE_IPPO_PNE_F.search(text):
            out["is_pne"] = False
            out["label"] = "non-PNE"
            out["rho"] = float("nan")
        else:
            out["is_pne"] = None
            out["label"] = "?"
            out["rho"] = float("nan")
    return out


def l1_dist(a, b):
    """L1 distance between two sorted tuples."""
    return sum(abs(x - y) for x, y in zip(a, b))


def find_nearest_pne(readout, ne_list):
    """Find the nearest PNE to a non-PNE readout.
    Returns (label, dist, reward).
    """
    if readout is None:
        return "?", -1, float("nan")
    best_label, best_dist, best_reward = "?", 999, 0.0
    for dist_tuple, reward in ne_list:
        d = l1_dist(readout, dist_tuple)
        if d < best_dist:
            best_dist = d
            best_reward = reward
            best_label = CANON.get(dist_tuple, str(dist_tuple))
    return best_label, best_dist, best_reward


def compute_ne_set(beta_db, gamma_db):
    """Compute S, I and full NE set for an operating point."""
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    _, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
    return S, I, opt_R, ne_list


def convergence_status(is_pne, label, dist_to_nearest):
    """Human-readable convergence status."""
    if is_pne:
        return f"exact {label}"
    if dist_to_nearest <= 2:
        return f"near {label.split()[0]} (off by {dist_to_nearest})"
    if dist_to_nearest <= 5:
        return f"approx {label.split()[0]} (off by {dist_to_nearest})"
    return f"far from any PNE (off by {dist_to_nearest})"


def main():
    # Pre-compute NE sets for each point
    ne_sets = {}
    for pn, (beta, gamma) in POINTS.items():
        S, I, opt_R, ne_list = compute_ne_set(beta, gamma)
        ne_sets[pn] = ne_list
        print(f"  {pn}: beta={beta:.2f} gamma={gamma:.2f}  "
              f"OPT={opt_R:.4f}  |NE|={len(ne_list)}")

    rows = []
    for pn in ["low", "mid", "high"]:
        beta, gamma = POINTS[pn]
        ne_list = ne_sets[pn]
        for algo in ["idqn", "ippo"]:
            for seed in SEEDS:
                tag = f"{algo}_{pn}_seed{seed}"
                log_path = os.path.join(LOG_DIR, f"{tag}.log")
                if not os.path.exists(log_path):
                    print(f"  [WARN] {tag}: not found")
                    continue
                with open(log_path) as f:
                    text = f.read()
                parsed = parse_log(text, algo)
                if not parsed:
                    print(f"  [WARN] {tag}: parse failed")
                    continue

                readout = parsed["readout"]
                is_pne = parsed["is_pne"]

                if is_pne:
                    nearest_label = parsed["label"]
                    dist_to_nearest = 0
                    nearest_reward = parsed["phi_L"]
                else:
                    nearest_label, dist_to_nearest, nearest_reward = \
                        find_nearest_pne(readout, ne_list)

                phi_L = parsed["phi_L"]
                opt_R = parsed["opt_R"]
                worst_ne = parsed["worst_ne"]
                poa = parsed["poa"]

                pct_opt = phi_L / opt_R * 100 if opt_R > 0 else float("nan")
                pct_nearest = phi_L / nearest_reward * 100 if nearest_reward > 0 else float("nan")
                emp_poa = opt_R / phi_L if phi_L > 0 else float("nan")
                gap = poa - emp_poa
                conv = convergence_status(is_pne, parsed["label"], dist_to_nearest)

                row = {
                    "point": pn,
                    "algo": algo,
                    "seed": seed,
                    "PoA": f"{poa:.4f}",
                    "OPT": f"{opt_R:.4f}",
                    "worst_NE": f"{worst_ne:.4f}",
                    "Phi_L": f"{phi_L:.4f}",
                    "pct_OPT": f"{pct_opt:.1f}",
                    "load_vector": str(readout) if readout else "?",
                    "is_pne": is_pne,
                    "label": parsed["label"],
                    "nearest_pne": nearest_label,
                    "dist_to_nearest": dist_to_nearest,
                    "nearest_pne_reward": f"{nearest_reward:.4f}",
                    "pct_of_nearest": f"{pct_nearest:.1f}",
                    "empirical_PoA": f"{emp_poa:.4f}",
                    "gap_theo_emp": f"{gap:.4f}",
                    "convergence_status": conv,
                }
                rows.append(row)
                print(f"  [{tag}] load={readout}  PNE={is_pne}  "
                      f"nearest={nearest_label}(d={dist_to_nearest})  "
                      f"emp_PoA={emp_poa:.4f}  status={conv}")

    # Write CSV
    cols = [
        "point", "algo", "seed",
        "PoA", "OPT", "worst_NE",
        "Phi_L", "pct_OPT",
        "load_vector", "is_pne", "label",
        "nearest_pne", "dist_to_nearest",
        "nearest_pne_reward", "pct_of_nearest",
        "empirical_PoA", "gap_theo_emp",
        "convergence_status",
    ]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\n{len(rows)} rows written -> {OUT_CSV}")

    # Print a readable summary table
    print("\n" + "=" * 120)
    print(f"{'point':>5} {'algo':>5} {'seed':>4} {'PoA':>6} {'Phi^L':>8} {'%OPT':>6} "
          f"{'load_vector':>16} {'PNE?':>4} {'label':>7} {'nearest':>8} {'dist':>4} "
          f"{'emp_PoA':>8} {'status':>25}")
    print("-" * 120)
    for r in sorted(rows, key=lambda x: (x["point"], x["algo"], x["seed"])):
        pne = "Y" if r["is_pne"] else "N"
        print(f"{r['point']:>5} {r['algo']:>5} {r['seed']:>4} {r['PoA']:>6} "
              f"{r['Phi_L']:>8} {r['pct_OPT']:>5}% {r['load_vector']:>16} "
              f"{pne:>4} {r['label']:>7} {r['nearest_pne']:>8} "
              f"{r['dist_to_nearest']:>4} {r['empirical_PoA']:>8} "
              f"{r['convergence_status']:>25}")


if __name__ == "__main__":
    main()
