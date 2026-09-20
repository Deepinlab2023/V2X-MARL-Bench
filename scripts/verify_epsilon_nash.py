"""
verify_epsilon_nash.py — Exhaustively verify that IDQN readouts are ε-Nash.

For each of the 15 IDQN runs, takes the final greedy readout load vector,
enumerates all 16×5=80 single-agent deviations, computes the exact reward
for each, and finds ε_max = max_i max_{a_i} [R(a_i, a_{-i}) - R(a)].

Compares ε_max against the analytical formula Δ(n₁+1) and reports which
deviation type achieves the maximum.

Output: Results/VE_batch/epsilon_nash_verification.csv

Usage:
  python3 verify_epsilon_nash.py
"""
import csv
import glob
import math
import os
import re
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.paper_utils import R_ch, Delta_ch, find_opt_and_ne, SIGMA2_MW

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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


def _compute_reward_from_actions(actions, S, I):
    """actions: list of 16 ints (0-3=SC, 4=NT). Returns total reward."""
    loads = np.zeros(N_SC, dtype=int)
    for a in actions:
        if a < N_SC:
            loads[a] += 1
    total = 0.0
    for a in actions:
        if a < N_SC:
            n_m = loads[a]
            total += math.log2(1.0 + S / (SIGMA2_MW + (n_m - 1) * I))
    return total


def _readout_to_actions(readout):
    """Convert sorted-desc load vector to a concrete action list.

    readout = (n0, n1, n2, n3), sum <= 16.
    Agents 0..n0-1  -> SC 0
    Agents n0..n0+n1-1 -> SC 1
    ...
    Remaining -> NT
    """
    actions = []
    for sc in range(N_SC):
        for _ in range(readout[sc]):
            actions.append(sc)
    n_silent = N_AGENTS - sum(readout)
    for _ in range(n_silent):
        actions.append(NT_ACT)
    assert len(actions) == N_AGENTS
    return actions


def exhaustive_epsilon(actions, S, I):
    """
    Enumerate all 16×5=80 single-agent deviations.
    Returns (epsilon_max, best_agent, best_action, best_dev_type, all_devs).
    """
    base_reward = _compute_reward_from_actions(actions, S, I)
    epsilon_max = -1e9
    best_agent = -1
    best_action = -1
    best_dev_type = ""
    all_devs = []

    for i in range(N_AGENTS):
        orig_action = actions[i]
        for a_new in range(N_SC + 1):
            if a_new == orig_action:
                continue
            dev_actions = list(actions)
            dev_actions[i] = a_new
            dev_reward = _compute_reward_from_actions(dev_actions, S, I)
            improvement = dev_reward - base_reward
            all_devs.append((i, orig_action, a_new, improvement))

            if improvement > epsilon_max:
                epsilon_max = improvement
                best_agent = i
                best_action = a_new

    # Classify the best deviation
    orig = actions[best_agent]
    if orig == NT_ACT and best_action < N_SC:
        # Silent agent joining a channel
        dev_actions = list(actions)
        dev_actions[best_agent] = best_action
        loads_after = np.zeros(N_SC, dtype=int)
        for a in dev_actions:
            if a < N_SC:
                loads_after[a] += 1
        best_dev_type = f"silent->ch{best_action} (load becomes {loads_after[best_action]})"
    elif orig < N_SC and best_action == NT_ACT:
        best_dev_type = f"ch{orig}->silent"
    elif orig < N_SC and best_action < N_SC:
        best_dev_type = f"ch{orig}->ch{best_action}"
    else:
        best_dev_type = f"action {orig}->{best_action}"

    return epsilon_max, best_agent, best_action, best_dev_type, all_devs


def parse_csv_final(filepath):
    """Read last row: returns (seed, final_reward, readout_tuple)."""
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
    loads_str = last_row.get("loads", "").strip().strip("[]")
    loads = tuple(sorted([int(x) for x in loads_str.split(",")], reverse=True))
    return seed, reward, loads


def main():
    out_dir = os.path.join(REPO, "Results", "VE_batch")
    out_csv = os.path.join(out_dir, "epsilon_nash_verification.csv")

    rows = []

    for point in ["low", "mid", "high"]:
        beta, gamma = POINTS[point]
        S, I = _get_SI(beta, gamma)
        _, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
        pne_set = {d for d, _ in ne_list}

        dir_path = os.path.join(REPO, f"Results/IDQL_MatrixGame/IDQN {point}")
        csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))

        print(f"\n=== {point} (beta={beta:.2f}, gamma={gamma:.2f}) ===")
        print(f"  OPT={opt_R:.4f}")

        for fpath in csv_files:
            seed, csv_reward, readout = parse_csv_final(fpath)
            if readout is None:
                continue

            is_pne = readout in pne_set
            label = CANON.get(readout, "non-PNE")

            # Convert readout to concrete actions
            actions = _readout_to_actions(readout)

            # Exhaustive epsilon
            eps_max, best_agent, best_action, best_dev_type, all_devs = \
                exhaustive_epsilon(actions, S, I)

            # Analytical formula: Delta(n1+1) where n1 = readout[0]
            n1 = readout[0]
            delta_analytical = Delta_ch(n1 + 1, S, I, SIGMA2_MW)

            # Also compute Phi(readout) from R_ch for cross-check
            phi_readout = sum(R_ch(n, S, I, SIGMA2_MW) for n in readout)

            pct_opt = eps_max / opt_R * 100 if opt_R > 0 else float("nan")
            match = abs(eps_max - delta_analytical) < 1e-9

            print(f"  seed{seed}: readout={readout}  PNE={is_pne}  "
                  f"eps_max={eps_max:.6f}  Delta(n1+1)={delta_analytical:.6f}  "
                  f"match={match}  %Phi*={pct_opt:.3f}%  "
                  f"best_dev={best_dev_type}")

            rows.append({
                "point": point,
                "seed": seed,
                "readout": str(readout),
                "is_pne": is_pne,
                "label": label,
                "n1": n1,
                "eps_max": f"{eps_max:.6f}",
                "delta_analytical": f"{delta_analytical:.6f}",
                "match": match,
                "pct_of_opt": f"{pct_opt:.4f}",
                "best_deviation": best_dev_type,
                "phi_readout_Rch": f"{phi_readout:.6f}",
                "csv_reward": f"{csv_reward:.6f}",
            })

    # Write CSV
    cols = ["point", "seed", "readout", "is_pne", "label", "n1",
            "eps_max", "delta_analytical", "match", "pct_of_opt",
            "best_deviation", "phi_readout_Rch", "csv_reward"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nSaved -> {out_csv}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_match = sum(1 for r in rows if r["match"])
    n_total = len(rows)
    n_counter = sum(1 for r in rows if not r["match"])
    print(f"  Total runs: {n_total}")
    print(f"  eps_max == Delta(n1+1): {n_match}/{n_total}")
    print(f"  Counterexamples: {n_counter}")

    if n_counter > 0:
        print("\n  *** COUNTEREXAMPLES FOUND ***")
        for r in rows:
            if not r["match"]:
                print(f"    {r['point']} seed{r['seed']}: "
                      f"eps_max={r['eps_max']}  Delta={r['delta_analytical']}  "
                      f"best_dev={r['best_deviation']}")
        print("  *** STOP — this directly contradicts our statement ***")
    else:
        print("\n  All match. ε = Δ(n₁+1) verified for all 15 runs.")

    # Deviation type breakdown
    print("\n  Best deviation types:")
    dev_types = Counter(r["best_deviation"].split(" ")[0] for r in rows)
    for dt, cnt in dev_types.most_common():
        print(f"    {dt}: {cnt}")


if __name__ == "__main__":
    main()
