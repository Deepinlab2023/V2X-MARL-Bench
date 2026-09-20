"""
plot_idqn_convergence.py — Plot single IDQN run convergence with NE reference lines.

X-axis: environment steps (= episodes for NFIG single-step game)
Y-axis: test reward
5 horizontal lines: reward of each canonical PNE (v0–v4)

Usage:
  python3 plot_idqn_convergence.py Results/IDQL_MatrixGame/IDQN\ low/IDQL_MG_b13.2932_g74.5837_ep30000_ef0.01_seed9_20260828_020147.csv
  python3 plot_idqn_convergence.py --csv <path> [--save <out.png>]
"""
import argparse
import csv
import math
import os
import re
import sys

import matplotlib.pyplot as plt
import numpy as np

from analysis.paper_utils import R_ch, find_opt_and_ne, SIGMA2_MW

N_AGENTS = 16
N_SC = 4

CANON = {
    (1, 1, 1, 1):   "v0",
    (13, 1, 1, 1):  "v1",
    (7, 7, 1, 1):   "v2",
    (5, 5, 5, 1):   "v3",
    (4, 4, 4, 4):   "v4",
}

_RE_FILENAME = re.compile(r"b([\d.]+)_g([\d.]+)")


def parse_beta_gamma_from_filename(filepath):
    m = _RE_FILENAME.search(os.path.basename(filepath))
    if not m:
        return None, None
    return float(m.group(1)), float(m.group(2))


def compute_ne_rewards(beta_db, gamma_db):
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    _, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
    ne_rewards = {}
    for dist, reward in ne_list:
        label = CANON.get(dist, str(dist))
        ne_rewards[label] = reward
    return ne_rewards, opt_R, S, I


def read_csv(filepath):
    episodes = []
    rewards = []
    loads_list = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["test_reward"]))
            loads_list.append(row.get("loads", ""))
    return episodes, rewards, loads_list


def plot(filepath, save_path=None):
    beta, gamma = parse_beta_gamma_from_filename(filepath)
    if beta is None:
        print("ERROR: cannot parse beta/gamma from filename")
        sys.exit(1)

    ne_rewards, opt_R, S, I = compute_ne_rewards(beta, gamma)
    episodes, rewards, _ = read_csv(filepath)

    steps = np.array(episodes)
    rewards_arr = np.array(rewards)

    ne_order = ["v0", "v1", "v2", "v3", "v4"]
    ne_colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728", "#9467bd"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, label in enumerate(ne_order):
        if label in ne_rewards:
            r = ne_rewards[label]
            ax.axhline(y=r, color=ne_colors[i], linestyle="--", linewidth=1.2,
                       alpha=0.7, label=f"{label} = {label.split('v')[0]}")
            ax.text(steps[-1] * 1.01, r, f" {label} ({r:.1f})",
                    va="center", fontsize=8, color=ne_colors[i])

    ax.plot(steps, rewards_arr, color="#333333", linewidth=1.5, alpha=0.85,
            label="test reward")

    ax.set_xlabel("Environment Steps", fontsize=12)
    ax.set_ylabel("Reward (sum spectral efficiency)", fontsize=12)
    ax.set_title(f"IDQN Convergence — beta={beta:.2f} gamma={gamma:.2f} "
                 f"(PoA={opt_R/ne_rewards['v4']:.2f})", fontsize=13)

    y_min = min(ne_rewards.get("v4", 30), rewards_arr.min()) - 5
    y_max = max(ne_rewards.get("v0", 110), rewards_arr.max()) + 5
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(0, steps[-1] * 1.08)

    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Plot IDQN convergence with NE reference lines")
    ap.add_argument("--csv", type=str, required=True,
                    help="Path to IDQN matrix game CSV")
    ap.add_argument("--save", type=str, default=None,
                    help="Save to file instead of showing")
    args = ap.parse_args()
    plot(args.csv, args.save)


if __name__ == "__main__":
    main()
