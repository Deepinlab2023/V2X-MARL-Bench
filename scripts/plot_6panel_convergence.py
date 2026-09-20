#!/usr/bin/env python3
"""
plot_6panel_convergence.py — single 2x3 publication figure.

Rows    : algorithms (IDQN, IPPO)
Columns : operating points (low / mid / high PoA)
Panels  : per-seed test-reward curves, normalised by the optimum of that
          operating point, over the normalised pure-NE levels (v0..v4).
          No cross-seed averaging: the outcome is a discrete choice among
          equilibria, so a mean/median curve would sit between attractors and
          describe no run at all. The per-panel box reports how many seeds
          land on each equilibrium.

Reward normalisation
    Every curve and every level in a column is divided by that column's
    optimum, so the y axis reads R / R*. Since v0 = (1,1,1,1) is itself the
    global optimum, the v0 level sits at 1.0 and no separate "optimum" line is
    drawn.

Curve colour = EXACT final pure profile
    A run is attributed to an equilibrium only if its terminal joint action
    profile, read out as a sorted load vector, is exactly a pure NE of that
    operating point. Anything else is non-PNE and drawn grey. Reward-nearest
    attribution is deliberately NOT used: in Regime III the NE levels are close
    together, so a non-equilibrium readout can sit nearer an NE level than the
    equilibrium it is being compared against. The per-panel counts come from
    the same exact classification.

    The profile is looked up, in order, from:
      1. --profiles-csv  : a manifest (e.g. summary_all_runs.csv) with columns
                           for algo, point, seed/run and a load-vector column;
      2. a load-vector column inside the per-seed CSV itself (last valid row);
      3. a sidecar <stem>.json / <stem>_profile.json / final_profiles.json.
    If no profile is found the script fails loudly, unless
    --missing-profile grey is passed.

IPPO curves are left untouched (whatever the log records, i.e. the rollout mean
under the learned stochastic policy).

Layout is sized for an IEEE double-column figure (7.16 in wide).
Sharing: sharey='col' — the two algorithms in a column solve the same game, so
their reward axes must be identical for a fair visual comparison. sharex='row'
only, since the algorithms are trained over different horizons.

Output: <outdir>/convergence_6panel.{pdf,png}

Usage:
  python3 plot_6panel_convergence.py --profiles-csv Results/VE_batch/summary_all_runs.csv
  python3 plot_6panel_convergence.py --smooth 9 --formats pdf png
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.paper_utils import R_ch, find_opt_and_ne, SIGMA2_MW  # noqa: F401

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# Problem configuration
# ----------------------------------------------------------------------------
N_AGENTS = 16
N_SC = 4

CANON = {
    (1, 1, 1, 1):  "v0",
    (13, 1, 1, 1): "v1",
    (7, 7, 1, 1):  "v2",
    (5, 5, 5, 1):  "v3",
    (4, 4, 4, 4):  "v4",
}
CANON_ORDER = ["v0", "v1", "v2", "v3", "v4"]
# Rendered form of the labels above. The plain strings stay the dictionary
# keys; only the display text is typeset, so the load vectors read as the bold
# vectors the text defines them to be rather than as scalars.
CANON_TEX = {l: rf"$\mathbf{{v}}_{{{l[1:]}}}$" for l in CANON_ORDER}

POINTS = {
    "low":  (13.293186199612448, 74.58369474788802),
    "mid":  (11.976708878135344, 79.92478010397355),
    "high": (10.141001516147227, 83.40988888212391),
}
POINT_ORDER = ["low", "mid", "high"]
# Column headers. The (beta, gamma, PoA) numbers are given in the text, and
# repeating them above each column invites the reader to read the panels as a
# sweep over PoA; what distinguishes the columns operationally is the density
# of the operating point, so that is what the header states.
POINT_TITLES = {
    "low":  "Low density",
    "mid":  "Intermediate density",
    "high": "High density",
}

ALGO_DIRS = {
    "idqn": "Results/IDQL_MatrixGame/IDQN {point}",
    "ippo": "Results/IPPO_MatrixGame/{point}",
}
ALGO_TITLES = {"idqn": "IDQN", "ippo": "IPPO"}
ALGO_ORDER = ["idqn", "ippo"]

# ----------------------------------------------------------------------------
# Style — muted, print-safe, colour-blind friendly
# ----------------------------------------------------------------------------
NE_COLORS = ["#4C9A6E", "#5B8DB8", "#D4A55C", "#C67B7B", "#9B8BB8"]
NONPNE_COLOR = "#7A7A7A"      # runs whose final pure profile is not a PNE
SEED_COLORS = ["#3B6E8F", "#B5651D", "#4E8A63", "#A6435A", "#6F5A9B"]
LEVEL_GREY = "#BDBDBD"        # NE levels when hue encodes the seed
NONPNE_DASH = (0, (1.3, 1.3))  # dotted, but denser than ":" so it holds up at 600 dpi

RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.4,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def compute_ne_rewards(beta_db: float, gamma_db: float):
    """Return ({label: reward}, optimum_reward, PoA) for one operating point."""
    S = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    _, opt_R, ne_list = find_opt_and_ne(N_AGENTS, N_SC, S, I, SIGMA2_MW)
    ne_rewards = {CANON.get(dist, str(dist)): r for dist, r in ne_list}
    worst = min(ne_rewards.values()) if ne_rewards else float("nan")
    poa = opt_R / worst if worst else float("nan")
    return ne_rewards, opt_R, poa


_X_KEYS = ("env_step", "step", "steps", "episode")
_Y_KEYS = ("test_reward", "eval_reward", "reward")

# columns that may carry the terminal joint action / load vector
_PROFILE_KEYS = ("load_vector", "loadvector", "load", "loads", "counts",
                 "n_per_channel", "channel_counts", "profile", "final_profile",
                 "readout", "final_readout", "actions", "joint_action",
                 "final_actions")
# ... or four separate integer columns
_PROFILE_GROUPS = (("n0", "n1", "n2", "n3"),
                   ("c0", "c1", "c2", "c3"),
                   ("ch0", "ch1", "ch2", "ch3"),
                   ("sc0", "sc1", "sc2", "sc3"))


def parse_profile(raw) -> tuple | None:
    """Coerce a raw cell/field into a sorted load vector, or None.

    Accepts either
      * a load vector over the M subchannels, e.g. "[13, 1, 1, 1]" or
        "13 1 1 1". Its entries need not sum to N: agents may stay silent, and
        the optimum v0 = (1,1,1,1) has only M of the N agents transmitting.
      * a per-agent action list of length N, binned into subchannel counts.
        Silence must be encoded as -1 or as N_SC; channels are 0..N_SC-1. Any
        other encoding (e.g. 0 = silence with 1-indexed channels) is rejected
        rather than silently misread — pass a load vector instead.
    """
    if raw is None:
        return None
    if isinstance(raw, (list, tuple)):
        try:
            nums = [int(v) for v in raw]
        except (TypeError, ValueError):
            return None
    else:
        s = str(raw).strip()
        if not s or s.lower() in {"nan", "none", "na", "-"}:
            return None
        nums = [int(t) for t in re.findall(r"-?\d+", s)]
    if len(nums) == N_SC and min(nums) >= 0 and 0 < sum(nums) <= N_AGENTS:
        return tuple(sorted(nums, reverse=True))
    if len(nums) == N_AGENTS and all(-1 <= a <= N_SC for a in nums):
        active = np.asarray([a for a in nums if 0 <= a < N_SC])
        counts = np.bincount(active, minlength=N_SC) if active.size \
            else np.zeros(N_SC, dtype=int)
        return tuple(sorted(counts.tolist(), reverse=True))
    return None


def profile_from_row(row: dict, cols: dict) -> tuple | None:
    """Pull a load vector out of one CSV row, trying single- then multi-column."""
    for k in _PROFILE_KEYS:
        if k in cols:
            p = parse_profile(row.get(cols[k]))
            if p is not None:
                return p
    for group in _PROFILE_GROUPS:
        if all(g in cols for g in group):
            p = parse_profile([row.get(cols[g]) for g in group])
            if p is not None:
                return p
    return None


def read_run(filepath: str):
    """Read one seed log.

    Returns (x, y, profile) sorted by x; `profile` is the terminal load vector
    if the log itself carries one, else None.
    """
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return np.empty(0), np.empty(0), None
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        xk = next((cols[k] for k in _X_KEYS if k in cols), None)
        yk = next((cols[k] for k in _Y_KEYS if k in cols), None)
        if xk is None or yk is None:
            raise ValueError(f"{filepath}: no usable columns in {reader.fieldnames}")
        xs, ys, profile = [], [], None
        for row in reader:
            try:
                xs.append(float(row[xk]))
                ys.append(float(row[yk]))
            except (TypeError, ValueError):
                continue
            p = profile_from_row(row, cols)
            if p is not None:
                profile = p          # keep the last valid readout
    x, y = np.asarray(xs), np.asarray(ys)
    order = np.argsort(x)
    return x[order], y[order], profile


def sidecar_profile(filepath: str) -> tuple | None:
    """Look for a JSON readout next to the seed log."""
    stem, _ = os.path.splitext(filepath)
    base = os.path.basename(filepath)
    d = os.path.dirname(filepath)
    for cand in (stem + ".json", stem + "_profile.json",
                 os.path.join(d, "final_profiles.json"),
                 os.path.join(d, "final_profile.json")):
        if not os.path.exists(cand):
            continue
        try:
            with open(cand) as f:
                obj = json.load(f)
        except (OSError, ValueError):
            continue
        if isinstance(obj, dict):
            for key in (base, os.path.splitext(base)[0]):
                if key in obj:
                    p = parse_profile(obj[key])
                    if p is not None:
                        return p
            for k in _PROFILE_KEYS:
                if k in obj:
                    p = parse_profile(obj[k])
                    if p is not None:
                        return p
        else:
            p = parse_profile(obj)
            if p is not None:
                return p
    return None


_ALGO_KEYS = ("algo", "algorithm", "alg", "method")
_POINT_KEYS = ("point", "op", "operating_point", "op_point", "setting")
_SEED_KEYS = ("seed", "run", "run_id", "seed_id")


def seed_from_name(path: str) -> str | None:
    """Extract a seed id from a file name, e.g. '..._seed91.csv' -> '91'."""
    name = os.path.basename(path)
    m = re.search(r"seed[_\-]?(\d+)", name, flags=re.I)
    if m:
        return m.group(1)
    nums = re.findall(r"\d+", name)
    return nums[-1] if nums else None


def load_manifest(path: str) -> dict:
    """Index a summary CSV as {(algo, point): {"by_seed": ..., "ordered": ...}}."""
    out: dict = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: empty manifest")
        cols = {c.strip().lower(): c for c in reader.fieldnames}
        ak = next((cols[k] for k in _ALGO_KEYS if k in cols), None)
        pk = next((cols[k] for k in _POINT_KEYS if k in cols), None)
        sk = next((cols[k] for k in _SEED_KEYS if k in cols), None)
        if ak is None or pk is None:
            raise ValueError(f"{path}: manifest needs an algo and a point column; "
                             f"got {reader.fieldnames}")
        for row in reader:
            prof = profile_from_row(row, cols)
            if prof is None:
                continue
            key = (str(row[ak]).strip().lower(), str(row[pk]).strip().lower())
            slot = out.setdefault(key, {"by_seed": {}, "ordered": []})
            slot["ordered"].append(prof)
            if sk is not None:
                seed = re.sub(r"\D", "", str(row[sk]))
                if seed:
                    slot["by_seed"][seed] = prof
    return out


def resolve_profile(manifest, algo, point, filepath, index, csv_profile):
    """Pick the terminal pure profile for one run; manifest wins if present."""
    if manifest:
        slot = manifest.get((algo, point))
        if slot:
            seed = seed_from_name(filepath)
            if seed is not None and seed in slot["by_seed"]:
                return slot["by_seed"][seed]
            if index < len(slot["ordered"]):
                return slot["ordered"][index]
    if csv_profile is not None:
        return csv_profile
    return sidecar_profile(filepath)


def classify_exact(profile, ne_rewards):
    """Label a run by its exact terminal pure profile.

    Returns the canonical NE label iff the sorted load vector is exactly a pure
    NE of this operating point, else None (non-PNE). No reward-based fallback.
    """
    if profile is None:
        return None
    label = CANON.get(tuple(profile))
    return label if label in ne_rewards else None


def smooth(y: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average with correct edge normalisation."""
    if window <= 1 or y.size == 0:
        return y
    w = min(window, y.size)
    kernel = np.ones(w)
    return np.convolve(y, kernel, "same") / np.convolve(np.ones_like(y), kernel, "same")


def shade(color, f: float):
    """Lighten (f>0) or darken (f<0) a colour, keeping its hue."""
    rgb = mcolors.to_rgb(color)
    return tuple(c + (1.0 - c) * f if f >= 0 else c * (1.0 + f) for c in rgb)


def interleave(labels):
    """Round-robin draw order across basins, so overlapping runs blend evenly."""
    groups = {}
    for i, l in enumerate(labels):
        groups.setdefault(l, []).append(i)
    order, k = [], 0
    while len(order) < len(labels):
        for g in groups.values():
            if k < len(g):
                order.append(g[k])
        k += 1
    return order


def x_unit(xmax: float, exp: int | None = None):
    """Pick the decimal multiplier for an episode axis and its label fragment.

    The exponent follows the order of magnitude of the longest run on that
    axis, so a 10^5-episode schedule is labelled 10^5 rather than 100 x 10^3.
    Because the two algorithms are trained over different horizons and the
    figure only shares x within a row, this is resolved per row, not globally.
    """
    if exp is None:
        if not np.isfinite(xmax) or xmax <= 0:
            return 1.0, ""
        exp = int(np.floor(np.log10(xmax)))
    if exp <= 0:
        return 1.0, ""
    return 10.0 ** exp, rf"($\times 10^{{{exp}}}$)"


def stagger(values, min_gap: float):
    """Push overlapping label positions apart, preserving order."""
    vals = np.asarray(values, dtype=float)
    out = vals.copy()
    prev = -np.inf
    for i in np.argsort(vals):
        out[i] = max(vals[i], prev + min_gap)
        prev = out[i]
    return out


# ----------------------------------------------------------------------------
# Plotting
# ----------------------------------------------------------------------------
def draw_panel(ax, curves, seeds, labels, ne_rewards, opt_R, xscale, args,
               label_levels, mode, seed_colors=None):
    """Draw one panel. Rewards are normalised by `opt_R`.

    `labels[i]` is the exact-profile classification of run i (a canonical NE
    label, or None for non-PNE); it is computed upstream, never from reward.
    """
    by_seed = mode == "seed"

    ne_norm = {l: r / opt_R for l, r in ne_rewards.items()}
    reached = [l for l in labels if l is not None]
    counts = {l: reached.count(l) for l in CANON_ORDER if l in reached}

    # --- NE levels: pale gridline-like references, never over the curves ------
    # v0 is the global optimum and sits at 1.0, so no separate optimum line.
    for i, label in enumerate(CANON_ORDER):
        if label not in ne_norm:
            continue
        color = LEVEL_GREY if by_seed else shade(NE_COLORS[i], 0.40)
        ax.axhline(ne_norm[label], color=color, ls="--", lw=0.7,
                   alpha=0.75, zorder=0.5)

    # --- seed curves, coloured by their exact terminal pure profile -----------
    # Seeds are drawn round-robin across basins so that no single colour is
    # systematically painted on top of the others where curves coincide.
    rank, seen = {}, {}
    for i, l in enumerate(labels):
        seen[l] = seen.get(l, -1) + 1
        rank[i] = seen[l]
    sizes = {l: seen[l] + 1 for l in seen}

    for k, i in enumerate(interleave(list(labels))):
        x, y = curves[i]
        y = y / opt_R
        label = labels[i]
        tail = max(1, int(round(args.tail_frac * y.size)))
        v = float(np.median(y[-tail:]))
        m = sizes[label]
        t = (2.0 * rank[i] / (m - 1) - 1.0) if m > 1 else 0.0
        if by_seed:
            color = seed_colors[seeds[i]]
        elif label:
            # same hue per basin, graded lightness per seed, so overlapping runs
            # remain individually countable
            color = shade(NE_COLORS[CANON_ORDER.index(label)],
                          args.seed_shade * t)
        else:
            # grey group is spread towards dark only: a symmetric spread washes
            # the lightest seeds out to near-white when a whole panel is non-PNE
            color = shade(NONPNE_COLOR, 0.5 * args.seed_shade * (t - 1.0))
        # non-PNE runs carry the same weight as the others: they converge in
        # reward just as cleanly, and only their terminal profile differs. The
        # dash pattern, not a faded stroke, is what marks them.
        ax.plot(x / xscale, y, color=color,
                lw=0.6,
                alpha=args.alpha if label else args.alpha * 0.85,
                ls="-" if label else NONPNE_DASH,
                zorder=2 + 0.01 * k, solid_joinstyle="round")
        # filled marker = exact PNE, hollow = non-PNE (survives greyscale print)
        if label:
            ax.plot([x[-1] / xscale], [v], marker="o", ms=2.4, mew=0,
                    color=color, alpha=min(1.0, args.alpha + 0.25), zorder=4)
        else:
            ax.plot([x[-1] / xscale], [v], marker="o", ms=3.0, mfc="white",
                    mec=color, mew=0.8, zorder=4)

    # --- outcome distribution over equilibria --------------------------------
    n = len(curves)
    n_nonpne = n - len(reached)
    if n:
        lines = [f"{CANON_TEX[l]}: {c}/{n}" for l, c in
                 sorted(counts.items(), key=lambda kv: -kv[1])]
        if n_nonpne:
            lines.append(f"non-PNE: {n_nonpne}/{n}")
        ax.text(0.975, 0.045, "\n".join(lines), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6.5, linespacing=1.3,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.8",
                          lw=0.4, alpha=0.85), zorder=5)

    # --- limits --------------------------------------------------------------
    levels = list(ne_norm.values()) + [1.0]
    data_lo = min([(y / opt_R).min() for _, y in curves], default=min(levels))
    data_hi = max([(y / opt_R).max() for _, y in curves], default=max(levels))
    lo = min(min(levels), data_lo)
    hi = max(max(levels), data_hi)
    pad = 0.06 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)

    ax.grid(True, alpha=0.20, lw=0.4)
    ax.tick_params(length=2.5, pad=2)

    # --- NE labels, in the right margin of the panel -------------------------
    # Drawn on every panel: the levels differ from column to column (each
    # operating point has its own NE rewards and its own optimum), so a reader
    # comparing panels should not have to trace a level across the figure to
    # the rightmost column to find out which equilibrium it is.
    if label_levels:
        lbls = [l for l in CANON_ORDER if l in ne_norm]
        ypos = stagger([ne_norm[l] for l in lbls],
                       0.045 * (ax.get_ylim()[1] - ax.get_ylim()[0]))
        for l, y in zip(lbls, ypos):
            ax.text(1.012, y, CANON_TEX[l], transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=6,
                    color=LEVEL_GREY if by_seed
                          else shade(NE_COLORS[CANON_ORDER.index(l)], 0.30),
                    clip_on=False)

    return counts, n_nonpne


def build_figure(data, args, mode):
    fig, axes = plt.subplots(
        len(ALGO_ORDER), len(POINT_ORDER),
        figsize=(args.width, args.height),
        sharex="row", sharey="col",
    )
    axes = np.atleast_2d(axes)

    row_xmax = {algo: max((x[-1] for point in POINT_ORDER
                           for x, _ in data[(algo, point)]["curves"]),
                          default=1.0)
                for algo in ALGO_ORDER}
    exp_override = dict(zip(ALGO_ORDER, args.xscale_exp or []))

    all_seeds = sorted({sd for panel in data.values() for sd in panel["seeds"]})
    seed_colors = {sd: SEED_COLORS[i % len(SEED_COLORS)]
                   for i, sd in enumerate(all_seeds)}

    n_seeds, hit, any_nonpne = 0, set(), False
    for r, algo in enumerate(ALGO_ORDER):
        xscale, xunit = x_unit(row_xmax[algo], exp_override.get(algo))
        for c, point in enumerate(POINT_ORDER):
            ax = axes[r, c]
            panel = data[(algo, point)]
            n_seeds = max(n_seeds, len(panel["curves"]))

            if not panel["curves"]:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", fontsize=7, color="0.5")

            counts, n_nonpne = draw_panel(
                ax, panel["curves"], panel["seeds"], panel["labels"],
                panel["ne"], panel["opt"], xscale, args,
                label_levels=(args.label_levels == "all"
                              or c == len(POINT_ORDER) - 1),
                mode=mode, seed_colors=seed_colors)
            hit |= set(counts)
            any_nonpne |= bool(n_nonpne)

            if r == 0:
                ax.set_title(POINT_TITLES[point], pad=4)
            ax.set_xlabel(f"{args.xlabel} {xunit}".strip())
            if c == 0:
                ax.set_ylabel(r"Normalized reward $R(\mathbf{n})/R^{\star}$")
                ax.annotate(ALGO_TITLES[algo], xy=(-0.30, 0.5),
                            xycoords="axes fraction", rotation=90,
                            ha="center", va="center", fontweight="bold",
                            fontsize=9)

    if mode == "seed":
        handles = [Line2D([], [], color=seed_colors[sd], lw=1.0,
                          label=f"seed {sd}") for sd in all_seeds]
        legend_title = (f"colour: seed; the box gives which PNE the final "
                        f"profile attains "
                        f"(dashed lines: PNE reward levels; "
                        f"{n_seeds} seeds per panel)")
    else:
        handles = [Line2D([], [], color=NE_COLORS[i], lw=1.0,
                          label=CANON_TEX[l])
                   for i, l in enumerate(CANON_ORDER) if l in hit]
        # Named after the object the text already defines, rather than
        # stacked adjectives: "exact/pure/deterministic profile class" reads as
        # a synonym for "PNE", which is wrong — grey runs attain none.
        # The seed count is not written as n: n is the load vector here.
        legend_title = (r"colour: which PNE the final profile $\mathbf{a}^{L}$ "
                        r"attains; grey: none "
                        f"(dashed lines: PNE reward levels; "
                        f"{n_seeds} seeds per panel)")
    if any_nonpne:
        handles.append(Line2D([], [], color=NONPNE_COLOR, lw=0.7,
                              ls=NONPNE_DASH, label="non-PNE"))
    if mode != "seed":
        # The y axis is policy performance (for IPPO, averaged over independent
        # action samples) while the hue is a property of one deterministic
        # readout. Two different quantities, so the legend names the one the
        # colours encode instead of leaving it to be inferred from the swatches.
        # Carried by an invisible handle rather than the legend title so it
        # stays on the same row as the swatches; the handle slot leaves a small
        # gap at the left of the centred legend, which is the price of one row.
        handles.insert(0, Line2D([], [], ls="none", marker="none",
                                 label="Deterministic final action:"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               bbox_to_anchor=(0.5, 0.0), handlelength=1.6,
               columnspacing=1.2, borderaxespad=0.0)

    # The level labels sit just outside each panel's right spine, so labelling
    # every panel needs room between the columns: at the tighter spacing they
    # land on the next column's y tick labels.
    wspace = args.wspace
    if wspace is None:
        wspace = 0.24 if args.label_levels == "all" else 0.13
    fig.subplots_adjust(left=0.085, right=0.965, top=0.885, bottom=0.185,
                        wspace=wspace, hspace=0.42)
    return fig


# ----------------------------------------------------------------------------
def collect(args):
    manifest = load_manifest(args.profiles_csv) if args.profiles_csv else None
    data, missing = {}, []
    for algo in ALGO_ORDER:
        for point in POINT_ORDER:
            beta, gamma = POINTS[point]
            ne, opt_R, poa = compute_ne_rewards(beta, gamma)

            dir_path = os.path.join(REPO, ALGO_DIRS[algo].format(point=point))
            files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
            if not files:
                print(f"[WARN] {algo:5s} {point:4s}: no CSV in {dir_path}",
                      file=sys.stderr)

            curves, seeds, labels, profiles = [], [], [], []
            for fp in files:
                x, y, csv_prof = read_run(fp)
                if x.size < 2:
                    print(f"[WARN] {fp}: <2 usable rows, skipped", file=sys.stderr)
                    continue
                prof = resolve_profile(manifest, algo, point, fp,
                                       len(curves), csv_prof)
                if prof is None:
                    missing.append(fp)
                curves.append((x, smooth(y, args.smooth)))
                # runs are numbered 1..n in file order, not by their raw seed
                seeds.append(len(seeds) + 1)
                profiles.append(prof)
                labels.append(classify_exact(prof, ne))

            n_pne = sum(l is not None for l in labels)
            print(f"  {algo:5s} {point:4s}: {len(curves)} seeds, "
                  f"PoA = {poa:.3f}, opt = {opt_R:.2f}, "
                  f"exact PNE {n_pne}/{len(curves)}")
            for sd, prof, lab in zip(seeds, profiles, labels):
                print(f"      run {sd}: profile = {prof}, "
                      f"class = {lab or 'non-PNE'}")

            data[(algo, point)] = {"curves": curves, "seeds": seeds,
                                   "labels": labels, "profiles": profiles,
                                   "ne": ne, "opt": opt_R, "poa": poa}

    if missing:
        msg = ("no terminal pure profile found for:\n  "
               + "\n  ".join(missing)
               + "\nExact PNE classification needs the final joint action "
                 "profile. Supply it with --profiles-csv <summary.csv>, add a "
                 "load-vector column to the run logs, or drop a "
                 "final_profiles.json next to them. Pass --missing-profile grey "
                 "to plot these as non-PNE instead.")
        if args.missing_profile == "error":
            raise SystemExit("[ERROR] " + msg)
        print("[WARN] " + msg, file=sys.stderr)
    return data


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", default=os.path.join(REPO, "Results", "VE_batch", "figures"))
    p.add_argument("--name", default="convergence_6panel")
    p.add_argument("--formats", nargs="+", default=["pdf", "png"])
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--width", type=float, default=7.16, help="inches (IEEE 2-col)")
    p.add_argument("--height", type=float, default=4.8, help="inches")
    p.add_argument("--color-by", choices=["both", "seed", "basin"],
                   default="both",
                   help="what the curve colour encodes; 'both' writes one "
                        "figure per convention")
    p.add_argument("--label-levels", choices=["all", "right"], default="all",
                   help="which panels get the v0..v4 level labels in their "
                        "right margin ('right' = rightmost column only)")
    p.add_argument("--wspace", type=float, default=None,
                   help="horizontal gap between columns; default adapts to "
                        "--label-levels (0.24 for 'all', 0.13 for 'right')")
    p.add_argument("--seed-shade", type=float, default=0.35,
                   help="lightness spread between seeds within one basin "
                        "(0 = all seeds identical colour)")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="opacity of individual seed curves")
    p.add_argument("--xlabel", default="Training episodes",
                   help="x-axis label (the logs store 'episode', not env steps)")
    p.add_argument("--xscale-exp", type=int, nargs="+", default=None,
                   metavar="EXP",
                   help="decimal exponent of the x-axis multiplier, one per "
                        f"row in the order {ALGO_ORDER}; default is the order "
                        "of magnitude of the longest run in that row")
    p.add_argument("--smooth", type=int, default=1,
                   help="moving-average window over test-reward points (1 = off)")
    p.add_argument("--tail-frac", type=float, default=0.10,
                   help="fraction of the run used for the terminal marker "
                        "(display only; classification is by pure profile)")
    p.add_argument("--profiles-csv", default=None,
                   help="manifest CSV holding each run's terminal load vector "
                        "(e.g. summary_all_runs.csv)")
    p.add_argument("--missing-profile", choices=["error", "grey"],
                   default="error",
                   help="what to do when a run has no readable pure profile")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.outdir, exist_ok=True)

    modes = ["seed", "basin"] if args.color_by == "both" else [args.color_by]
    suffix = {"seed": "by_seed", "basin": "by_ne"}

    with plt.rc_context(RC):
        data = collect(args)
        for mode in modes:
            fig = build_figure(data, args, mode)
            stem = f"{args.name}_{suffix[mode]}" if len(modes) > 1 else args.name
            for ext in args.formats:
                out = os.path.join(args.outdir, f"{stem}.{ext}")
                fig.savefig(out, dpi=args.dpi, bbox_inches="tight",
                            pad_inches=0.02)
                print(f"Saved -> {out}")
            plt.close(fig)


if __name__ == "__main__":
    main()