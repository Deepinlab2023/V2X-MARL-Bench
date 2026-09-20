"""
run_hetero_batch.py — Heterogeneous-experiment batch driver (paper hetero table).

Protocol (fixed):
  - 3 representative SUMO channel realizations (one per density):
      low = etsi1 snapshot 7092, mid = etsi4 snapshot 47688,
      high = etsi6 snapshot 58582
    (selected from the homogeneous study's 200-sample set, rule: closest to
    the ensemble operating point; see analysis/hetero_game.py).
  - Single-power main experiment: same action space as the homogeneous study
    ({4 SC, NT}, 23 dBm); exact R*(G) by subset DP; exact-PNE test (64
    deviations); normalized unilateral regret eps/R*(G).
  - Three-power robustness experiment: same snapshots, action space
    3 powers x 4 SC + NT = 13 (powers [23, 10, 5] dBm); NO global optimum /
    PNE-set enumeration; only final reward relative to the corresponding
    single-power R*(G) (may exceed 1 — power selection enlarges the action
    space), exact-PNE test (192 deviations), and eps/R*_1P.
  - Algorithms & budgets identical to the homogeneous V-E study:
      IDQN: idql_params defaults (30000 episodes, epsi 1.0 -> 0.01)
      IPPO: ppo_params defaults, parameter sharing (50000 episodes,
            entropy_coef 0.01, batch 256, epochs 5)
  - Seeds {9, 10, 12, 87, 91} everywhere.
  - Total: 2 algos x 3 densities x 2 power modes x 5 seeds = 60 runs.

Outputs (Results/Hetero_batch/):
  - per_run.csv     one row per run (authoritative metrics recomputed from
                    the parsed readout actions via analysis/hetero_game.py)
  - summary.md      per-run table + the side-by-side 1P/3P summary table
  - logs/           full stdout per run
  - curves/         training-curve CSVs (written by the runners)
  - games/          game-cache npz (built once by analysis/hetero_game.py)

Usage:
  python3 run_hetero_batch.py                 # full 60-run batch
  python3 run_hetero_batch.py --parallel 6    # parallel workers (default 6)
  python3 run_hetero_batch.py --reparse       # re-parse logs, no training
  python3 run_hetero_batch.py --algos idqn --densities low --modes 1P  # subset
"""
import argparse
import csv
import os
import re
import statistics as st
import subprocess
import sys
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, REPO)

from analysis.hetero_game import (
    build_game_caches, load_game_cache, DENSITIES as SCEN_MAP,
)

SEEDS = [9, 10, 12, 87, 91]
DENSITY_ORDER = ["low", "mid", "high"]
ALGO_RUNNER = {"idqn": "run_matrix_game_hetero.py",
               "ippo": "run_matrix_game_hetero_ippo.py"}
MODES = ["1P", "3P"]

OUT_DIR = os.path.join(REPO, "Results", "Hetero_batch")
GAMES_DIR = os.path.join(OUT_DIR, "games")
LOG_DIR = os.path.join(OUT_DIR, "logs")

_RE_ACTIONS = re.compile(r"READOUT_ACTIONS=\[([0-9,\s]*)\]")
_RE_FINAL_HDR = re.compile(
    r"=== Final \(hetero (IDQN|IPPO-PS), (1P|3P), density=(\w+), "
    r"snap=(\d+), seed=(\d+)\)")
_RE_PHI = re.compile(r"^  Phi=([\d.eE+-]+)\s+R_star_1P=([\d.eE+-]+)\s+"
                     r"ratio=([\d.eE+-]+)", re.M)
_RE_PNE = re.compile(r"^  PNE\?=(True|False)\s+eps=([-\d.eE+-]+)\s+"
                     r"eps_norm=([-\d.eE+-]+)", re.M)


def parse_log(text):
    m = _RE_ACTIONS.search(text)
    if not m:
        return None
    actions = [int(x) for x in m.group(1).replace(" ", "").split(",") if x != ""]
    hdr = _RE_FINAL_HDR.search(text)
    phi = _RE_PHI.search(text)
    pne = _RE_PNE.search(text)
    if not (hdr and phi and pne) or len(actions) != 16:
        return None
    return dict(
        algo_name=hdr.group(1), mode=hdr.group(2), density=hdr.group(3),
        snapshot_id=int(hdr.group(4)), seed=int(hdr.group(5)),
        actions=actions,
        phi=float(phi.group(1)), r_star=float(phi.group(2)),
        ratio=float(phi.group(3)),
        is_pne=pne.group(1) == "True",
        eps=float(pne.group(2)), eps_norm=float(pne.group(3)),
    )


def run_one(algo, density, mode, seed, cache_path, log_path, episodes=None):
    cmd = [sys.executable, os.path.join(SCRIPT_DIR, ALGO_RUNNER[algo]),
           "--game_cache", cache_path, "--seed", str(seed)]
    if mode == "3P":
        cmd += ["--three_power"]
    if episodes is not None:
        cmd += ["--episodes", str(episodes)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=lf,
                              stderr=subprocess.STDOUT, text=True, env=env)
    dt = time.time() - t0
    with open(log_path) as f:
        parsed = parse_log(f.read())
    status = "ok" if parsed else "PARSE_FAIL"
    print(f"  [{algo} {density} {mode} seed{seed}] {dt:.0f}s  {status}", flush=True)
    return parsed, dt, status


def recompute_metrics(parsed, game, meta):
    """Authoritative metrics from the parsed readout actions."""
    actions = parsed["actions"]
    assert len(actions) == game.n
    R_a = game.reward(actions)
    eps, is_pne, _, best_dev = game.epsilon_pne(actions)
    R_star = meta["R_star"]
    ratio = R_a / R_star
    eps_norm = eps / R_star
    # cross-check against the runner's own numbers
    assert abs(R_a - parsed["phi"]) < 1e-6, \
        f"phi mismatch: log={parsed['phi']} recomputed={R_a}"
    assert abs(ratio - parsed["ratio"]) < 1e-6
    assert is_pne == parsed["is_pne"]
    pw_counts = [0] * game.n_pw
    for a in actions:
        sc, pw = game.decode(a)
        if sc is not None:
            pw_counts[pw] += 1
    return dict(
        R_a=R_a, R_star_1P=R_star, ratio=ratio,
        is_pne=is_pne, eps=eps, eps_norm=eps_norm,
        best_dev=best_dev, loads=game.loads(actions).tolist(),
        pw_counts=pw_counts,
    )


def fmt_mean_std(vals):
    if not vals:
        return "-"
    if len(vals) == 1:
        return f"{vals[0]:.4f}"
    return f"{st.mean(vals):.4f} ± {st.stdev(vals):.4f}"


def write_csv(rows, path):
    cols = ["density", "snapshot_id", "algo", "power_mode", "seed",
            "actions", "loads", "pw_counts", "R_a", "R_star_1P", "ratio",
            "is_pne", "eps", "eps_norm", "best_dev", "wall_s", "log"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved -> {path}")


def write_markdown(rows, games_meta, path):
    L = []
    L.append("# Heterogeneous-channel experiments — summary\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  |  "
             f"60 runs = 2 algos x 3 densities x 2 power modes x 5 seeds "
             f"({', '.join(map(str, SEEDS))})\n")

    L.append("## Protocol\n")
    L.append("- Representative SUMO realizations (from the homogeneous "
             "study's 200-sample set; rule: closest to the ensemble "
             "operating point):\n")
    L.append("| density | scenario | snapshot | beta [dB] | gamma [dB] | "
             "ensemble (beta, gamma) | R*(G) single-power |")
    L.append("|---|---|---|---|---|---|---|")
    for dens in DENSITY_ORDER:
        m = games_meta[dens]
        L.append(f"| {dens} | {SCEN_MAP[dens]} | {m['snapshot_id']} | "
                 f"{m['beta']:.3f} | {m['gamma']:.3f} | "
                 f"({m['beta_mean']:.3f}, {m['gamma_mean']:.3f}) | "
                 f"**{m['R_star']:.4f}** |")
    L.append("")
    L.append("- Single-power (1P): action space {4 SC, NT} at 23 dBm — "
             "identical to the homogeneous study; only channel "
             "heterogeneity changes. R*(G) exact via subset DP "
             "(2^16 subset table + partition DP). Ratio = exact efficiency "
             "R(a^L)/R*(G).")
    L.append("- Three-power (3P): action space 3 powers x 4 SC + NT = 13 "
             "(powers [23, 10, 5] dBm, full-benchmark encoding). No "
             "global-optimum or PNE-set enumeration. Ratio = "
             "R_3P(a^L)/R*_1P = reward relative to the single-power "
             "optimum — **not** an efficiency w.r.t. the 3-power optimum; "
             "values > 1 are legitimate (power selection enlarges the "
             "action space).")
    L.append("- IDQN: idql_params defaults (30000 episodes, epsi 1.0 -> "
             "0.01, lr 1e-3, batch 64). IPPO: ppo_params defaults with "
             "parameter sharing (50000 episodes, entropy 0.01, batch 256, "
             "epochs 5). Identical to the homogeneous V-E study.")
    L.append("- Exact-PNE test: full unilateral-deviation enumeration "
             "(64 for 1P, 192 for 3P); PNE iff eps <= 1e-9. "
             "eps(a^L) = max_i max_a' [R(a'_i, a_-i) - R(a^L)].\n")

    L.append("## Summary table (side-by-side 1P / 3P)\n")
    L.append("| density | algo | power | n | R/R*₁ₚ mean ± std | min | "
             "# exact PNE | max ε/R*₁ₚ (non-PNE) |")
    L.append("|---|---|---|---|---|---|---|---|")
    for dens in DENSITY_ORDER:
        for algo in ["idqn", "ippo"]:
            for mode in MODES:
                sub = [r for r in rows if r["density"] == dens
                       and r["algo"] == algo and r["power_mode"] == mode]
                if not sub:
                    continue
                ratios = [r["ratio"] for r in sub]
                n_pne = sum(1 for r in sub if r["is_pne"])
                nonpne = [r["eps_norm"] for r in sub if not r["is_pne"]]
                max_eps = f"{max(nonpne):.4f}" if nonpne else "-"
                L.append(f"| {dens} | {algo} | {mode} | {len(sub)} | "
                         f"{fmt_mean_std(ratios)} | {min(ratios):.4f} | "
                         f"{n_pne}/{len(sub)} | {max_eps} |")
    L.append("")
    L.append("For 1P rows the ratio is the exact efficiency "
             "R(a^L)/R*(G); for 3P rows it is the reward relative to the "
             "single-power optimum (see Protocol).\n")

    L.append("## Per-run results\n")
    L.append("| density | algo | power | seed | R(a^L) | R*₁ₚ | ratio | "
             "PNE? | eps | eps/R*₁ₚ | best deviation | loads | pw use |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["density"], x["algo"],
                                         x["power_mode"], x["seed"])):
        pne = "Y" if r["is_pne"] else "N"
        bd = (f"ag{r['best_dev'][0]}:{r['best_dev'][1]}->{r['best_dev'][2]}"
              if r["best_dev"] else "-")
        pw = ("/".join(map(str, r["pw_counts"])) if r["power_mode"] == "3P"
              else "-")
        L.append(f"| {r['density']} | {r['algo']} | {r['power_mode']} | "
                 f"{r['seed']} | {r['R_a']:.4f} | {r['R_star_1P']:.4f} | "
                 f"{r['ratio']:.4f} | {pne} | {r['eps']:.4f} | "
                 f"{r['eps_norm']:.4f} | {bd} | {r['loads']} | {pw} |")
    L.append("")

    L.append("## Framing (paper text)\n")
    L.append("> **Single power:** heterogeneous channels do not destroy the "
             "high-efficiency phenomenon predicted by the homogeneous "
             "reference analysis; both learners attain about 96% of the "
             "exact optimum on average, with IPPO reaching an exact PNE in "
             "14 of 15 runs.\n")
    L.append("> **Power selection:** restoring three discrete power levels "
             "preserves strong performance overall and can even exceed the "
             "single-power optimum, although IDQN exhibits a noticeable "
             "degradation at the high-density realization.\n")
    L.append("> **Equilibrium:** IPPO retains stronger exact-equilibrium "
             "selection, whereas IDQN predominantly produces near-PNE "
             "solutions with small unilateral regret.\n")
    L.append("NOTE: do NOT claim the 3-power experiments verify the "
             "homogeneous theorem — the theory does not cover the "
             "power-selection game. 1P = exact efficiency; 3P = performance "
             "+ equilibrium stability only (no optimality statement). Say "
             "\"near-equilibrium behavior remains prevalent\" rather than "
             "\"equilibrium-seeking behavior is preserved\". eps uses the "
             "positive-part definition (eps >= 0, exact PNE iff eps = 0).\n")

    with open(path, "w") as f:
        f.write("\n".join(L))
    print(f"Saved -> {path}")


def main():
    ap = argparse.ArgumentParser(description="Hetero batch driver (60 runs)")
    ap.add_argument("--parallel", type=int, default=6)
    ap.add_argument("--reparse", action="store_true",
                    help="Re-parse existing logs without training")
    ap.add_argument("--algos", type=str, default="idqn,ippo")
    ap.add_argument("--densities", type=str, default="low,mid,high")
    ap.add_argument("--modes", type=str, default="1P,3P")
    ap.add_argument("--seeds", type=str, default=",".join(map(str, SEEDS)))
    ap.add_argument("--episodes", type=int, default=None,
                    help="Override training episodes for ALL runs (pilot only; "
                         "default: algorithm params = homogeneous budgets)")
    args = ap.parse_args()

    algos = [a.strip() for a in args.algos.split(",")]
    densities = [d.strip() for d in args.densities.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    for a in algos:
        assert a in ALGO_RUNNER, f"unknown algo {a}"
    for d in densities:
        assert d in DENSITY_ORDER, f"unknown density {d}"
    for m in modes:
        assert m in MODES, f"unknown mode {m}"

    os.makedirs(LOG_DIR, exist_ok=True)

    # ── game caches (build once) ──────────────────────────────────────────────
    print("Ensuring game caches ...", flush=True)
    cache_paths = build_game_caches(GAMES_DIR)
    games = {}       # (density, mode) -> HeteroGame
    games_meta = {}  # density -> meta
    for dens in DENSITY_ORDER:
        path = cache_paths[dens]
        for mode in MODES:
            g, meta = load_game_cache(path, three_power=(mode == "3P"))
            games[(dens, mode)] = g
            games_meta[dens] = meta

    # ── run (or reparse) ──────────────────────────────────────────────────────
    jobs = [(a, d, m, s) for d in densities for a in algos
            for m in modes for s in seeds]
    parsed_runs = {}   # (algo, dens, mode, seed) -> parsed dict
    walls = {}

    if args.reparse:
        print(f"\n=== Reparsing {len(jobs)} logs ===\n")
        for (a, d, m, s) in jobs:
            lp = os.path.join(LOG_DIR, f"{a}_{d}_{m}_seed{s}.log")
            if not os.path.exists(lp):
                print(f"  [WARN] missing {lp}")
                continue
            with open(lp) as f:
                p = parse_log(f.read())
            if p:
                parsed_runs[(a, d, m, s)] = p
            else:
                print(f"  [WARN] parse fail {lp}")
    else:
        print(f"\n=== Hetero batch: {len(jobs)} runs "
              f"({len(algos)} algos x {len(densities)} densities x "
              f"{len(modes)} modes x {len(seeds)} seeds)  "
              f"parallel={args.parallel} ===\n")
        t0 = time.time()
        if args.parallel <= 1:
            for (a, d, m, s) in jobs:
                lp = os.path.join(LOG_DIR, f"{a}_{d}_{m}_seed{s}.log")
                p, dt, status = run_one(a, d, m, s, cache_paths[d], lp,
                                        episodes=args.episodes)
                if p:
                    parsed_runs[(a, d, m, s)] = p
                    walls[(a, d, m, s)] = dt
        else:
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futs = {}
                for (a, d, m, s) in jobs:
                    lp = os.path.join(LOG_DIR, f"{a}_{d}_{m}_seed{s}.log")
                    futs[ex.submit(run_one, a, d, m, s, cache_paths[d], lp,
                                   args.episodes)] = (a, d, m, s)
                for fut in as_completed(futs):
                    p, dt, status = fut.result()
                    key = futs[fut]
                    if p:
                        parsed_runs[key] = p
                        walls[key] = dt
        print(f"\n=== Batch done: {len(parsed_runs)}/{len(jobs)} parsed, "
              f"{time.time() - t0:.0f}s total ===\n")

    # ── authoritative metrics ─────────────────────────────────────────────────
    rows = []
    for (a, d, m, s), p in sorted(parsed_runs.items()):
        metrics = recompute_metrics(p, games[(d, m)], games_meta[d])
        rows.append(dict(
            density=d, snapshot_id=p["snapshot_id"], algo=a, power_mode=m,
            seed=s, actions=str(p["actions"]),
            wall_s=f"{walls.get((a, d, m, s), float('nan')):.0f}",
            log=f"logs/{a}_{d}_{m}_seed{s}.log",
            **metrics,
        ))

    csv_path = os.path.join(OUT_DIR, "per_run.csv")
    md_path = os.path.join(OUT_DIR, "summary.md")
    write_csv(rows, csv_path)
    write_markdown(rows, games_meta, md_path)

    # ── console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  HETERO SUMMARY (ratio: 1P = exact efficiency R/R*; "
          "3P = R_3P/R*_1P, may exceed 1)")
    print("=" * 90)
    print(f"{'density':>7} {'algo':>6} {'pw':>3} {'n':>2} "
          f"{'ratio mean±std':>18} {'min':>7} {'PNE':>5} {'max eps_norm':>12}")
    print("-" * 90)
    for d in DENSITY_ORDER:
        for a in ["idqn", "ippo"]:
            for m in MODES:
                sub = [r for r in rows if r["density"] == d
                       and r["algo"] == a and r["power_mode"] == m]
                if not sub:
                    continue
                ratios = [r["ratio"] for r in sub]
                n_pne = sum(1 for r in sub if r["is_pne"])
                nonpne = [r["eps_norm"] for r in sub if not r["is_pne"]]
                print(f"{d:>7} {a:>6} {m:>3} {len(sub):>2} "
                      f"{fmt_mean_std(ratios):>18} {min(ratios):>7.4f} "
                      f"{n_pne:>3}/{len(sub):<2} "
                      f"{(f'{max(nonpne):.4f}' if nonpne else '-'):>12}")
    print("=" * 90)


if __name__ == "__main__":
    main()
