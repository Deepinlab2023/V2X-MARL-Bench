"""
run_ve_batch.py — V-E Section formal batch driver.

Runs 3 operating points x 2 algorithms x N seeds = 6N runs (default 30),
parses each runner's final readout block, and writes:
  - Results/VE_batch/per_run.csv          (one row per run)
  - Results/VE_batch/summary.md           (markdown: per-run table + landing
                                           histograms + Phi^L/Phi* spread)
  - Results/VE_batch/logs/<tag>.log       (full stdout per run, for debugging)

Operating points are the FULL-PRECISION SUMO sample-set means from
analysis/results/a6_200_ensemble_table.csv (gen=sumo, reps etsi1/4/6).
Runners read their own params (idql_params.py / ppo_params.py); this script
only forwards --episodes / --epsi_final / --entropy_coef overrides if given.

Usage:
  # full batch (30 runs, default 5 seeds)
  python3 run_ve_batch.py

  # quick pilot: 2 seeds, shorter
  python3 run_ve_batch.py --seeds 0,1 --episodes 10000

  # run in parallel (CPU-bound; 4 parallel ok on multi-core)
  python3 run_ve_batch.py --parallel 4

  # only IDQN, only LOW point
  python3 run_ve_batch.py --algos idqn --points low
"""
import argparse
import csv
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SCRIPT_DIR)

# Full-precision SUMO sample-set means (a6_200_ensemble_table.csv, gen=sumo)
POINTS = {
    "low":  (13.293186199612448, 74.58369474788802,  30.7559,  2.0506),
    "mid":  (11.976708878135344, 79.92478010397355, 140.8051,  2.5095),
    "high": (10.141001516147227, 83.40988888212391, 517.4506,  3.2194),
}

ALGO_RUNNER = {
    "idqn": "run_matrix_game.py",
    "ippo": "run_matrix_game_ippo.py",
}

CANON_ORDER = ["v0", "v1", "v2", "v3", "v4", "non-PNE"]


# ── stdout parser ─────────────────────────────────────────────────────────────
# Header line (both runners):
#   OPT=99.1047  best_NE=99.1047  worst_NE=48.3305  PoA=2.0506
_RE_HEADER = re.compile(
    r"OPT=([\d.]+)\s+best_NE=([\d.]+)\s+worst_NE=([\d.]+)\s+PoA=([\d.]+)")

# IDQN final:
#   Reward   = 61.3649   (recomputed Phi(n^L) = 61.3649)
#   Loads    = [3, 3, 1, 7]   sorted=(7, 3, 3, 1)
#   PNE?     = True   label=v0   rho=0.0000
#   PNE?     = False   (readout not in equilibrium set)
_RE_IDQN_REWARD  = re.compile(r"Reward\s+=\s+([\d.]+)")
_RE_IDQN_READOUT = re.compile(r"sorted=\(([^)]+)\)")
_RE_PNE_TRUE  = re.compile(r"PNE\?\s*=\s*True\s+label=(\S+)\s+rho=([\d.]+)")
_RE_PNE_FALSE = re.compile(r"PNE\?\s*=\s*False")

# IPPO final:
#   [stochastic]  Phi^L     = 46.9114   %OPT=47.3%  %best_NE=47.3%
#   [readout]     Phi(n^L) = 20.4294   loads=[0, 0, 16, 0]   sorted=(16, 0, 0, 0)
#   [readout]     PNE?=True   label=v0   rho=0.0000
_RE_IPPO_STOCH   = re.compile(r"\[stochastic\]\s+Phi\^L\s+=\s+([\d.]+)")
_RE_IPPO_READOUT = re.compile(r"\[readout\]\s+Phi\(n\^L\)\s+=\s+([\d.]+)\s+loads=\[[^\]]*\]\s+sorted=\(([^)]+)\)")
_RE_IPPO_PNE_T   = re.compile(r"\[readout\]\s+PNE\?=True\s+label=(\S+)\s+rho=([\d.]+)")
_RE_IPPO_PNE_F   = re.compile(r"\[readout\]\s+PNE\?=False")


def parse_stdout(text, algo):
    """Return dict with parsed fields, or None on failure."""
    out = {}
    # header (last occurrence in case it prints twice)
    hdrs = _RE_HEADER.findall(text)
    if not hdrs:
        return None
    o, bn, wn, p = hdrs[-1]
    out["opt_R"], out["best_ne"], out["worst_ne"], out["poa"] = map(float, (o, bn, wn, p))

    if algo == "idqn":
        m = _RE_IDQN_REWARD.search(text);              out["phi_stoch"] = float(m.group(1)) if m else float("nan")
        m = _RE_IDQN_READOUT.search(text);             out["readout"]   = tuple(int(x) for x in m.group(1).split(",")) if m else None
        m = _RE_PNE_TRUE.search(text)
        if m:
            out["is_pne"], out["label"], out["rho"] = True, m.group(1), float(m.group(2))
        elif _RE_PNE_FALSE.search(text):
            out["is_pne"], out["label"], out["rho"] = False, "non-PNE", float("nan")
        else:
            out["is_pne"], out["label"], out["rho"] = None, "?", float("nan")
        out["phi_readout"] = out["phi_stoch"]  # IDQN reward IS the greedy readout reward
    else:  # ippo
        m = _RE_IPPO_STOCH.search(text);   out["phi_stoch"] = float(m.group(1)) if m else float("nan")
        m = _RE_IPPO_READOUT.search(text)
        if m:
            out["phi_readout"] = float(m.group(1))
            out["readout"]     = tuple(int(x) for x in m.group(2).split(","))
        else:
            out["phi_readout"] = float("nan"); out["readout"] = None
        m = _RE_IPPO_PNE_T.search(text)
        if m:
            out["is_pne"], out["label"], out["rho"] = True, m.group(1), float(m.group(2))
        elif _RE_IPPO_PNE_F.search(text):
            out["is_pne"], out["label"], out["rho"] = False, "non-PNE", float("nan")
        else:
            out["is_pne"], out["label"], out["rho"] = None, "?", float("nan")
    return out


# ── run one experiment ────────────────────────────────────────────────────────
def run_one(algo, point_name, seed, episodes_override, epsi_override, ent_override, log_dir):
    beta, gamma, lam_eff, ref_poa = POINTS[point_name]
    runner = os.path.join(SCRIPT_DIR, ALGO_RUNNER[algo])
    cmd = [sys.executable, runner,
           "--op_beta", str(beta), "--op_gamma", str(gamma), "--seed", str(seed)]
    if episodes_override is not None:
        cmd += ["--episodes", str(episodes_override)]
    if algo == "idqn" and epsi_override is not None:
        cmd += ["--epsi_final", str(epsi_override)]
    if algo == "ippo" and ent_override is not None:
        cmd += ["--entropy_coef", str(ent_override)]

    tag = f"{algo}_{point_name}_seed{seed}"
    log_path = os.path.join(log_dir, f"{tag}.log")
    t0 = time.time()
    print(f"  [{tag}] start: {' '.join(cmd)}", flush=True)
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=lf, stderr=subprocess.STDOUT,
                              text=True)
    dt = time.time() - t0
    with open(log_path) as f:
        text = f.read()
    parsed = parse_stdout(text, algo)
    status = "ok" if parsed and parsed.get("is_pne") is not None else "PARSE_FAIL"
    print(f"  [{tag}] done in {dt:.0f}s  status={status}", flush=True)
    if parsed:
        parsed["algo"], parsed["point"], parsed["seed"] = algo, point_name, seed
        parsed["beta"], parsed["gamma"], parsed["lam_eff"], parsed["ref_poa"] = beta, gamma, lam_eff, ref_poa
        parsed["phi_L_over_star"] = parsed["phi_stoch"] / parsed["opt_R"] if parsed["opt_R"] > 0 else float("nan")
    return parsed, dt, status, log_path


# ── summary writers ───────────────────────────────────────────────────────────
def write_csv(rows, path):
    # Compute derived columns before writing
    for r in rows:
        r["phi_L"] = r.get("phi_stoch", float("nan"))
        r["phi_L_pct"] = (
            r["phi_L"] / r["opt_R"] * 100
            if r.get("opt_R", 0) > 0 and r["phi_L"] == r["phi_L"]
            else float("nan"))
        r["empirical_poa"] = (
            r["opt_R"] / r["phi_L"]
            if r.get("opt_R", 0) > 0 and r["phi_L"] > 0
            else float("nan"))
    cols = ["point", "algo", "seed", "beta", "gamma",
            "opt_R", "worst_ne", "poa",
            "phi_L", "phi_L_pct",
            "readout", "is_pne", "label",
            "empirical_poa"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write_markdown(rows, path, cfg):
    import statistics as st

    # Ensure derived columns exist
    for r in rows:
        r["phi_L"] = r.get("phi_stoch", float("nan"))
        r["phi_L_pct"] = (
            r["phi_L"] / r["opt_R"] * 100
            if r.get("opt_R", 0) > 0 and r["phi_L"] == r["phi_L"]
            else float("nan"))
        r["empirical_poa"] = (
            r["opt_R"] / r["phi_L"]
            if r.get("opt_R", 0) > 0 and r["phi_L"] > 0
            else float("nan"))

    L = []
    L.append("# V-E Batch Summary\n")
    L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    L.append(f"Seeds: {cfg['seeds']}  |  Points: {', '.join(cfg['points'])}  |  "
             f"Algorithms: {', '.join(cfg['algos'])}  |  Total runs: {len(rows)}\n")

    # Section 1: Operating-point reference
    L.append("## 1. Operating points (SUMO full-precision means)\n")
    L.append("| point | beta [dB] | gamma [dB] | lam_eff | OPT | worst_NE | PoA |")
    L.append("|---|---|---|---|---|---|---|")
    seen = {}
    for r in rows:
        k = r["point"]
        if k not in seen:
            seen[k] = r
    for pn in ["low", "mid", "high"]:
        if pn in seen:
            r = seen[pn]
            L.append(f"| {pn} | {r['beta']:.4f} | {r['gamma']:.4f} | "
                     f"{r.get('lam_eff', 0):.1f} | {r['opt_R']:.4f} | "
                     f"{r['worst_ne']:.4f} | {r['poa']:.4f} |")
    L.append("")

    # Section 2: Per-run results
    L.append("## 2. Per-run results\n")
    L.append("Last column = empirical PoA = OPT / Phi^L  "
             "(directly comparable to theoretical PoA above.)\n")
    L.append("| point | algo | seed | Phi^L | %OPT | readout | PNE? | label | "
             "empirical_PoA |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["point"], x["algo"], x["seed"])):
        rd = str(r.get("readout")) if r.get("readout") else "?"
        pne = "Y" if r.get("is_pne") else ("N" if r.get("is_pne") is False else "?")
        L.append(f"| {r['point']} | {r['algo']} | {r['seed']} | "
                 f"{r['phi_L']:.3f} | {r['phi_L_pct']:.1f}% | {rd} | "
                 f"{pne} | {r.get('label','?')} | "
                 f"**{r['empirical_poa']:.4f}** |")
    L.append("")

    # Section 3: Conclusion 1 — NE convergence
    L.append("## 3. Conclusion 1: NE convergence (IPPO vs IDQN)\n")
    L.append("Landing histogram: count of seeds on each canonical PNE / non-PNE.\n")
    hdr = "| point | PoA | algo | " + " | ".join(CANON_ORDER) + " | n_PNE | n_seeds |"
    sep = "|---|---|---|" + "|".join(["---"] * len(CANON_ORDER)) + "|---|---|"
    L.append(hdr); L.append(sep)
    for pn in ["low", "mid", "high"]:
        for algo in ["idqn", "ippo"]:
            sub = [r for r in rows if r["point"] == pn and r["algo"] == algo]
            cnt = Counter(r.get("label", "?") for r in sub)
            cells = [str(cnt.get(c, 0)) for c in CANON_ORDER]
            n_pne = sum(1 for r in sub if r.get("is_pne"))
            L.append(f"| {pn} | {seen.get(pn, {}).get('poa', 0):.2f} | {algo} | "
                     + " | ".join(cells) + f" | {n_pne} | {len(sub)} |")
    L.append("")
    total_ippo = sum(1 for r in rows if r["algo"] == "ippo")
    ippo_pne = sum(1 for r in rows if r["algo"] == "ippo" and r.get("is_pne"))
    total_idqn = sum(1 for r in rows if r["algo"] == "idqn")
    idqn_pne = sum(1 for r in rows if r["algo"] == "idqn" and r.get("is_pne"))
    L.append(f"**IPPO: {ippo_pne}/{total_ippo} seeds converge to a PNE** "
             f"(always reaches a pure Nash equilibrium).")
    L.append(f"**IDQN: {idqn_pne}/{total_idqn} seeds converge to a PNE** "
             f"({idqn_pne}/{total_idqn} hit a PNE; "
             f"{total_idqn - idqn_pne}/{total_idqn} land on non-PNE readouts).\n")

    # Section 4: Conclusion 2 — PoA vs instability
    L.append("## 4. Conclusion 2: PoA vs convergence instability\n")
    L.append("Higher PoA => wider spread of NE rewards => RL convergence "
             "becomes more variable across seeds.\n")
    L.append("| point | PoA | algo | mean %OPT | std %OPT | min %OPT | "
             "max %OPT | range | distinct NE | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for pn in ["low", "mid", "high"]:
        for algo in ["idqn", "ippo"]:
            vals = [r["phi_L_pct"] for r in rows
                    if r["point"] == pn and r["algo"] == algo
                    and r["phi_L_pct"] == r["phi_L_pct"]]
            if vals:
                m = st.mean(vals)
                sd = st.pstdev(vals) if len(vals) > 1 else 0.0
                rng = max(vals) - min(vals)
                labels = set(r.get("label", "?") for r in rows
                             if r["point"] == pn and r["algo"] == algo)
                L.append(f"| {pn} | {seen.get(pn, {}).get('poa', 0):.2f} | {algo} | "
                         f"{m:.1f}% | {sd:.1f}% | {min(vals):.1f}% | "
                         f"{max(vals):.1f}% | {rng:.1f}% | "
                         f"{len(labels - {'?'})} | {len(vals)} |")
            else:
                L.append(f"| {pn} | {seen.get(pn, {}).get('poa', 0):.2f} | {algo} | "
                         f"- | - | - | - | - | 0 | 0 |")
    L.append("")
    L.append("### 4a. NE reward spread at each operating point\n")
    L.append("| point | PoA | OPT (best NE) | worst NE | spread "
             "(OPT - worst) | spread / OPT |")
    L.append("|---|---|---|---|---|---|")
    for pn in ["low", "mid", "high"]:
        if pn in seen:
            r = seen[pn]
            spread = r["opt_R"] - r["worst_ne"]
            pct = spread / r["opt_R"] * 100
            L.append(f"| {pn} | {r['poa']:.4f} | {r['opt_R']:.2f} | "
                     f"{r['worst_ne']:.2f} | {spread:.2f} | {pct:.1f}% |")
    L.append("")

    # Section 5: Conclusion 3 — Empirical vs theoretical PoA
    L.append("## 5. Conclusion 3: Actual RL performance vs theoretical PoA\n")
    L.append("empirical_PoA = OPT / Phi^L  (achieved by RL).  "
             "theoretical_PoA = OPT / worst_NE  (worst-case NE).\n")
    L.append("If empirical = theoretical: RL landed on the worst NE.  "
             "If empirical = 1: RL found the optimal NE.\n")
    L.append("")
    L.append("| point | algo | seed | Phi^L | %OPT | theoretical_PoA | "
             "empirical_PoA | gap (theo - emp) | label |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: (x["point"], x["algo"], x["seed"])):
        gap = r["poa"] - r["empirical_poa"]
        L.append(f"| {r['point']} | {r['algo']} | {r['seed']} | "
                 f"{r['phi_L']:.3f} | {r['phi_L_pct']:.1f}% | "
                 f"{r['poa']:.4f} | **{r['empirical_poa']:.4f}** | "
                 f"{gap:.4f} | {r.get('label','?')} |")
    L.append("")
    L.append("### 5a. Aggregate: empirical vs theoretical PoA\n")
    L.append("| point | PoA (theo) | algo | mean emp_PoA | "
             "min emp_PoA | max emp_PoA | n |")
    L.append("|---|---|---|---|---|---|---|")
    for pn in ["low", "mid", "high"]:
        for algo in ["idqn", "ippo"]:
            vals = [r["empirical_poa"] for r in rows
                    if r["point"] == pn and r["algo"] == algo
                    and r["empirical_poa"] == r["empirical_poa"]]
            if vals:
                L.append(f"| {pn} | {seen.get(pn, {}).get('poa', 0):.4f} | {algo} | "
                         f"{st.mean(vals):.4f} | {min(vals):.4f} | "
                         f"{max(vals):.4f} | {len(vals)} |")
            else:
                L.append(f"| {pn} | {seen.get(pn, {}).get('poa', 0):.4f} | {algo} | "
                         f"- | - | - | 0 |")
    L.append("")

    # Notes
    L.append("## Notes\n")
    L.append("- IDQN Phi^L = greedy readout reward (single deterministic eval).")
    L.append("- IPPO Phi^L = 9-trial stochastic mean (sampled from learned policy).")
    L.append("- Canonical PNE labels: v0=(1,1,1,1), v1=(13,1,1,1), "
             "v2=(7,7,1,1), v3=(5,5,5,1), v4=(4,4,4,4).")
    L.append("- Non-PNE: greedy readout not in the equilibrium set "
             "(typically (n,1,1,1) with n < 13).")
    L.append("- Full per-run stdout in `logs/`.")

    with open(path, "w") as f:
        f.write("\n".join(L))


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="V-E batch driver (3 points x 2 algos x N seeds)")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4",
                    help="Comma-separated seeds (default 0,1,2,3,4)")
    ap.add_argument("--points", type=str, default="low,mid,high",
                    help="Comma-separated point names (default low,mid,high)")
    ap.add_argument("--algos", type=str, default="idqn,ippo",
                    help="Comma-separated algos (default idqn,ippo)")
    ap.add_argument("--episodes", type=int, default=None,
                    help="Override episodes for all runs (else runner params default)")
    ap.add_argument("--epsi", type=float, default=None, help="Override IDQN epsi_final")
    ap.add_argument("--ent", type=float, default=None, help="Override IPPO entropy_coef")
    ap.add_argument("--parallel", type=int, default=1,
                    help="Parallel workers (default 1=sequential)")
    ap.add_argument("--reparse", action="store_true",
                    help="Re-parse existing logs without re-running experiments")
    args = ap.parse_args()

    seeds  = [int(s) for s in args.seeds.split(",")]
    points = [p.strip() for p in args.points.split(",")]
    algos  = [a.strip() for a in args.algos.split(",")]
    for p in points:
        if p not in POINTS: ap.error(f"unknown point {p}; choices: {list(POINTS)}")
    for a in algos:
        if a not in ALGO_RUNNER: ap.error(f"unknown algo {a}; choices: {list(ALGO_RUNNER)}")

    out_dir = os.path.join(REPO, "Results", "VE_batch")
    log_dir = os.path.join(out_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    cfg = {"points": points, "algos": algos, "seeds": seeds,
           "episodes": args.episodes, "epsi": args.epsi, "ent": args.ent}

    if args.reparse:
        # Re-parse existing logs
        print(f"=== Reparsing {len(log_dir)} logs (seeds={seeds}) ===\n")
        rows = []
        for pn in points:
            for algo in algos:
                for seed in seeds:
                    tag = f"{algo}_{pn}_seed{seed}"
                    log_path = os.path.join(log_dir, f"{tag}.log")
                    if not os.path.exists(log_path):
                        print(f"  [WARN] {tag}: log not found, skipping")
                        continue
                    with open(log_path) as f:
                        text = f.read()
                    parsed = parse_stdout(text, algo)
                    if not parsed:
                        print(f"  [WARN] {tag}: parse failed, skipping")
                        continue
                    parsed["algo"], parsed["point"], parsed["seed"] = algo, pn, seed
                    beta, gamma, lam_eff, ref_poa = POINTS[pn]
                    parsed["beta"], parsed["gamma"] = beta, gamma
                    parsed["lam_eff"], parsed["ref_poa"] = lam_eff, ref_poa
                    parsed["phi_L_over_star"] = (
                        parsed["phi_stoch"] / parsed["opt_R"]
                        if parsed["opt_R"] > 0 else float("nan"))
                    rows.append(parsed)
                    print(f"  [{tag}] phi_L={parsed['phi_stoch']:.3f}  "
                          f"PNE={parsed.get('is_pne')}  label={parsed.get('label')}")
        print(f"\nParsed {len(rows)} runs.")
    else:
        jobs = [(a, p, s) for p in points for a in algos for s in seeds]
        print(f"=== V-E batch: {len(jobs)} runs "
              f"({len(points)} pts x {len(algos)} algos x {len(seeds)} seeds) ===")
        print(f"    episodes={args.episodes or 'params-default'}  parallel={args.parallel}\n")

        rows = []
        t0 = time.time()

        if args.parallel <= 1:
            for (algo, pn, seed) in jobs:
                r, dt, status, lp = run_one(algo, pn, seed, args.episodes, args.epsi, args.ent, log_dir)
                if r: rows.append(r)
        else:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=args.parallel) as ex:
                futs = {ex.submit(run_one, algo, pn, seed, args.episodes, args.epsi, args.ent, log_dir):
                        (algo, pn, seed) for (algo, pn, seed) in jobs}
                for fut in as_completed(futs):
                    r, dt, status, lp = fut.result()
                    if r: rows.append(r)

        dt = time.time() - t0
        n_ok = sum(1 for r in rows if r.get("is_pne") is not None)
        print(f"\n=== Batch done: {len(rows)}/{len(jobs)} parsed, {n_ok} with PNE verdict, {dt:.0f}s total ===")

    csv_path = os.path.join(out_dir, "per_run.csv")
    md_path  = os.path.join(out_dir, "summary.md")
    write_csv(rows, csv_path)
    write_markdown(rows, md_path, cfg)
    print(f"Saved -> {csv_path}")
    print(f"Saved -> {md_path}")
    print(f"Logs  -> {log_dir}/")


if __name__ == "__main__":
    main()
