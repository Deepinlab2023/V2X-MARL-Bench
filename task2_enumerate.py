"""
task2_enumerate.py — Deterministic NE enumeration at the λ=333 operating point.

Outputs:
  1. Phi*, Phi^NE_worst, Phi(13,1,1,1) and all K<=kappa welfare bounds
  2. Full NE list with K, Phi, structure
  3. Structural assertion: K<=1 NE should be exactly {(1,1,1,1), (13,1,1,1)}

Runtime: a few seconds. No simulation, no ML.
"""
import math
import sys
sys.path.insert(0, ".")
from analysis.paper_utils import SIGMA2_MW, find_opt_and_ne, classify_4, n_plus_exact

N, M = 16, 4
DENSITY = 333
BETA_DB, GAMMA_DB = 10.89, 82.06

S  = SIGMA2_MW * 10.0 ** (GAMMA_DB / 10.0)
I  = SIGMA2_MW * 10.0 ** ((GAMMA_DB - BETA_DB) / 10.0)
s2 = SIGMA2_MW


def phi_welfare(dist):
    """Social welfare Phi(n) = sum_m n_m * r_bar(n_m)."""
    total = 0.0
    for nm in dist:
        if nm > 0:
            total += nm * math.log2(1.0 + S / (s2 + (nm - 1) * I))
    return total


def K(dist):
    return sum(1 for nm in dist if nm >= 2)


def main():
    # ── Operating point info ─────────────────────────────────────────────────────
    np_ = n_plus_exact(S, I, s2)
    regime = classify_4(S, I, s2)
    print(f"Operating point: λ={DENSITY}, β={BETA_DB} dB, γ={GAMMA_DB} dB")
    print(f"n⁺ = {np_}   Regime = {regime}")
    print()

    # ── Enumeration ──────────────────────────────────────────────────────────────
    opt_dist, opt_R_ch, ne_list = find_opt_and_ne(N, M, S, I, s2)

    # Recompute everything with social welfare Phi (not channel-sum R)
    opt_phi = phi_welfare(opt_dist)
    ne_phi  = [(d, phi_welfare(d)) for d, _ in ne_list]
    ne_phi.sort(key=lambda x: x[1])   # worst NE first

    worst_ne_phi = ne_phi[0][1]  if ne_phi else float("nan")
    best_ne_phi  = ne_phi[-1][1] if ne_phi else float("nan")

    print(f"{'='*55}")
    print(f"  Phi*                    = {opt_phi:.4f}   dist={opt_dist}")
    print(f"  Phi_worst_NE            = {worst_ne_phi:.4f}")
    print(f"  Phi_best_NE             = {best_ne_phi:.4f}")
    print(f"  PoA (Phi*/Phi_worst_NE) = {opt_phi/worst_ne_phi:.4f}")
    print(f"  Total NE count          = {len(ne_phi)}")
    print(f"{'='*55}")
    print()

    # ── Phi(13,1,1,1) ────────────────────────────────────────────────────────────
    stall_dist = (13, 1, 1, 1)
    phi_stall  = phi_welfare(stall_dist)
    print(f"Phi(13,1,1,1) = {phi_stall:.4f}")
    print(f"  learning-selected PoA = Phi*/Phi(13,1,1,1) = {opt_phi/phi_stall:.4f}")
    print()

    # ── K<=kappa welfare bounds ────────────────────────────────────────────────
    print("K≤κ constrained welfare  Phi_kappa = best Phi among NE with K≤kappa:")
    print(f"  {'kappa':>5}  {'best NE Phi':>12}  {'NE count':>9}  {'dist':}")
    print(f"  {'-'*60}")
    for kappa in range(M + 1):
        subset = [(d, phi) for d, phi in ne_phi if K(d) <= kappa]
        if subset:
            best_d, best_phi = max(subset, key=lambda x: x[1])
            print(f"  {kappa:>5}  {best_phi:>12.4f}  {len(subset):>9}  {best_d}")
        else:
            print(f"  {kappa:>5}  {'—':>12}  {0:>9}  (no NE with K≤{kappa})")
    print()

    # ── Full NE list ─────────────────────────────────────────────────────────────
    print(f"All {len(ne_phi)} pure NEs (sorted worst-first):")
    print(f"  {'dist':20}  {'K':>3}  {'Phi':>10}  {'PoA contribution'}")
    print(f"  {'-'*55}")
    for d, phi in ne_phi:
        poa_contr = opt_phi / phi if phi > 0 else float("inf")
        print(f"  {str(d):20}  {K(d):>3}  {phi:>10.4f}  {poa_contr:.4f}")
    print()

    # ── Structural assertion: K<=1 NEs ───────────────────────────────────────────
    print("STRUCTURAL ASSERTION (n⁺=3, Regime III):")
    print("  When silent agents exist, Condition 3 forces every channel load = 1.")
    print("  Therefore K≤1 NEs should be exactly {(1,1,1,1), (13,1,1,1)}.")
    print()

    k1_ne = [(d, phi) for d, phi in ne_phi if K(d) <= 1]
    print(f"  Found {len(k1_ne)} NE(s) with K≤1:")
    for d, phi in k1_ne:
        silent = N - sum(d)
        print(f"    {d}  K={K(d)}  silent={silent}  Phi={phi:.4f}")

    expected = {(1,1,1,1), (13,1,1,1)}
    found    = {d for d, _ in k1_ne}
    if found == expected:
        print()
        print("  ✓ ASSERTION PASSED: K≤1 NE set = {(1,1,1,1), (13,1,1,1)}")
    else:
        extra   = found - expected
        missing = expected - found
        print()
        print("  ✗ ASSERTION FAILED:")
        if extra:
            print(f"    Unexpected NEs: {extra}")
            print("    → Theory of equilibrium structure is WRONG. Stop and report.")
        if missing:
            print(f"    Missing expected NEs: {missing}")


if __name__ == "__main__":
    main()
