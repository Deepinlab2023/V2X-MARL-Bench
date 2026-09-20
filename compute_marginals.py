"""
compute_marginals.py — Print Δ(n) table and K-increasing move costs at given density.

Δ(n) = n·r̄(n) − (n−1)·r̄(n−1)  where  r̄(n) = log2(1 + S/(σ²+(n−1)I))
     = marginal social welfare when the n-th agent joins a subchannel

Usage:
  python compute_marginals.py           # default λ=333
  python compute_marginals.py 167       # override density
"""
import math
import sys
sys.path.insert(0, ".")
from analysis.paper_utils import SIGMA2_MW

OPERATING_POINTS = {
     35: (13.50, 74.45),
     62: (13.30, 76.33),
    123: (12.71, 79.03),
    167: (11.94, 79.96),
    333: (10.89, 82.06),
    500: (10.19, 83.43),
}

# Reference values at λ=333 for self-check
_REF_333 = {1: 27.26, 2: -19.7987, 3: 1.0449, 9: 0.3842, 10: 0.3369}


def r_bar(n, S, I, sigma2):
    """Per-link rate r̄(n) = log2(1 + S/(σ²+(n−1)I))."""
    if n <= 0:
        return 0.0
    return math.log2(1.0 + S / (sigma2 + (n - 1) * I))


def Delta(n, S, I, sigma2):
    """Marginal social welfare Δ(n) = n·r̄(n) − (n−1)·r̄(n−1)."""
    rn  = r_bar(n,     S, I, sigma2)
    rn1 = r_bar(n - 1, S, I, sigma2)   # r_bar(0) = 0 by convention
    return n * rn - (n - 1) * rn1


def main(density=333):
    beta_db, gamma_db = OPERATING_POINTS[density]
    S  = SIGMA2_MW * 10.0 ** (gamma_db / 10.0)
    I  = SIGMA2_MW * 10.0 ** ((gamma_db - beta_db) / 10.0)
    s2 = SIGMA2_MW

    print(f"λ={density}  β={beta_db} dB  γ={gamma_db} dB")
    print(f"S/σ²={S/s2:.4e}   I/σ²={I/s2:.4e}")
    print()

    # ── Δ(n) table ──────────────────────────────────────────────────────────────
    print(f"{'n':>4}  {'Δ(n)=n·r̄(n)−(n−1)·r̄(n−1)':>32}")
    print("-" * 38)
    deltas = {}
    for n in range(1, 18):
        d = Delta(n, S, I, s2)
        deltas[n] = d
        print(f"{n:>4}  {d:>32.4f}")
    print()

    # ── Self-check against reference values (λ=333 only) ────────────────────────
    if density == 333:
        print("Self-check vs reference values (λ=333):")
        ok = True
        for n, ref in _REF_333.items():
            got = deltas[n]
            err = abs(got - ref)
            status = "✓" if err < 0.01 else "✗"
            print(f"  Δ({n:2d}) computed={got:+.4f}  ref={ref:+.4f}  err={err:.4f}  {status}")
            if err >= 0.01:
                ok = False
        if ok:
            print("  All reference values match. ✓")
        else:
            print("  *** MISMATCH — check OPERATING_POINTS or SIGMA2_MW ***")
        print()

    # ── Barrier B = |Δ(2)| ──────────────────────────────────────────────────────
    d1 = deltas[1]
    d2 = deltas[2]
    d3 = deltas[3]
    B  = -d2   # d2 is negative
    print(f"Δ(1)={d1:.4f}  Δ(2)={d2:.4f}  Δ(3)={d3:.4f}  B=|Δ(2)|={B:.4f}")
    print()

    # ── Allowed phi_drop bands (analytically derived) ───────────────────────────
    print("Allowed unilateral phi_drop bands:")
    print(f"  Band 1 (dense-channel exit):       (0, {d3:.4f}]")
    print(f"  Band 2 (K-increasing, enter sgl):  [{B:.4f}, {B+d3:.4f}]")
    print(f"  Band 3 (singleton exit):           [{d1-d3:.4f}, {d1:.4f}]")
    print(f"  Band 4 (singleton→singleton):      {{{d1+B:.4f}}}")
    print(f"  Forbidden interval:                ({d3:.4f}, {B:.4f})  ← any value here = BUG")
    print()

    # ── K-increasing c=1 move costs: agent moves from load-n_m channel to singleton
    #    phi_drop_W = Δ(n_m) − Δ(2) = Δ(n_m) + B
    # ──────────────────────────────────────────────────────────────────────────
    print("K-increasing (c=1) cost  [Δ(n_m) + B]  for entering a singleton:")
    print(f"{'n_m':>5}  {'Δ(n_m)':>10}  {'phi_drop':>10}  note")
    print("-" * 52)
    for n_m in range(1, 17):
        dm   = deltas.get(n_m, Delta(n_m, S, I, s2))
        drop = dm - d2          # = Δ(n_m) + B (since d2 < 0)
        note = ""
        if drop > d3 + 0.005 and drop < B - 0.005:
            note = "  ← FORBIDDEN BAND (should not occur)"
        elif drop >= B - 0.005:
            note = "  ← ≥ B  (K↑, φ drops by ≥ B)"
        elif drop > 0:
            note = "  ← small positive"
        else:
            note = "  ← negative (φ rises)"
        print(f"{n_m:>5}  {dm:>10.4f}  {drop:>10.4f}  {note}")

    # ── SC→NT lone-agent (singleton exits to silence, K unchanged) ──────────────
    print()
    print(f"Singleton→silence (K unchanged): phi_drop = Δ(1) = {d1:.4f}")
    print(f"  ∈ Band 3 [{d1-d3:.4f}, {d1:.4f}]  ✓")

    # ── Cross-check: singleton moves to dense channel (Band 3) ──────────────────
    print()
    print("Singleton→dense-channel costs  [Δ(1) − Δ(n_dest+1)]  (Band 3):")
    for n_dest in [2, 3, 4, 9, 10, 11]:
        d_dest = deltas.get(n_dest + 1, Delta(n_dest + 1, S, I, s2))
        drop   = d1 - d_dest
        print(f"  n_dest={n_dest}: Δ(1)−Δ({n_dest+1}) = {d1:.4f}−{d_dest:.4f} = {drop:.4f}")


if __name__ == "__main__":
    density = int(sys.argv[1]) if len(sys.argv) > 1 else 333
    main(density)
