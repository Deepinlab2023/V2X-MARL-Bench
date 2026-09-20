"""
Price-of-Anarchy (PoA) analysis — C-V2X homogeneous baseline.

Identical-interest interference game: N V2V links each choose one of M
orthogonal subchannels or stay silent. Shared reward. Homogeneous
approximation: all links see the same (S, I, I_V2I) on every subchannel.

Model (per subchannel m, load n_m):
    R_m(n) = n · W · log2(1 + S / (σ²_eff + (n−1)·I))
    Δ_m(n) = R_m(n) − R_m(n−1)

NE three conditions (user spec):
    stay    : Δ(n_m)   ≥ 0  for every active m
    silence : Δ(n_m+1) ≤ 0  for every m, when silent agents exist
    switch  : Δ(n_m2+1) ≤ Δ(n_m1)  for all active m1 and all m2 ≠ m1

Usage:
    python analysis/poa_homogeneous.py                 # N=16, with V2I
    python analysis/poa_homogeneous.py --N 8           # N=8,  with V2I
    python analysis/poa_homogeneous.py --no-v2i        # N=16, without V2I
    python analysis/poa_homogeneous.py --both          # both variants
"""

import argparse
import numpy as np

# =============================================================================
# Physical constants
# Source: Configuration/env_params.py (line numbers noted)
# =============================================================================
M_SC     = 4          # number of subchannels              env_params.py:113
P_V_DBM  = 23.0       # V2V max TX power [dBm]             env_params.py:31
P_C_DBM  = 23.0       # V2I TX power [dBm]                 env_params.py:32
SIG2_DBM = -114.0     # thermal noise power [dBm]          env_params.py:24
G_VEH    = 3.0        # vehicle antenna gain [dBi]         env_params.py:18
NF_VEH   = 9.0        # vehicle noise figure [dB]          env_params.py:19
W_HZ     = 1e6        # subchannel bandwidth [Hz]          env_params.py:38
H_TX     = 1.5        # V2V TX antenna height [m]          env_params.py:26
H_RX     = 1.5        # V2V RX antenna height [m]          env_params.py:27
FC_GHZ   = 2.0        # carrier frequency [GHz]            env_params.py:132

# Representative V2I-to-V2V-Rx distance [m].
# Derivation: E[|X-Y|] = L/3 for uniform X,Y on [0,L].
# NFIG_k16 road: x in [-514, 1000] m → L = 1514 m → L/3 ≈ 505 m ≈ 500 m.
D_V2I = 500.0

# =============================================================================
# V2V pathloss — 3GPP TR 36.885 Urban Micro LOS
# Source: Environment/environment.py:180–193
# =============================================================================
_D_BP = 4 * H_TX * H_RX * FC_GHZ * 1e9 / 3e8   # breakpoint ≈ 60 m

def pl_v2v(d: float) -> float:
    """Pathloss [dB] for V2V link of distance d [m]."""
    if d <= 3.0:
        return 22.7 * np.log10(3.0) + 41.0 + 20.0 * np.log10(FC_GHZ / 5.0)
    elif d < _D_BP:
        return 22.7 * np.log10(d)   + 41.0 + 20.0 * np.log10(FC_GHZ / 5.0)
    else:
        return (40.0 * np.log10(d) + 9.45
                - 17.3 * np.log10(H_TX) - 17.3 * np.log10(H_RX)
                + 2.7  * np.log10(FC_GHZ / 5.0))

def rx_mw(tx_dbm: float, d: float) -> float:
    """Received signal power [mW] including antenna gains.
    Formula: environment.py:392–395  (P_tx - PL + 2·G_veh - NF_veh)
    V2I-to-V2V link uses the same V2V pathloss model: environment.py:319–320.
    """
    return 10.0 ** ((tx_dbm - pl_v2v(d) + 2.0 * G_VEH - NF_VEH) / 10.0)

# =============================================================================
# ETSI TR 103 766 highway density scenarios
# =============================================================================
ETSI_SCENARIOS = [
    dict(idx=1, lam=35,  L=2000, label="70 veh / 2000 m"),
    dict(idx=2, lam=62,  L=2000, label="125 veh / 2000 m"),
    dict(idx=3, lam=123, L=2000, label="245 veh / 2000 m"),
    dict(idx=4, lam=167, L=600,  label="100 veh / 600 m"),
    dict(idx=5, lam=333, L=600,  label="200 veh / 600 m"),
    dict(idx=6, lam=500, L=600,  label="300 veh / 600 m"),
]

# =============================================================================
# Channel parameter computation (homogeneous approximation)
# =============================================================================
def compute_channel_params(N: int, lam: float, use_v2i: bool):
    """
    Compute (S, I, sigma2_eff) [mW] for the homogeneous model.

    Geometry: N pairs uniformly spaced at d = max(3, 1000/lam) m (1D, Method C).
      Pair k: Tx at x = 2k·d,  Rx at x = (2k+1)·d.
      Intra-pair (signal) distance  : d
      Interferer Tx→victim Rx dist  : |2(j-i)-1|·d  for ordered pairs (i,j), i≠j
      Average over all N(N-1) pairs → representative I (homogeneous proxy).

    S     = P^V · G(d)
    I     = P^V · mean_{(i,j), i≠j} [ G(|2(j-i)-1|·d) ]
    I_V2I = P^c · G(D_V2I)   [V2V pathloss model, env.py:319–320]
    σ²_eff = σ² + I_V2I  (or just σ² when use_v2i=False)
    """
    d = max(3.0, 1000.0 / lam)

    S = rx_mw(P_V_DBM, d)

    total_I, count = 0.0, 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d_int = abs(2 * (j - i) - 1) * d
            total_I += rx_mw(P_V_DBM, d_int)
            count   += 1
    I = total_I / count

    sig2     = 10.0 ** (SIG2_DBM / 10.0)
    I_V2I    = rx_mw(P_C_DBM, D_V2I) if use_v2i else 0.0
    sig2_eff = sig2 + I_V2I

    return S, I, sig2_eff, I_V2I

# =============================================================================
# PoA model — reward and marginal gain
# =============================================================================
def R_m(n: int, S: float, s2: float, I: float) -> float:
    """Total subchannel reward for n co-channel agents [bps]."""
    if n == 0:
        return 0.0
    sinr = S / (s2 + max(0, n - 1) * I)
    return n * W_HZ * np.log2(1.0 + sinr)

def delta_n(n: int, S: float, s2: float, I: float) -> float:
    """Marginal gain Δ(n) = R(n) − R(n−1) of adding the n-th agent."""
    return R_m(n, S, s2, I) - R_m(n - 1, S, s2, I)

# =============================================================================
# Regime classification
# =============================================================================
def classify_regime(N: int, S: float, s2: float, I: float):
    """
    Classify the Δ(n) sign pattern for n = 1 … N.

    Regime I  : Δ(2) < 0 and Δ(n) ≤ 0 for all n ≥ 2  [single-occupancy dominant]
    Regime II : Δ(n) ≥ 0 for all n ≥ 2, non-increasing [concave sharing]
    Regime III: Δ non-monotone (+…-…+ rebound)          [low-load anomaly / bistable]

    Returns (label, sign_string, delta_list).
    """
    ds   = [delta_n(n, S, s2, I) for n in range(1, N + 1)]
    sgns = "".join("+" if v >= 0.0 else "-" for v in ds)

    if ds[1] < 0.0:
        label = "III" if any(v > 0.0 for v in ds[2:]) else "I"
    elif all(v >= 0.0 for v in ds[1:]):
        label = "II"
    else:
        label = "III"

    return label, sgns, ds

# =============================================================================
# NE enumeration
# =============================================================================
def _iter_sorted_dists(N_total: int, M_ch: int):
    """
    Yield all sorted (desc) tuples (n_1 ≥ … ≥ n_M ≥ 0) with sum ≤ N_total.
    n_silent = N_total − sum(tuple).
    """
    def _gen(rem, slots, max_val, cur):
        if slots == 0:
            yield tuple(cur)
            return
        for n in range(min(rem, max_val), -1, -1):
            yield from _gen(rem - n, slots - 1, n, cur + [n])

    yield from _gen(N_total, M_ch, N_total, [])

def is_ne(dist: tuple, N: int, S: float, s2: float, I: float,
          tol: float = 1e-9) -> bool:
    """
    Check whether sorted distribution dist = (n_1, …, n_M) is a pure NE.
    n_silent = N − sum(dist).

    Conditions:
      1. stay    : Δ(n_m)   ≥ 0  for every active channel (n_m > 0)
      2. silence : Δ(n_m+1) ≤ 0  for every channel m, when n_silent > 0
      3. switch  : Δ(n_m2+1) ≤ Δ(n_m1)  for all active m1, all m2 ≠ m1
    """
    n_silent = N - sum(dist)

    # Precompute marginal gains for all relevant n values
    max_n = max(dist) + 1 if dist else 1
    cache = {n: delta_n(n, S, s2, I) for n in range(1, max_n + 2)}

    # Condition 1: stay
    for nm in dist:
        if nm > 0 and cache[nm] < -tol:
            return False

    # Condition 2: silence
    if n_silent > 0:
        for nm in dist:
            if cache[nm + 1] > tol:
                return False

    # Condition 3: switch (agent on m1 considers moving to m2)
    for i1, nm1 in enumerate(dist):
        if nm1 == 0:
            continue
        d_stay = cache[nm1]
        for i2, nm2 in enumerate(dist):
            if i1 == i2:          # same physical channel — skip
                continue
            if cache[nm2 + 1] > d_stay + tol:
                return False

    return True

def find_ne_and_opt(N: int, M_ch: int, S: float, s2: float, I: float):
    """
    Enumerate all pure NE distributions and the social optimum.
    Exhaustive brute-force over all sorted (n_1 ≥ … ≥ n_M ≥ 0, sum ≤ N).

    Returns:
      ne_list    : list of (dist_tuple, R_total) for every NE, sorted by R ascending
      opt_dist   : distribution maximising R_total
      opt_R      : maximum R_total [bps]
      n_explored : total distributions checked
    """
    ne_list    = []
    opt_R      = -np.inf
    opt_dist   = None
    n_explored = 0

    for dist in _iter_sorted_dists(N, M_ch):
        n_explored += 1
        r_tot = sum(R_m(nm, S, s2, I) for nm in dist)

        if r_tot > opt_R:
            opt_R    = r_tot
            opt_dist = dist

        if is_ne(dist, N, S, s2, I):
            ne_list.append((dist, r_tot))

    ne_list.sort(key=lambda x: x[1])   # ascending: worst first
    return ne_list, opt_dist, opt_R, n_explored

# =============================================================================
# Main analysis loop
# =============================================================================
def run(N: int, use_v2i: bool):
    """Run PoA analysis for all 6 ETSI scenarios."""
    sig2    = 10.0 ** (SIG2_DBM / 10.0)
    I_V2I_0 = rx_mw(P_C_DBM, D_V2I)
    s2_ref  = sig2 + (I_V2I_0 if use_v2i else 0.0)
    tag     = f"with V2I  σ²_eff = {10*np.log10(s2_ref):.1f} dBm" if use_v2i \
              else f"no V2I   σ²    = {SIG2_DBM:.1f} dBm"

    SEP = "=" * 76
    print(f"\n{SEP}")
    print(f"  PoA Analysis | N = {N} agents | M = {M_SC} subchannels | {tag}")
    print(f"  P^V = {P_V_DBM} dBm | P^c = {P_C_DBM} dBm | "
          f"d_V2I = {D_V2I:.0f} m | d_bp = {_D_BP:.0f} m")
    if use_v2i:
        print(f"  I_V2I = {10*np.log10(I_V2I_0):.1f} dBm  "
              f"(σ²_eff / σ² = {10*np.log10(s2_ref/sig2):.1f} dB above thermal)")
    print(SEP)

    rows = []

    for sc in ETSI_SCENARIOS:
        lam = sc["lam"]
        d   = max(3.0, 1000.0 / lam)
        S, I, s2_eff, I_V2I = compute_channel_params(N, lam, use_v2i)

        S_db  = 10.0 * np.log10(S)
        I_db  = 10.0 * np.log10(I)
        s2_db = 10.0 * np.log10(s2_eff)

        rgm, sgns, ds = classify_regime(N, S, s2_eff, I)
        ne_list, opt_dist, opt_R, n_exp = find_ne_and_opt(N, M_SC, S, s2_eff, I)

        if ne_list:
            worst_dist, worst_R = ne_list[0]       # already sorted ascending
            best_ne_dist, best_ne_R = ne_list[-1]
            poa = opt_R / worst_R if worst_R > 1e-30 else np.inf
            pos = 1.0 if any(d_ == opt_dist for d_, _ in ne_list) else np.inf
        else:
            worst_R = worst_dist = best_ne_dist = best_ne_R = None
            poa = pos = np.nan

        rows.append(dict(
            idx=sc["idx"], lam=lam, label=sc["label"], d=d,
            S_db=S_db, I_db=I_db, s2_db=s2_db,
            snr=S_db - s2_db, sir=S_db - I_db,
            rgm=rgm, sgns=sgns, ds=ds,
            opt_dist=opt_dist, opt_R=opt_R,
            ne_count=len(ne_list), n_exp=n_exp,
            ne_list=ne_list,
            worst_dist=worst_dist, worst_R=worst_R,
            poa=poa, pos=pos,
        ))

        # Per-scenario detail
        print(f"\n  Scenario #{sc['idx']}  λ = {lam:3d} veh/km  "
              f"({sc['label']})   d_signal = {d:.1f} m")
        print(f"    S = {S_db:.1f} dBm  |  I = {I_db:.1f} dBm  |  "
              f"S/σ²_eff = {S_db - s2_db:.1f} dB  |  S/I = {S_db - I_db:.1f} dB")
        print(f"    Δ(n=1..{N}):  {sgns}   → Regime {rgm}")
        print(f"    Distributions explored: {n_exp}  (exhaustive)")
        print(f"    Optimal: {opt_dist}   n_silent={N-sum(opt_dist)}  "
              f"R = {opt_R/1e6:.3f} Mbps")
        if ne_list:
            print(f"    Pure NE ({len(ne_list)} total, worst→best):")
            for rank, (nd, nr) in enumerate(ne_list):
                ns = N - sum(nd)
                tags = []
                if nd == worst_dist:
                    tags.append("WORST")
                if nd == opt_dist:
                    tags.append("OPTIMAL")
                elif nr == best_ne_R:
                    tags.append("best NE")
                tag_s = f"  ← {', '.join(tags)}" if tags else ""
                print(f"      [{rank+1}] {nd}  n_silent={ns}  "
                      f"R={nr/1e6:.3f} Mbps{tag_s}")
            print(f"    PoA = {poa:.4f}   PoS = {pos:.4f}")
        else:
            print("    *** No pure NE found ***")

    # Summary table
    print(f"\n{SEP}")
    print(f"  SUMMARY TABLE | N = {N} | {tag}")
    print(SEP)
    hdr = (f"  {'#':>2}  {'λ':>5}  {'d':>5}  "
           f"{'S/σ²':>7}  {'S/I':>7}  "
           f"{'Δ-signs':>{N}}  {'Rgm':>4}  "
           f"{'#NE':>4}  {'PoA':>8}  {'PoS':>6}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in rows:
        poa_s = f"{r['poa']:.4f}" if not np.isnan(r['poa']) else "  N/A  "
        pos_s = f"{r['pos']:.4f}" if not np.isnan(r['pos']) else " N/A"
        print(f"  {r['idx']:>2}  {r['lam']:>5}  {r['d']:>5.1f}  "
              f"{r['snr']:>7.1f}  {r['sir']:>7.1f}  "
              f"{r['sgns']:>{N}}  {r['rgm']:>4}  "
              f"{r['ne_count']:>4}  {poa_s:>8}  {pos_s:>6}")

    return rows


# =============================================================================
# Entry point
# =============================================================================
if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="PoA analysis for C-V2X homogeneous baseline")
    ap.add_argument(
        "--N", type=int, default=16,
        help="Number of V2V agents (default: 16)")
    ap.add_argument(
        "--no-v2i", action="store_true",
        help="Use pure thermal noise σ² (no V2I interference)")
    ap.add_argument(
        "--both", action="store_true",
        help="Run both without V2I and with V2I (overrides --no-v2i)")
    args = ap.parse_args()

    if args.both:
        run(args.N, use_v2i=False)
        run(args.N, use_v2i=True)
    else:
        run(args.N, use_v2i=not args.no_v2i)
