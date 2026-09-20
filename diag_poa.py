"""
Diagnostic script for poa_homogeneous.py — 4 疑点.
Reads constants/functions from poa_homogeneous.py; does NOT modify it.
Run: python3 diag_poa.py
"""

import sys
import math
import numpy as np

# ---------------------------------------------------------------------------
# Pull shared constants / helpers from the main module (no copy-paste drift)
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")
from poa_homogeneous import (
    M_SC, P_V_DBM, P_C_DBM, SIG2_DBM, G_VEH, NF_VEH, W_HZ, D_V2I,
    _D_BP, pl_v2v, rx_mw, R_m, delta_n,
    compute_channel_params, find_ne_and_opt,
    ETSI_SCENARIOS,
)

SEP  = "=" * 76
SEP2 = "-" * 76

N  = 16
M  = M_SC   # 4

sig2     = 10.0 ** (SIG2_DBM / 10.0)
I_V2I_0  = rx_mw(P_C_DBM, D_V2I)
s2_noV2I = sig2
s2_V2I   = sig2 + I_V2I_0


# ===========================================================================
# 疑点 1 — S/I 被几何焊死
# ===========================================================================
print(f"\n{SEP}")
print("  疑点 1(a)  d_sig vs d_int1 across densities")
print(SEP)
hdr = f"  {'λ':>5}  {'d_sig':>7}  {'d_int1':>8}  {'same?':>6}  {'S(dBm)':>8}  {'I(dBm)':>8}  {'S/I(dB)':>8}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))

for sc in ETSI_SCENARIOS:
    lam  = sc["lam"]
    d    = max(3.0, 1000.0 / lam)
    # nearest-neighbour interferer: j-i = 1  →  |2·1-1|·d = d
    d_int1 = abs(2 * 1 - 1) * d
    S  = rx_mw(P_V_DBM, d)
    # use only nearest pair for illustration; full avg:
    S_db     = 10.0 * np.log10(S)
    I_near   = rx_mw(P_V_DBM, d_int1)
    I_near_db = 10.0 * np.log10(I_near)
    same = "YES ←" if abs(d - d_int1) < 1e-9 else "no"
    print(f"  {lam:>5}  {d:>7.1f}  {d_int1:>8.1f}  {same:>6}  "
          f"{S_db:>8.1f}  {I_near_db:>8.1f}  {S_db - I_near_db:>8.2f}")

print("\n  (I above = nearest-neighbour only. Full-average I from main run shown next.)")
print()
for sc in ETSI_SCENARIOS:
    lam = sc["lam"]
    S, I, _, _ = compute_channel_params(N, lam, False)
    S_db = 10*np.log10(S); I_db = 10*np.log10(I)
    d = max(3.0, 1000.0/lam)
    print(f"  λ={lam:>3}  d={d:.1f}m  S={S_db:.2f}dBm  I_avg={I_db:.2f}dBm  S/I_avg={S_db-I_db:.2f}dB")

# ---------------------------
print(f"\n{SEP}")
print("  疑点 1(b)  解耦对照: 固定 d_sig=15m, 干扰几何随密度缩放")
print(SEP)
D_SIG_FIXED = 15.0
S_fixed     = rx_mw(P_V_DBM, D_SIG_FIXED)
S_fixed_db  = 10.0 * np.log10(S_fixed)

# Wide table: current vs decoupled
hdr2 = (f"  {'λ':>5}  "
        f"{'[curr] d':>9}  {'S_dBm':>7}  {'I_dBm':>7}  {'S/I_dB':>7}  "
        f"{'[decoup] d_sig':>14}  {'S_dBm':>7}  {'I_dBm':>7}  {'S/I_dB':>7}")
print(hdr2)
print("  " + "-" * (len(hdr2) - 2))

for sc in ETSI_SCENARIOS:
    lam = sc["lam"]
    d   = max(3.0, 1000.0 / lam)

    # Current
    S_c, I_c, _, _ = compute_channel_params(N, lam, False)

    # Decoupled: S from fixed 15m; I uses actual spacing d for interferer geometry
    total_I, count = 0.0, 0
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            d_int = abs(2 * (j - i) - 1) * d   # interferer geometry still uses lam
            total_I += rx_mw(P_V_DBM, d_int)
            count   += 1
    I_dec = total_I / count

    S_c_db  = 10*np.log10(S_c)
    I_c_db  = 10*np.log10(I_c)
    I_d_db  = 10*np.log10(I_dec)

    print(f"  {lam:>5}  "
          f"{d:>9.1f}  {S_c_db:>7.2f}  {I_c_db:>7.2f}  {S_c_db-I_c_db:>7.2f}  "
          f"{D_SIG_FIXED:>14.1f}  {S_fixed_db:>7.2f}  {I_d_db:>7.2f}  {S_fixed_db-I_d_db:>7.2f}")

print(f"\n  [decoup] S is fixed at d_sig={D_SIG_FIXED}m for all rows; I uses actual lam spacing.")


# ===========================================================================
# 疑点 2 — 均值化 I 在 worst NE (4,4,4,4) 下的失真
# ===========================================================================
print(f"\n{SEP}")
print("  疑点 2  均值化 I vs 真实 pairwise — worst NE (4,4,4,4)")
print(SEP)

# Use λ=35 (most spread out) as primary; also λ=333 (clamped to 3m)
for sc in [ETSI_SCENARIOS[0], ETSI_SCENARIOS[4]]:
    lam = sc["lam"]
    d   = max(3.0, 1000.0 / lam)
    S, I_avg, s2_eff, _ = compute_channel_params(N, lam, True)  # with V2I

    load_per_ch = 4   # worst NE (4,4,4,4) — 4 agents per channel, 0 silent

    # (a) Homogeneous formula: each occupant gets (load-1)*I_avg
    R_homo = 4 * R_m(load_per_ch, S, s2_eff, I_avg)   # 4 channels × R_m(4)

    def true_ch_reward(link_indices, d_spacing, s2):
        """Reward for one channel whose occupants are the given link indices."""
        r = 0.0
        for i in link_indices:
            intra_I = sum(
                rx_mw(P_V_DBM, abs(2 * (j - i) - 1) * d_spacing)
                for j in link_indices if j != i
            )
            sinr = S / (s2 + intra_I)
            r += W_HZ * math.log2(1.0 + sinr)
        return r

    # (b1) Consecutive: ch1→{0,1,2,3}, ch2→{4,5,6,7}, ch3→{8,9,10,11}, ch4→{12,13,14,15}
    groups_consec = [list(range(k*4, k*4+4)) for k in range(4)]
    R_consec = sum(true_ch_reward(g, d, s2_eff) for g in groups_consec)

    # (b2) Uniform spacing: ch1→{0,4,8,12}, ch2→{1,5,9,13}, ...
    groups_uniform = [list(range(k, 16, 4)) for k in range(4)]
    R_uniform = sum(true_ch_reward(g, d, s2_eff) for g in groups_uniform)

    def reldiff(a, b):
        return abs(a - b) / a * 100.0

    print(f"\n  λ = {lam} veh/km   d = {d:.1f} m   S = {10*np.log10(S):.1f} dBm   "
          f"I_avg = {10*np.log10(I_avg):.1f} dBm")
    print(f"  (a) Homogeneous  R = {R_homo/1e6:.4f} Mbps")
    print(f"  (b1) Consecutive {groups_consec}  R = {R_consec/1e6:.4f} Mbps   "
          f"rel-diff vs (a) = {reldiff(R_homo, R_consec):.2f}%")
    print(f"  (b2) Uniform-gap {groups_uniform}  R = {R_uniform/1e6:.4f} Mbps   "
          f"rel-diff vs (a) = {reldiff(R_homo, R_uniform):.2f}%")

    # Also show per-link SINR spread for consecutive case (worst/best)
    sinrs_consec = []
    for g in groups_consec:
        for i in g:
            intra_I = sum(rx_mw(P_V_DBM, abs(2*(j-i)-1)*d) for j in g if j != i)
            sinrs_consec.append(10*np.log10(S / (s2_eff + intra_I)))
    print(f"  SINR spread (consec, all 16 links): "
          f"min={min(sinrs_consec):.1f} dB  max={max(sinrs_consec):.1f} dB  "
          f"std={np.std(sinrs_consec):.2f} dB")

    sinrs_uniform = []
    for g in groups_uniform:
        for i in g:
            intra_I = sum(rx_mw(P_V_DBM, abs(2*(j-i)-1)*d) for j in g if j != i)
            sinrs_uniform.append(10*np.log10(S / (s2_eff + intra_I)))
    print(f"  SINR spread (uniform, all 16 links): "
          f"min={min(sinrs_uniform):.1f} dB  max={max(sinrs_uniform):.1f} dB  "
          f"std={np.std(sinrs_uniform):.2f} dB")


# ===========================================================================
# 疑点 3 — d_V2I 敏感性
# ===========================================================================
print(f"\n{SEP}")
print("  疑点 3  d_V2I sensitivity  (λ=35 and λ=333 representative)")
print(SEP)

D_V2I_SWEEP = [250.0, 375.0, 500.0, 625.0, 750.0]
lam_rep = [35, 333]

hdr3 = (f"  {'d_V2I':>7}  {'I_V2I(dBm)':>11}  {'σ²_eff/σ²(dB)':>14}  "
        f"{'PoA λ=35':>10}  {'PoA λ=333':>10}")
print(hdr3)
print("  " + "-" * (len(hdr3) - 2))

for dv2i in D_V2I_SWEEP:
    i_v2i     = rx_mw(P_C_DBM, dv2i)
    i_v2i_db  = 10*np.log10(i_v2i)
    s2_eff_v  = sig2 + i_v2i
    ratio_db  = 10*np.log10(s2_eff_v / sig2)

    poas = []
    for lam in lam_rep:
        S, I, _, _ = compute_channel_params(N, lam, False)   # get S, I
        s2_here = sig2 + i_v2i                               # override σ²_eff
        ne_list, _, opt_R, _ = find_ne_and_opt(N, M, S, s2_here, I)
        if ne_list:
            worst_R = ne_list[0][1]
            poa = opt_R / worst_R
        else:
            poa = float("nan")
        poas.append(poa)

    print(f"  {dv2i:>7.0f}  {i_v2i_db:>11.2f}  {ratio_db:>14.2f}  "
          f"{poas[0]:>10.4f}  {poas[1]:>10.4f}")

print(f"\n  (* 500m row is the current baseline)")


# ===========================================================================
# 自检 — λ=35 with-V2I: (13,1,1,1) vs (1,1,1,1,s=12)  0.4% gap
# ===========================================================================
print(f"\n{SEP}")
print("  自检  λ=35 with-V2I: opt (13,1,1,1) vs best-NE (1,1,1,1,s=12)")
print(SEP)

lam = 35
S35, I35, s2_35, _ = compute_channel_params(N, lam, True)

def R_total(dist, S, s2, I):
    return sum(R_m(n, S, s2, I) for n in dist)

d_opt   = (13, 1, 1, 1)
d_bestNE = (1,  1, 1, 1)   # n_silent = 12

R_opt     = R_total(d_opt,    S35, s2_35, I35)
R_bestNE  = R_total(d_bestNE, S35, s2_35, I35)
gap_pct   = (R_opt - R_bestNE) / R_opt * 100.0

print(f"\n  S = {10*np.log10(S35):.4f} dBm   I = {10*np.log10(I35):.4f} dBm   "
      f"σ²_eff = {10*np.log10(s2_35):.4f} dBm")
print(f"\n  R{d_opt}   = {R_opt:.6f} bps  ({R_opt/1e6:.6f} Mbps)")
print(f"  R{d_bestNE}    = {R_bestNE:.6f} bps  ({R_bestNE/1e6:.6f} Mbps)")
print(f"  Gap = {gap_pct:.4f}%")

# Component breakdown
print(f"\n  Component breakdown (n · W · log2(1 + S/σ²_eff_shifted)):")
for dist, label in [(d_opt, "opt (13,1,1,1)"), (d_bestNE, "bestNE (1,1,1,1,s=12)")]:
    parts = []
    for n in dist:
        if n == 0:
            parts.append(f"R(0)=0")
        else:
            sinr = S35 / (s2_35 + max(0, n-1)*I35)
            rn   = n * W_HZ * math.log2(1 + sinr)
            parts.append(f"R({n})={rn/1e6:.4f}Mbps (SINR={10*math.log10(sinr):.2f}dB)")
    print(f"  {label}: {', '.join(parts)}")

# Re-check using Python's math (64-bit) vs numpy — should be identical
R_opt_py    = sum(n * W_HZ * math.log2(1 + S35/(s2_35+max(0,n-1)*I35)) for n in d_opt)
R_ne_py     = sum(n * W_HZ * math.log2(1 + S35/(s2_35+max(0,n-1)*I35)) for n in d_bestNE)
gap_py      = (R_opt_py - R_ne_py) / R_opt_py * 100.0
print(f"\n  Recomputed with math.log2 (Python float64): gap = {gap_py:.6f}%")
print(f"  Difference in R_opt between numpy/math: {abs(R_opt - R_opt_py):.3e} bps")
