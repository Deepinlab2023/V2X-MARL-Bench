# Heterogeneous-channel experiments — summary

Generated: 2026-09-03 07:54:42  |  60 runs = 2 algos x 3 densities x 2 power modes x 5 seeds (9, 10, 12, 87, 91)

## Protocol

- Representative SUMO realizations (from the homogeneous study's 200-sample set; rule: closest to the ensemble operating point):

| density | scenario | snapshot | beta [dB] | gamma [dB] | ensemble (beta, gamma) | R*(G) single-power |
|---|---|---|---|---|---|---|
| low | etsi1 | 7092 | 13.303 | 74.545 | (13.293, 74.584) | **151.1664** |
| mid | etsi4 | 47688 | 11.867 | 79.917 | (11.977, 79.925) | **114.3827** |
| high | etsi6 | 58582 | 10.159 | 83.456 | (10.141, 83.410) | **114.1911** |

- Single-power (1P): action space {4 SC, NT} at 23 dBm — identical to the homogeneous study; only channel heterogeneity changes. R*(G) exact via subset DP (2^16 subset table + partition DP). Ratio = exact efficiency R(a^L)/R*(G).
- Three-power (3P): action space 3 powers x 4 SC + NT = 13 (powers [23, 10, 5] dBm, full-benchmark encoding). No global-optimum or PNE-set enumeration. Ratio = R_3P(a^L)/R*_1P = reward relative to the single-power optimum — **not** an efficiency w.r.t. the 3-power optimum; values > 1 are legitimate (power selection enlarges the action space).
- IDQN: idql_params defaults (30000 episodes, epsi 1.0 -> 0.01, lr 1e-3, batch 64). IPPO: ppo_params defaults with parameter sharing (50000 episodes, entropy 0.01, batch 256, epochs 5). Identical to the homogeneous V-E study.
- Exact-PNE test: full unilateral-deviation enumeration (64 for 1P, 192 for 3P); PNE iff eps <= 1e-9. eps(a^L) = max_i max_a' [R(a'_i, a_-i) - R(a^L)].

## Summary table (side-by-side 1P / 3P)

| density | algo | power | n | R/R*₁ₚ mean ± std | min | # exact PNE | max ε/R*₁ₚ (non-PNE) |
|---|---|---|---|---|---|---|---|
| low | idqn | 1P | 5 | 0.9855 ± 0.0086 | 0.9782 | 4/5 | 0.0001 |
| low | idqn | 3P | 5 | 1.0191 ± 0.0090 | 1.0070 | 0/5 | 0.0093 |
| low | ippo | 1P | 5 | 0.9778 ± 0.0152 | 0.9522 | 4/5 | 0.0090 |
| low | ippo | 3P | 5 | 1.0232 ± 0.0102 | 1.0080 | 0/5 | 0.0153 |
| mid | idqn | 1P | 5 | 0.9588 ± 0.0180 | 0.9331 | 2/5 | 0.0239 |
| mid | idqn | 3P | 5 | 0.9715 ± 0.0224 | 0.9375 | 0/5 | 0.0140 |
| mid | ippo | 1P | 5 | 0.9678 ± 0.0066 | 0.9588 | 5/5 | - |
| mid | ippo | 3P | 5 | 1.0058 ± 0.0110 | 0.9936 | 4/5 | 0.0239 |
| high | idqn | 1P | 5 | 0.9389 ± 0.0509 | 0.8909 | 2/5 | 0.0142 |
| high | idqn | 3P | 5 | 0.8378 ± 0.0676 | 0.7621 | 0/5 | 0.0139 |
| high | ippo | 1P | 5 | 0.9584 ± 0.0460 | 0.9062 | 5/5 | - |
| high | ippo | 3P | 5 | 0.9106 ± 0.0082 | 0.9031 | 4/5 | 0.0009 |

For 1P rows the ratio is the exact efficiency R(a^L)/R*(G); for 3P rows it is the reward relative to the single-power optimum (see Protocol).

## Per-run results

| density | algo | power | seed | R(a^L) | R*₁ₚ | ratio | PNE? | eps | eps/R*₁ₚ | best deviation | loads | pw use |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| high | idqn | 1P | 9 | 113.6246 | 114.1911 | 0.9950 | Y | 0.0000 | 0.0000 | - | [1, 1, 1, 1] | - |
| high | idqn | 1P | 10 | 101.7330 | 114.1911 | 0.8909 | N | 0.1345 | 0.0012 | ag9:0->4 | [8, 1, 1, 1] | - |
| high | idqn | 1P | 12 | 103.4054 | 114.1911 | 0.9055 | N | 0.4889 | 0.0043 | ag13:0->4 | [5, 1, 1, 1] | - |
| high | idqn | 1P | 87 | 103.9026 | 114.1911 | 0.9099 | N | 1.6271 | 0.0142 | ag2:4->1 | [1, 4, 1, 1] | - |
| high | idqn | 1P | 91 | 113.4115 | 114.1911 | 0.9932 | Y | 0.0000 | 0.0000 | - | [1, 1, 1, 1] | - |
| high | idqn | 3P | 9 | 101.8732 | 114.1911 | 0.8921 | N | 0.5928 | 0.0052 | ag11:12->10 | [1, 1, 1, 7] | 5/2/3 |
| high | idqn | 3P | 10 | 100.7724 | 114.1911 | 0.8825 | N | 1.5903 | 0.0139 | ag15:2->0 | [11, 1, 1, 1] | 7/2/5 |
| high | idqn | 3P | 12 | 87.0260 | 114.1911 | 0.7621 | N | 1.5048 | 0.0132 | ag12:0->12 | [7, 1, 5, 1] | 9/3/2 |
| high | idqn | 3P | 87 | 87.4239 | 114.1911 | 0.7656 | N | 0.6296 | 0.0055 | ag5:10->11 | [10, 1, 1, 3] | 7/3/5 |
| high | idqn | 3P | 91 | 101.2532 | 114.1911 | 0.8867 | N | 0.4518 | 0.0040 | ag11:4->3 | [1, 10, 1, 1] | 8/1/4 |
| high | ippo | 1P | 9 | 103.4783 | 114.1911 | 0.9062 | Y | 0.0000 | 0.0000 | - | [1, 5, 1, 1] | - |
| high | ippo | 1P | 10 | 113.4115 | 114.1911 | 0.9932 | Y | 0.0000 | 0.0000 | - | [1, 1, 1, 1] | - |
| high | ippo | 1P | 12 | 113.1414 | 114.1911 | 0.9908 | Y | 0.0000 | 0.0000 | - | [1, 1, 1, 1] | - |
| high | ippo | 1P | 87 | 113.2884 | 114.1911 | 0.9921 | Y | 0.0000 | 0.0000 | - | [1, 1, 1, 1] | - |
| high | ippo | 1P | 91 | 103.8943 | 114.1911 | 0.9098 | Y | 0.0000 | 0.0000 | - | [4, 1, 1, 1] | - |
| high | ippo | 3P | 9 | 103.1221 | 114.1911 | 0.9031 | Y | 0.0000 | 0.0000 | - | [1, 5, 1, 1] | 8/0/0 |
| high | ippo | 3P | 10 | 103.4783 | 114.1911 | 0.9062 | Y | 0.0000 | 0.0000 | - | [1, 1, 5, 1] | 8/0/0 |
| high | ippo | 3P | 12 | 105.5297 | 114.1911 | 0.9241 | Y | 0.0000 | 0.0000 | - | [5, 1, 1, 1] | 8/0/0 |
| high | ippo | 3P | 87 | 104.1041 | 114.1911 | 0.9117 | Y | 0.0000 | 0.0000 | - | [1, 5, 1, 1] | 7/1/0 |
| high | ippo | 3P | 91 | 103.7048 | 114.1911 | 0.9082 | N | 0.0983 | 0.0009 | ag13:6->7 | [1, 1, 6, 1] | 9/0/0 |
| low | idqn | 1P | 9 | 147.8699 | 151.1664 | 0.9782 | N | 0.0080 | 0.0001 | ag13:4->3 | [4, 4, 2, 2] | - |
| low | idqn | 1P | 10 | 147.8779 | 151.1664 | 0.9782 | Y | 0.0000 | 0.0000 | - | [2, 3, 4, 4] | - |
| low | idqn | 1P | 12 | 150.5435 | 151.1664 | 0.9959 | Y | 0.0000 | 0.0000 | - | [3, 3, 2, 3] | - |
| low | idqn | 1P | 87 | 150.1990 | 151.1664 | 0.9936 | Y | 0.0000 | 0.0000 | - | [3, 4, 2, 3] | - |
| low | idqn | 1P | 91 | 148.3606 | 151.1664 | 0.9814 | Y | 0.0000 | 0.0000 | - | [3, 2, 3, 3] | - |
| low | idqn | 3P | 9 | 152.2198 | 151.1664 | 1.0070 | N | 0.9237 | 0.0061 | ag4:10->11 | [4, 3, 3, 3] | 6/5/2 |
| low | idqn | 3P | 10 | 153.8700 | 151.1664 | 1.0179 | N | 1.3379 | 0.0089 | ag8:10->9 | [3, 4, 4, 3] | 6/2/6 |
| low | idqn | 3P | 12 | 154.2179 | 151.1664 | 1.0202 | N | 0.7274 | 0.0048 | ag3:7->6 | [4, 3, 3, 3] | 5/4/4 |
| low | idqn | 3P | 87 | 153.9010 | 151.1664 | 1.0181 | N | 0.6213 | 0.0041 | ag12:8->7 | [3, 3, 5, 3] | 6/4/4 |
| low | idqn | 3P | 91 | 156.0300 | 151.1664 | 1.0322 | N | 1.4016 | 0.0093 | ag12:4->7 | [3, 3, 3, 3] | 8/4/0 |
| low | ippo | 1P | 9 | 143.9426 | 151.1664 | 0.9522 | N | 1.3611 | 0.0090 | ag7:0->2 | [4, 3, 2, 3] | - |
| low | ippo | 1P | 10 | 147.5562 | 151.1664 | 0.9761 | Y | 0.0000 | 0.0000 | - | [3, 4, 2, 3] | - |
| low | ippo | 1P | 12 | 149.1124 | 151.1664 | 0.9864 | Y | 0.0000 | 0.0000 | - | [3, 2, 3, 3] | - |
| low | ippo | 1P | 87 | 148.7882 | 151.1664 | 0.9843 | Y | 0.0000 | 0.0000 | - | [4, 2, 3, 2] | - |
| low | ippo | 1P | 91 | 149.6788 | 151.1664 | 0.9902 | Y | 0.0000 | 0.0000 | - | [2, 3, 3, 3] | - |
| low | ippo | 3P | 9 | 153.8217 | 151.1664 | 1.0176 | N | 1.6900 | 0.0112 | ag4:3->4 | [3, 3, 4, 3] | 9/3/1 |
| low | ippo | 3P | 10 | 155.8808 | 151.1664 | 1.0312 | N | 0.0765 | 0.0005 | ag13:12->11 | [4, 3, 3, 2] | 7/4/1 |
| low | ippo | 3P | 12 | 155.4458 | 151.1664 | 1.0283 | N | 0.3791 | 0.0025 | ag10:12->7 | [4, 4, 2, 2] | 8/3/1 |
| low | ippo | 3P | 87 | 155.8605 | 151.1664 | 1.0311 | N | 0.9996 | 0.0066 | ag10:12->4 | [3, 2, 2, 4] | 8/3/0 |
| low | ippo | 3P | 91 | 152.3758 | 151.1664 | 1.0080 | N | 2.3141 | 0.0153 | ag12:0->1 | [3, 3, 2, 3] | 8/3/0 |
| mid | idqn | 1P | 9 | 111.8486 | 114.3827 | 0.9778 | N | 1.1002 | 0.0096 | ag5:4->3 | [1, 1, 1, 2] | - |
| mid | idqn | 1P | 10 | 109.6069 | 114.3827 | 0.9582 | N | 2.7334 | 0.0239 | ag14:3->4 | [1, 3, 1, 2] | - |
| mid | idqn | 1P | 12 | 111.3515 | 114.3827 | 0.9735 | Y | 0.0000 | 0.0000 | - | [1, 3, 2, 1] | - |
| mid | idqn | 1P | 87 | 108.7937 | 114.3827 | 0.9511 | Y | 0.0000 | 0.0000 | - | [1, 5, 1, 1] | - |
| mid | idqn | 1P | 91 | 106.7276 | 114.3827 | 0.9331 | N | 0.3230 | 0.0028 | ag7:0->4 | [4, 1, 1, 1] | - |
| mid | idqn | 3P | 9 | 113.0590 | 114.3827 | 0.9884 | N | 1.1430 | 0.0100 | ag6:2->12 | [4, 1, 1, 5] | 5/2/4 |
| mid | idqn | 3P | 10 | 107.2304 | 114.3827 | 0.9375 | N | 0.4992 | 0.0044 | ag2:12->2 | [4, 1, 1, 5] | 5/4/2 |
| mid | idqn | 3P | 12 | 113.5350 | 114.3827 | 0.9926 | N | 0.6437 | 0.0056 | ag5:8->10 | [1, 1, 4, 3] | 5/2/2 |
| mid | idqn | 3P | 87 | 111.7480 | 114.3827 | 0.9770 | N | 0.7292 | 0.0064 | ag11:10->12 | [4, 1, 1, 6] | 6/4/2 |
| mid | idqn | 3P | 91 | 110.0637 | 114.3827 | 0.9622 | N | 1.5981 | 0.0140 | ag3:5->2 | [2, 5, 1, 1] | 4/2/3 |
| mid | ippo | 1P | 9 | 111.3057 | 114.3827 | 0.9731 | Y | 0.0000 | 0.0000 | - | [1, 1, 3, 1] | - |
| mid | ippo | 1P | 10 | 111.4562 | 114.3827 | 0.9744 | Y | 0.0000 | 0.0000 | - | [1, 1, 1, 3] | - |
| mid | ippo | 1P | 12 | 109.6706 | 114.3827 | 0.9588 | Y | 0.0000 | 0.0000 | - | [1, 1, 1, 5] | - |
| mid | ippo | 1P | 87 | 110.8849 | 114.3827 | 0.9694 | Y | 0.0000 | 0.0000 | - | [3, 1, 2, 1] | - |
| mid | ippo | 1P | 91 | 110.1896 | 114.3827 | 0.9633 | Y | 0.0000 | 0.0000 | - | [1, 1, 3, 1] | - |
| mid | ippo | 3P | 9 | 114.4141 | 114.3827 | 1.0003 | Y | 0.0000 | 0.0000 | - | [1, 1, 6, 1] | 5/1/3 |
| mid | ippo | 3P | 10 | 114.4023 | 114.3827 | 1.0002 | N | 2.7334 | 0.0239 | ag14:3->12 | [5, 2, 1, 1] | 6/1/2 |
| mid | ippo | 3P | 12 | 116.3951 | 114.3827 | 1.0176 | Y | 0.0000 | 0.0000 | - | [5, 1, 1, 1] | 5/1/2 |
| mid | ippo | 3P | 87 | 116.3712 | 114.3827 | 1.0174 | Y | 0.0000 | 0.0000 | - | [1, 7, 1, 1] | 5/1/4 |
| mid | ippo | 3P | 91 | 113.6529 | 114.3827 | 0.9936 | Y | 0.0000 | 0.0000 | - | [1, 1, 7, 1] | 5/1/4 |

## Framing (paper text)

> **Single power:** heterogeneous channels do not destroy the high-efficiency phenomenon predicted by the homogeneous reference analysis; both learners attain about 96% of the exact optimum on average, with IPPO reaching an exact PNE in 14 of 15 runs.

> **Power selection:** restoring three discrete power levels preserves strong performance overall and can even exceed the single-power optimum, although IDQN exhibits a noticeable degradation at the high-density realization.

> **Equilibrium:** IPPO retains stronger exact-equilibrium selection, whereas IDQN predominantly produces near-PNE solutions with small unilateral regret.

NOTE: do NOT claim the 3-power experiments verify the homogeneous theorem — the theory does not cover the power-selection game. 1P = exact efficiency; 3P = performance + equilibrium stability only (no optimality statement). Say "near-equilibrium behavior remains prevalent" rather than "equilibrium-seeking behavior is preserved". eps uses the positive-part definition (eps >= 0, exact PNE iff eps = 0).
