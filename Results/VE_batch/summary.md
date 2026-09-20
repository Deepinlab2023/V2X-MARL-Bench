# V-E Batch Summary

Generated: 2026-08-29 03:31:48
Seeds: [9, 10, 12, 91, 87]  |  Points: low  |  Algorithms: ippo  |  Total runs: 5

## 1. Operating points (SUMO full-precision means)

| point | beta [dB] | gamma [dB] | lam_eff | OPT | worst_NE | PoA |
|---|---|---|---|---|---|---|
| low | 13.2932 | 74.5837 | 30.8 | 99.1047 | 48.3305 | 2.0506 |

## 2. Per-run results

Last column = empirical PoA = OPT / Phi^L  (directly comparable to theoretical PoA above.)

| point | algo | seed | Phi^L | %OPT | readout | PNE? | label | empirical_PoA |
|---|---|---|---|---|---|---|---|---|
| low | ippo | 9 | 93.497 | 94.3% | (13, 1, 1, 1) | Y | v1 | **1.0600** |
| low | ippo | 10 | 93.497 | 94.3% | (13, 1, 1, 1) | Y | v1 | **1.0600** |
| low | ippo | 12 | 93.497 | 94.3% | (13, 1, 1, 1) | Y | v1 | **1.0600** |
| low | ippo | 87 | 93.497 | 94.3% | (13, 1, 1, 1) | Y | v1 | **1.0600** |
| low | ippo | 91 | 93.497 | 94.3% | (13, 1, 1, 1) | Y | v1 | **1.0600** |

## 3. Conclusion 1: NE convergence (IPPO vs IDQN)

Landing histogram: count of seeds on each canonical PNE / non-PNE.

| point | PoA | algo | v0 | v1 | v2 | v3 | v4 | non-PNE | n_PNE | n_seeds |
|---|---|---|---|---|---|---|---|---|---|---|
| low | 2.05 | idqn | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| low | 2.05 | ippo | 0 | 5 | 0 | 0 | 0 | 0 | 5 | 5 |
| mid | 0.00 | idqn | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| mid | 0.00 | ippo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| high | 0.00 | idqn | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| high | 0.00 | ippo | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

**IPPO: 5/5 seeds converge to a PNE** (always reaches a pure Nash equilibrium).
**IDQN: 0/0 seeds converge to a PNE** (0/0 hit a PNE; 0/0 land on non-PNE readouts).

## 4. Conclusion 2: PoA vs convergence instability

Higher PoA => wider spread of NE rewards => RL convergence becomes more variable across seeds.

| point | PoA | algo | mean %OPT | std %OPT | min %OPT | max %OPT | range | distinct NE | n |
|---|---|---|---|---|---|---|---|---|---|
| low | 2.05 | idqn | - | - | - | - | - | 0 | 0 |
| low | 2.05 | ippo | 94.3% | 0.0% | 94.3% | 94.3% | 0.0% | 1 | 5 |
| mid | 0.00 | idqn | - | - | - | - | - | 0 | 0 |
| mid | 0.00 | ippo | - | - | - | - | - | 0 | 0 |
| high | 0.00 | idqn | - | - | - | - | - | 0 | 0 |
| high | 0.00 | ippo | - | - | - | - | - | 0 | 0 |

### 4a. NE reward spread at each operating point

| point | PoA | OPT (best NE) | worst NE | spread (OPT - worst) | spread / OPT |
|---|---|---|---|---|---|
| low | 2.0506 | 99.10 | 48.33 | 50.77 | 51.2% |

## 5. Conclusion 3: Actual RL performance vs theoretical PoA

empirical_PoA = OPT / Phi^L  (achieved by RL).  theoretical_PoA = OPT / worst_NE  (worst-case NE).

If empirical = theoretical: RL landed on the worst NE.  If empirical = 1: RL found the optimal NE.


| point | algo | seed | Phi^L | %OPT | theoretical_PoA | empirical_PoA | gap (theo - emp) | label |
|---|---|---|---|---|---|---|---|---|
| low | ippo | 9 | 93.497 | 94.3% | 2.0506 | **1.0600** | 0.9906 | v1 |
| low | ippo | 10 | 93.497 | 94.3% | 2.0506 | **1.0600** | 0.9906 | v1 |
| low | ippo | 12 | 93.497 | 94.3% | 2.0506 | **1.0600** | 0.9906 | v1 |
| low | ippo | 87 | 93.497 | 94.3% | 2.0506 | **1.0600** | 0.9906 | v1 |
| low | ippo | 91 | 93.497 | 94.3% | 2.0506 | **1.0600** | 0.9906 | v1 |

### 5a. Aggregate: empirical vs theoretical PoA

| point | PoA (theo) | algo | mean emp_PoA | min emp_PoA | max emp_PoA | n |
|---|---|---|---|---|---|---|
| low | 2.0506 | idqn | - | - | - | 0 |
| low | 2.0506 | ippo | 1.0600 | 1.0600 | 1.0600 | 5 |
| mid | 0.0000 | idqn | - | - | - | 0 |
| mid | 0.0000 | ippo | - | - | - | 0 |
| high | 0.0000 | idqn | - | - | - | 0 |
| high | 0.0000 | ippo | - | - | - | 0 |

## Notes

- IDQN Phi^L = greedy readout reward (single deterministic eval).
- IPPO Phi^L = 9-trial stochastic mean (sampled from learned policy).
- Canonical PNE labels: v0=(1,1,1,1), v1=(13,1,1,1), v2=(7,7,1,1), v3=(5,5,5,1), v4=(4,4,4,4).
- Non-PNE: greedy readout not in the equilibrium set (typically (n,1,1,1) with n < 13).
- Full per-run stdout in `logs/`.