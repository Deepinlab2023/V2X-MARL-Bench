# Task presets

`<family>_<task>.json` is loaded automatically for every run of that task (see `Configuration/param_loader.py`);
`--config` overrides it. Families: `idql` (IDQN, Hys-IDQN), `qmix` (VDN, QMIX), `a2c` (IA2C, MAA2C),
`ppo` (IPPO, MAPPO). Values follow the paper's hyperparameter tables (VIII–XI). The same preset is used for 4, 8
and 16 agents. Opt-in alternatives that are not part of the benchmark settings live in `Configuration/experimental/`.

## Validation status

A preset is **validated** for a setting once all its algorithms were trained to the full budget with the current
environment code, over several seeds, without a late-training collapse.

| Task | Presets | Validated for | Status |
|------|---------|---------------|--------|
| SIG ML | `*_SIG_ML.json` | 16 agents: all 8 algorithms × 5 seeds, commit `e5fc5f0` (2026-10). No late-training collapse. IDQN / Hys-IDQN / VDN / QMIX are still improving at 30000 episodes. | Validated (16 agents) |
| POSIG | `*_POSIG.json` | 16 agents: all 8 algorithms × 5 seeds, commit `e5fc5f0` (2026-10). No late-training collapse. | Validated (16 agents) |
| SIG SL | `*_SIG_SL.json` | IDQN, 16 agents, 100 topologies × 5 seeds, before the fast-fading timing fix (`76820b3`). IPPO was run with `experimental/ppo_SIG_SL_stable.json` added, not with the preset alone. | Partly validated |
| NFIG | `*_NFIG.json` | Not yet run to the full budget with the current code. | Not validated |

4 and 8 agents: no task has been validated with the current code yet.
