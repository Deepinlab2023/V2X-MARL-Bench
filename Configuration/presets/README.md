# Task presets

`<family>_<task>.json` is loaded automatically for every run of that task (see `Configuration/param_loader.py`);
`--config <json>` overrides it. Families: `idql` (IDQN, Hys-IDQN), `qmix` (VDN, QMIX), `a2c` (IA2C, MAA2C),
`ppo` (IPPO, MAPPO).

The presets reproduce the settings of the paper's Table V, with the same preset for 4, 8 and 16 agents. Two
settings differ from their family preset; for these, pass a sparse JSON with `--config`:

| Setting | Table V value | Preset value | `--config` file content |
|---------|---------------|--------------|-------------------------|
| VDN, SIG ML (4 / 8 / 16 agents) | agent lr 5e-6 | 1e-5 (QMIX) | `{"agent_lr": 5e-6}` |
| QMIX, SIG SL, 16 agents | mixer lr 1e-5 | 1e-6 (4 / 8 agents) | `{"mixer_lr": 1e-5}` |

## Other settings of the Table V runs

| Task | `--state_order` | Topologies |
|------|-----------------|------------|
| NFIG | `global` (default) | `--train_data Environment/SUMOData/eval_sets/rep9_k<n>.csv --loc <snapshot_id>`, one run per topology |
| SIG SL | `agent_first` | as NFIG |
| SIG ML | `agent_first` (VDN: `global`) | default training data, UNSEEN-100 held out |
| POSIG | `global` (default) | default training data, UNSEEN-100 held out |

NFIG and SIG SL with 4 agents were also run with fast fading off (`fast_fading_enabled` in
`Configuration/env_params.py`). All Table V runs used 5 seeds (9, 10, 12, 87, 91) and code commit `71563c4`, except VDN
and QMIX on 16-agent POSIG (`e5fc5f0`, which has the same POSIG code path).
