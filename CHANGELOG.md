# Changelog

[Back to README](README.md)

## Table V reproduction notes

These two exceptions to the family presets are **not applied automatically**. Save the JSON override to a file
and pass it with `--config <file.json>` when using the commands in the README.

| Setting | Required override |
|---------|-------------------|
| VDN · SIG ML (4 / 8 / 16 agents) | `{"agent_lr": 5e-6}`; replace `--state_order agent_first` with `--state_order global` |
| QMIX · SIG SL (16 agents only) | `{"mixer_lr": 1e-5}` |

For NFIG and the 4-agent SIG SL result without fast fading, set `fast_fading_enabled = False` in
`Configuration/env_params.py`. Fast fading is enabled by default.

SIG ML and POSIG exclude UNSEEN-100 from the default training pool. Keep this split when reproducing the paper.
Custom files supplied through `--train_data` are not filtered automatically; check that they exclude the held-out set.

<details>
<summary>rep9 topology IDs and code versions</summary>

`<snapshot>` is a `snapshot_id` of `rep9_k<n>.csv` (in file order: ETSI #1 far, mid, near, #3 …, #6 …):

| Agents | rep9 snapshot IDs |
|--------|-------------------|
| 4 | 1399, 33, 1237, 15534, 17638, 19092, 29929, 29898, 29132 |
| 8 | 9164, 156, 8157, 28615, 39558, 33362, 57754, 59386, 57072 |
| 16 | 5470, 6271, 3712, 38835, 25493, 30212, 58914, 56736, 57627 |

The recorded Table V runs used code commit `71563c4`, except VDN and QMIX on 16-agent POSIG, which used
`e5fc5f0` with the same POSIG code path. All runs used seeds 9, 10, 12, 87 and 91.

</details>

## 2026-10

- Added `rep9`, `unseen100` and `seen100` evaluation sets for 4, 8 and 16 agents. UNSEEN-100 is excluded
  from default SIG ML / POSIG training.
- Added final checkpoint saving for all eight algorithms.
- Updated the task presets for the paper's Table V settings, including training budgets, the POSIG IDQN
  learning rate and SIG SL PPO update stabilization. See the [reproduction notes](#table-v-reproduction-notes)
  for the two exceptions.
- Vectorized state construction and fast-fading generation.

## 2026-09-29

- Fixed PopArt critic scaling in IPPO and MAPPO. **IPPO/MAPPO results produced with `popart=True` before
  this update should be re-run.**
- Added per-task presets and sparse `--config` overrides.
- Enabled fast fading by default. See the [reproduction notes](#table-v-reproduction-notes) for runs without fading.
- Added the random seed to result filenames.
