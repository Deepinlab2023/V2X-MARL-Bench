# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

V2X-MARL-Bench is a benchmarking framework for evaluating multi-agent reinforcement learning (MARL) algorithms in Cellular Vehicle-to-Everything (C-V2X) radio resource allocation. It supports 8 MARL algorithms across 4 task types of increasing complexity, published at IEEE ICC 2025.

## Setup

```bash
pip install -r requirements.txt  # torch==2.6.0, numpy, pandas, matplotlib, scipy, scikit-learn
```

Requires Python 3.10+.

## Running Experiments

```bash
python main.py --env <ENV> --algo <ALGO> [--loc <LOC>] [--seed <SEED>]
```

- `--env`: `NFIG` (single-step, no fading), `SIG` (50-step, full observability), or `POSIG` (50-step, partial observability)
- `--algo`: `idql`, `hys`, `vdn`, `qmix`, `ia2c`, `maa2c`, `ippo`, `mappo`
- `--loc`: Location index `0.0`–`8.0`, only used for NFIG and SIG single-location runs
- `--seed`: Optional integer seed for reproducible experiments (seeds NumPy, Python random, and PyTorch)

```bash
python main.py --env NFIG --algo idql --loc 0.0        # Single-step, value-based
python main.py --env SIG --algo mappo                   # Multi-location (omit --loc)
python main.py --env SIG --algo maa2c --loc 2.5        # Single-location
python main.py --env POSIG --algo ippo                  # Partial observability
python main.py --env SIG --algo mappo --seed 42         # Reproducible run
```

Results are saved as CSV files in `Results/<algo_name>/` (auto-created). Filename format:
`{algo}_{task}_{n_agent}ag_{n_sc}sc_{ff_tag}[_{features}]_trial{n}_{timestamp}.csv`
e.g. `IA2C_NFIG_loc2.5_4ag_4sc_NFF_MASK_NORM_trial0_20260326_153416.csv`

The naming logic lives in `build_csv_name()` in `Environment/environment_utility.py`.

There is no automated test suite or linting configuration.

## Architecture

### Task Types (increasing difficulty)

| Task | Observability | Episode Length |
|------|--------------|----------------|
| NFIG | Full | 1 step (no fading) |
| SIG Single-Location | Full | 50 steps, fixed topology |
| SIG Multi-Location | Full | 50 steps, diverse topologies |
| POSIG | Partial | 50 steps, decentralized execution |

### Algorithm Families

- **Value-based**: `idql` (Independent DQL), `hys` (Hysteretic Q-learning), `vdn` (VDN), `qmix` (QMIX)
- **Actor-Critic**: `ia2c` (Independent A2C), `maa2c` (Multi-agent A2C), `ippo` (Independent PPO), `mappo` (MAPPO)

### Component Flow

```
main.py
  ├─> V2XParams (Configuration/) — loads physical constants and data paths
  ├─> Environ (Environment/environment.py) — channel models, reward, state generation
  └─> Runner (Runners/) — coordinates training
       └─> Trainer (Trainers/) — algorithm-specific training
            ├─> Networks/ — actors, critics, Q-networks
            ├─> Helpers/ — GAE, replay buffers, logging
            └─> Benchmarkers/ — periodic evaluation
```

### Key Files

- **`main.py`**: Parses args, instantiates environment and runner, dispatches to the correct algorithm family runner
- **`Environment/environment.py`**: Core `Environ` class (~34 KB) — implements 3GPP TR 36.885 channel models (pathloss, shadow fading, fast fading), reward computation, and state generation
- **`Environment/environment_utility.py`**: Shared utilities — `build_csv_name()` (single source of truth for CSV naming), vehicle position samplers, action enumeration helpers, and analysis/plotting utilities
- **`Configuration/env_params.py`**: Physical constants (antenna gains, power levels, subchannels), task-specific reward weights, dataset paths, and agent count (`n_agent`: 4, 8, or 16)
- **`Runners/`**: Three runners — `policy_gradient_runner.py` (IA2C/MAA2C/IPPO/MAPPO), `idql_runner.py` (IDQL/Hys-IDQL), `qmix_runner.py` (VDN/QMIX)
- **`Trainers/`**: One trainer per algorithm implementing the core update logic; all trainers now write CSV results via `build_csv_name()`
- **`Networks/Agents/`**: Actor networks (`a2c_actor.py`, `ppo_actor.py`, `idql_agent.py`, `qmix_agent.py`)
- **`Networks/Critics/`**: Critic networks (`a2c_critic.py`, `ppo_critic.py`)
- **`Benchmarkers/`**: Evaluation modules called at `test_interval` during training
- **`Helpers/`**: `a2c_helper.py`, `ppo_helper.py`, `qmix_helper.py` (GAE, replay buffers, logging); `stat_helper.py` (statistics); `plotting_helper.py`

### Configuration

Each algorithm family has its own params file in `Configuration/`:
- `idql_params.py`: `training_episodes=30000`, `lr=1e-5`, `hidden_dim=128`, hysteretic LR settings
- `a2c_params.py`: `training_episodes=100000`, `action_masking=True`, `adv_normalization=True`
- `ppo_params.py`: `training_episodes=100000`, PPO clip `eps=0.2`, GAE `lam=0.95`, `popart=True`
- `qmix_params.py`: `training_episodes=30000`, separate `agent_lr` and `mixer_lr`

### Compute Considerations

All trainers auto-detect GPU via `th.device("cuda" if th.cuda.is_available() else "cpu")` and move networks/tensors accordingly — so a GPU node will be used if available. However, **GPU provides little benefit in practice**: networks are small (128-dim MLP), batch sizes are small (8–256), and the dominant cost is the environment simulation (3GPP channel models implemented in Python/NumPy for-loops) which runs on CPU regardless. On Compute Canada, CPU nodes are more cost-effective; the recommended strategy is to run multiple independent experiments (different seeds or algorithms) in parallel across CPU cores rather than requesting GPU nodes.

### Important Design Decisions

- **Action masking**: Enforces "No Transmission" when queue is empty — controlled by `action_masking` in `a2c_params.py` and `ppo_params.py`
- **Fast fading toggle**: `fast_fading_enabled` in `env_params.py` switches between realistic and idealized channel models; the tag `FF`/`NFF` is embedded in CSV filenames
- **POSIG constraint**: Requires `no_sharing=False` (parameter sharing) due to partial observability; MAPPO uses `feature_pruning=True` in `ppo_params.py` to handle partial observations
- **Agent count**: Controlled by `n_agent` in `env_params.py` (4, 8, or 16); must match the loaded CSV dataset
- **Data files**: `Environment/SUMOData/` contains SUMO-generated vehicle position CSVs; multi-location files are large (up to ~106 MB)
- **CSV results**: All algorithms route output to `Results/<algo_name>/` (auto-created at runtime); naming unified via `build_csv_name()` in `environment_utility.py`
