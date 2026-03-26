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
python main.py --env <ENV> --algo <ALGO> [--loc <LOC>]
```

- `--env`: `NFIG` (single-step, no fading), `SIG` (50-step, full observability), or `POSIG` (50-step, partial observability)
- `--algo`: `idql`, `hys`, `vdn`, `qmix`, `ia2c`, `maa2c`, `ippo`, `mappo`
- `--loc`: Location index `0.0`–`8.0`, only used for NFIG and SIG single-location runs

```bash
python main.py --env NFIG --algo idql --loc 0.0        # Single-step, value-based
python main.py --env SIG --algo mappo                   # Multi-location (omit --loc)
python main.py --env SIG --algo maa2c --loc 2.5        # Single-location
python main.py --env POSIG --algo ippo                  # Partial observability
```

Results are saved as CSV files in the working directory (e.g., `IA2C_trial_0_NFIG44_0.0.csv`).

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
- **`Configuration/env_params.py`**: Physical constants (antenna gains, power levels, subchannels), task-specific reward weights, dataset paths, and agent count (`n_agent`: 4, 8, or 16)
- **`Runners/`**: Three runners — `policy_gradient_runner.py` (IA2C/MAA2C/IPPO/MAPPO), `idql_runner.py` (IDQL/Hys-IDQL), `qmix_runner.py` (VDN/QMIX)
- **`Trainers/`**: One trainer per algorithm implementing the core update logic
- **`Benchmarkers/`**: Evaluation modules called at `test_interval` during training

### Configuration

Each algorithm family has its own params file in `Configuration/`:
- `idql_params.py`: `training_episodes=30000`, `lr=1e-5`, `hidden_dim=128`, hysteretic LR settings
- `a2c_params.py`: `training_episodes=100000`, `action_masking=True`, `adv_normalization=True`
- `ppo_params.py`: `training_episodes=100000`, PPO clip `eps=0.2`, GAE `lam=0.95`, `popart=True`
- `qmix_params.py`: `training_episodes=30000`, separate `agent_lr` and `mixer_lr`

### Important Design Decisions

- **Action masking**: Enforces "No Transmission" when queue is empty — controlled by `action_masking` in `a2c_params.py`
- **Fast fading toggle**: `fast_fading_enabled` in `env_params.py` switches between realistic and idealized channel models
- **POSIG constraint**: Requires `no_sharing=False` (parameter sharing) due to partial observability
- **Agent count**: Controlled by `n_agent` in `env_params.py` (4, 8, or 16); must match the loaded CSV dataset
- **Data files**: `Environment/SUMOData/` contains SUMO-generated vehicle position CSVs; multi-location files are large (up to ~106 MB)
