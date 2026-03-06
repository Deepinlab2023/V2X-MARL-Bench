# V2X-MARL-Bench

A benchmarking framework for **multi-agent reinforcement learning (MARL)** in **cellular V2X resource allocation**. It models the problem as a set of interference games with increasing complexity and provides standardized environments and evaluation protocols for fair, reproducible comparison.

## Overview

V2X-MARL-Bench provides:

- **Three interference game formulations** of increasing complexity:
  - **NFIG** (Normal-Form Interference Game) -- single-step, static channel
  - **SIG** (Stochastic Interference Game) -- multi-step with time-varying channels (single-location SL or multi-location ML)
  - **POSIG** (Partially Observable SIG) -- agents observe only local information
- **Eight MARL algorithms** ready to benchmark out of the box
- A **realistic C-V2X channel model** following 3GPP TR 36.885 and ETSI standards
- **SUMO-based vehicle mobility data** for 4, 8, and 16 agent configurations

## Supported Algorithms

| Algorithm | Type | Description |
|-----------|------|-------------|
| IA2C | Actor-Critic | Independent Advantage Actor-Critic |
| MAA2C | Actor-Critic | Multi-Agent A2C with parameter sharing |
| IPPO | Policy Gradient | Independent Proximal Policy Optimization |
| MAPPO | Policy Gradient | Multi-Agent PPO with centralized critic |
| IDQL | Value-Based | Independent Deep Q-Learning |
| Hysteretic DQL | Value-Based | IDQL with hysteretic learning rates |
| VDN | Value-Based | Value Decomposition Network |
| QMIX | Value-Based | QMIX with monotonic mixing network |

## Repository Structure

```
V2X-MARL-Bench/
├── main.py                  # Entry point
├── Configuration/           # Hyperparameters for environment and algorithms
│   ├── env_params.py        #   V2X environment parameters (3GPP constants, topology, data paths)
│   ├── a2c_params.py        #   IA2C / MAA2C hyperparameters
│   ├── ppo_params.py        #   IPPO / MAPPO hyperparameters
│   ├── idql_params.py       #   IDQL / Hysteretic DQL hyperparameters
│   └── qmix_params.py       #   QMIX / VDN hyperparameters
├── Environment/
│   ├── Environment.py       #   V2X simulation environment (channel model, rewards, state)
│   ├── environment_utility.py # Data loading, sampling, and analysis utilities
│   └── SUMOData/            #   SUMO vehicle mobility CSV files
├── Networks/
│   ├── Agents/              #   Actor / Q-network architectures
│   └── Critics/             #   Critic network architectures
├── Trainers/                # Training loops for each algorithm
├── Runners/                 # Experiment runners that wire environment + trainer + evaluation
├── Benchmarkers/            # Greedy evaluation (test) logic for each algorithm
└── Helpers/                 # Shared utilities (GAE, PopArt, batch processing, plotting)
```

## Installation

**Prerequisites:** Python 3.8+

Install the required packages:

```bash
pip install torch numpy pandas scipy matplotlib
```

Clone the repository:

```bash
git clone https://github.com/Deepinlab2023/V2X-MARL-Bench.git
cd V2X-MARL-Bench
```

## Usage

All experiments are launched through `main.py`. You specify the **environment** (`--env`), optionally a **time index** (`--loc`), and the **algorithm** (`--algo`).

### NFIG (single-step, fixed location)

```bash
python main.py --env NFIG --loc 25.0 --algo maa2c
```

### SIG -- Single Location (SL)

```bash
python main.py --env SIG --loc 30.0 --algo mappo
```

### SIG -- Multi Location (ML)

```bash
python main.py --env SIG --algo ippo
```

### POSIG (partially observable, multi-location)

```bash
python main.py --env POSIG --algo qmix
```

### Available `--algo` values

`ia2c`, `maa2c`, `ippo`, `mappo`, `idql`, `hys`, `vdn`, `qmix`

## Configuration

### Environment

Key parameters are defined in `Configuration/env_params.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_agent` | 4 | Number of V2V agents (also supports 8, 16) |
| `n_sc` | 4 | Number of sub-channels |
| `n_step_per_episode` | 1 (NFIG) / 50 (SIG, POSIG) | Steps per episode |
| `fast_fading_enabled` | True | Toggle Rayleigh fast fading |
| `v2v_power_levels_dbm` | [23, 10, 5] | Transmit power options (dBm) |

### Algorithms

Each algorithm has its own config file in `Configuration/`. Common settings include:

- `training_episodes` -- total training episodes
- `batch_size` -- episodes per training batch
- `gamma` -- discount factor
- `actor_hidden_dim` / `critic_hidden_dim` / `hidden_dim` -- network sizes
- `test_interval` -- how often to evaluate

Algorithm-specific toggles:

- **A2C:** `no_sharing` (independent vs. shared parameters), `action_masking`
- **PPO:** `popart` (value normalization), `eps_clip`, `feature_pruning` (MAPPO + POSIG)
- **IDQL:** `hysteretic_high_lr` / `hysteretic_low_lr`
- **QMIX:** `two_hyper_layers`, separate `agent_lr` / `mixer_lr`

## Results

During training, test rewards are periodically evaluated and logged to CSV files. At the end of a run, the framework prints:

- **Max mean test reward** with 95% confidence interval
- **Mean reward over time** for each test checkpoint

## License

This project is licensed under the [MIT License](LICENSE).
