# V2X-MARL-Bench

A benchmarking framework for evaluating multi-agent reinforcement learning (MARL) algorithms in Cellular Vehicle-to-Everything (C-V2X) radio resource allocation. The framework models the problem as a set of interference games with increasing complexity, and provides standardized environments and evaluation protocols for fair and reproducible comparison.

Built with simplicity in mind — no heavy dependencies, no complex setup. If you have Python and PyTorch, you are ready to go.

---

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Repository Structure](#repository-structure)
- [Task Types](#task-types)
- [Algorithms](#algorithms)
- [Citation](#citation)
- [License](#license)

---

## Overview

C-V2X resource allocation requires vehicles to jointly select subchannels and transmission power levels to maximize network throughput while managing interference. V2X-MARL-Bench formulates this as a multi-agent interference game and benchmarks 8 MARL algorithms across 4 task types of increasing complexity and observability.

Key features:
- 8 MARL algorithms spanning value-based and actor-critic families
- 4 task formulations: NFIG, SIG SL, SIG ML, and POSIG
- 3GPP TR 36.885-compliant channel models (path loss + Rayleigh fading)
- SUMO-based vehicle mobility traces
- Scalable evaluation across 4, 8, and 16 agents
- Lightweight implementation using only standard scientific Python libraries

---

## Installation

**Requirements:** Python 3.10+

Clone the repository:
```bash
git clone https://github.com/Deepinlab2023/V2X-MARL-Bench.git
cd V2X-MARL-Bench
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Dependencies (`requirements.txt`):
```
torch==2.6.0
numpy==2.2.2
pandas==2.2.3
matplotlib==3.10.0
scipy==1.15.1
scikit-learn==1.5.2
```

---

## Quick Start

Run an experiment using the following command pattern:
```bash
python main.py --env <ENV> --algo <ALGO> [--loc <LOC>]
```

**Arguments:**

| Argument | Description |
|----------|-------------|
| `--env`  | Task type: `NFIG`, `SIG`, or `POSIG` |
| `--algo` | Algorithm: `idql`, `hys`, `vdn`, `qmix`, `ia2c`, `maa2c`, `ippo`, `mappo` |
| `--loc`  | Location index (NFIG and SIG SL only; omit for SIG ML and POSIG) |

**Examples:**
```bash
# NFIG
python main.py --env NFIG --loc 0.0 --algo idql

# SIG Single-Location
python main.py --env SIG --loc 0.0 --algo idql

# SIG Multi-Location
python main.py --env SIG --algo idql

# POSIG
python main.py --env POSIG --algo idql
```

For NFIG and SIG SL, `--loc` selects one of 9 predefined vehicle topologies that vary traffic density and relative distance to the base station (BS). BS distance reflects the longitudinal offset between the road segment occupied by vehicles and the BS, which depends on the number of agents and vehicle density. All 9 topologies are supported for 4, 8, and 16 agents. For SIG ML and POSIG, the number of agents can be set to 4, 8, or 16 by changing `self.n_agent` in `Configuration/env_params.py` (default datasets are provided for all three settings). Algorithm hyperparameters and environment settings can be configured in the `Configuration/` directory.

| `--loc` | Density | BS Distance |
|---------|---------|-------------|
| `0.0`   | Low     | Far         |
| `1.0`   | Low     | Mid         |
| `2.0`   | Low     | Close       |
| `3.0`   | Mid     | Far         |
| `4.0`   | Mid     | Mid         |
| `5.0`   | Mid     | Close       |
| `6.0`   | High    | Far         |
| `7.0`   | High    | Mid         |
| `8.0`   | High    | Close       |

<!-- ![Testing locations for L=4](Assets/testing_dataset.png) -->

<p align="center">
  <img src="Assets/testing_dataset.png" alt="Testing locations for L=4">
  <br>
  <em>Nine testing topologies for the 4-agent setting.</em>
</p>




---

## Repository Structure
```
V2X-MARL-Bench/
├── main.py                  # Entry point
├── Configuration/           # Environment and algorithm hyperparameters
├── Environment/             # V2X simulation (channel models, rewards, state generation)
├── Runners/                 # Experiment orchestration per algorithm family
├── Trainers/                # Algorithm-specific training logic
├── Networks/                # Actor, critic, and Q-network architectures
├── Helpers/                 # Utility functions (GAE, returns, baseline)
├── Benchmarkers/            # Evaluation and performance analysis tools
└── requirements.txt
```

---

## Task Types

| Task | Observability | Episode Length | Description |
|------|--------------|----------------|-------------|
| NFIG | Full | 1 step | Instantaneous resource allocation without fading |
| SIG SL | Full | 50 steps | Sequential allocation at a fixed vehicle topology |
| SIG ML | Full | 50 steps | Sequential allocation across diverse vehicle topologies |
| POSIG | Partial | 50 steps | Decentralized execution with local observations only |

Tasks increase in difficulty from NFIG (simplest) to POSIG (most realistic).

---

## Algorithms

| Algorithm | Type         | Config File                        |
|-----------|--------------|------------------------------------|
| IDQN      | Value-based  | `Configuration/idql_params.py`     |
| Hys-IDQN  | Value-based  | `Configuration/idql_params.py`     |
| VDN       | Value-based  | `Configuration/qmix_params.py`     |
| QMIX      | Value-based  | `Configuration/qmix_params.py`     |
| IA2C      | Actor-Critic | `Configuration/a2c_params.py`      |
| MAA2C     | Actor-Critic | `Configuration/a2c_params.py`      |
| IPPO      | Actor-Critic | `Configuration/ppo_params.py`      |
| MAPPO     | Actor-Critic | `Configuration/ppo_params.py`      |

Value-based algorithms currently do not support parameter sharing.

---

## Citation

This framework extends our preliminary work presented at ICC 2025. If you use V2X-MARL-Bench in your research, please cite:
```bibtex
@inproceedings{wang2025v2x,
  author    = {Wang, Siyuan and Maheshwari, Pranav and Lei, Lei and Mei, Jie and Zheng, Kan},
  title     = {Multi-Agent DRL for Resource Allocation in Vehicular Networks: A Comparative Study},
  booktitle = {ICC 2025 - IEEE International Conference on Communications},
  year      = {2025},
  pages     = {1936--1941},
  doi       = {10.1109/ICC52391.2025.11161818}
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.