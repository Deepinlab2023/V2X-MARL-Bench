# V2X-MARL-Bench

<p align="center">
  <a href="https://arxiv.org/abs/2603.06607"><img src="https://img.shields.io/badge/arXiv-2603.06607-b31b1b" alt="arXiv"></a>
  <a href="https://doi.org/10.1109/ICC52391.2025.11161818"><img src="https://img.shields.io/badge/ICC-2025-blue" alt="ICC 2025"></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/PyTorch-2.6-ee4c2c" alt="PyTorch 2.6">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="MIT License"></a>
</p>

**A benchmark for multi-agent reinforcement learning (MARL) in C-V2X radio resource allocation.**

V2X-MARL-Bench casts sidelink resource allocation as a family of interference games. Vehicles jointly choose a subchannel and a transmit power level, and are rewarded for delivering their messages while limiting interference to other links. The benchmark provides four tasks of increasing difficulty, eight MARL baselines, and a fixed evaluation protocol, so that algorithms can be compared on the same footing.

## News

- **2026-09-29** — Code update since the previous release:
    - **Fix:** PopArt critic scaling in IPPO and MAPPO. IPPO/MAPPO runs made with `popart=True` before this update should be re-run.
    - **New:** optional PPO update stabilization (KL early stopping, KL-adaptive learning rate, Adam epsilon, gradient clipping), off by default; see `Configuration/experimental/`.
    - **New:** per-task parameter presets, loaded automatically, and sparse `--config` overrides.
    - **New:** model checkpoints; the seed is now part of result and checkpoint filenames.
    - **Changed:** fast fading is enabled by default. Set `fast_fading_enabled = False` in `Configuration/env_params.py` for the previous behaviour.
    - **Fix:** MAA2C crash on NFIG, NumPy 2.x compatibility, and `--n_agent` / `--test_data` mismatch.

## Overview

- Four tasks of increasing difficulty: NFIG, SIG single-location, SIG multi-location and POSIG.
- Eight baselines: IDQN, Hysteretic IDQN, VDN, QMIX, IA2C, MAA2C, IPPO and MAPPO.
- A channel model following 3GPP TR 36.885 (pathloss, shadowing, fast fading), with vehicle positions from SUMO traffic simulations.
- Datasets for 4, 8 and 16 agents.
- Per-task parameter presets, JSON overrides and global seeding.
- Dependencies limited to Python, NumPy and PyTorch; runs on a CPU.

## Contents

- [News](#news)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Tasks](#tasks)
- [Algorithms](#algorithms)
- [Configuration](#configuration)
- [Outputs](#outputs)
- [Repository Structure](#repository-structure)
- [Citation](#citation)
- [License](#license)

## Installation

Requires Python 3.10 or later and [Git LFS](https://git-lfs.com), which stores the vehicle-position datasets.

```bash
git lfs install
git clone https://github.com/Deepinlab2023/V2X-MARL-Bench.git
cd V2X-MARL-Bench
pip install -r requirements.txt
```

If the CSV files in `Environment/SUMOData/` are only a few hundred bytes, they are LFS pointers; run `git lfs pull` to download the data.

## Quick Start

```bash
python main.py --env <ENV> --algo <ALGO> [options]
```

```bash
# NFIG at topology 0, 4 agents
python main.py --env NFIG --algo idql --loc 0

# SIG single-location at topology 3
python main.py --env SIG --algo maa2c --loc 3

# POSIG (partial observability)
python main.py --env POSIG --algo ippo --n_agent 16
```

| Argument | Description |
|----------|-------------|
| `--env` | Task: `NFIG`, `SIG` or `POSIG` (see [Tasks](#tasks)) |
| `--algo` | Algorithm: `idql`, `hys`, `vdn`, `qmix`, `ia2c`, `maa2c`, `ippo`, `mappo` |
| `--loc` | Topology index for NFIG and SIG single-location; omit for SIG multi-location and POSIG |
| `--n_agent` | Number of agents: `4` (default), `8` or `16` |
| `--seed` | Random seed for NumPy, Python and PyTorch |
| `--config` | Sparse JSON file that overrides the task preset (see [Configuration](#configuration)) |
| `--train_data`, `--test_data` | Custom training / evaluation CSV, replacing the default dataset |
| `--no-save_model` | Skip saving the final model checkpoint |

## Tasks

Each task adds one MARL challenge to the previous one.

| Task | Full name | Steps / episode | Topologies | Challenge added |
|------|-----------|-----------------|------------|-----------------|
| NFIG | Normal-Form Interference Game | 1 | One, selected by `--loc` | Coordination, non-stationarity |
| SIG SL | Stochastic Interference Game, single location | 50 | One, selected by `--loc` | Multi-step decisions, fast fading |
| SIG ML | Stochastic Interference Game, multiple locations | 50 | Sampled from the training set | Generalization across topologies |
| POSIG | Partially Observable Stochastic Interference Game | 50 | Sampled from the training set | Partial observability |

For NFIG and SIG SL, `--loc` selects one of the topologies below. For SIG ML and POSIG, the default evaluation set is the same nine topologies; use `--test_data` to evaluate on another set.

| `--loc` | Density (veh/km) | Distance to BS |
|---------|------------------|----------------|
| 0 | 35 | Far |
| 1 | 35 | Mid |
| 2 | 35 | Close |
| 3 | 123 | Far |
| 4 | 123 | Mid |
| 5 | 123 | Close |
| 6 | 500 | Far |
| 7 | 500 | Mid |
| 8 | 500 | Close |

Distance to BS is the longitudinal offset between the road segment and the base station. All nine topologies are available for 4, 8 and 16 agents. Vehicle-position datasets live in `Environment/SUMOData/`.

## Algorithms

The eight baselines cover two design axes: how agents learn (value-based or actor-critic) and how much is centralized during training (independent learning, IL, or centralized training with decentralized execution, CTDE).

| | Independent learning (IL) | CTDE |
|---|---|---|
| **Value-based** | IDQN (`idql`), Hysteretic IDQN (`hys`) | VDN (`vdn`), QMIX (`qmix`) |
| **Actor-critic** | IA2C (`ia2c`), IPPO (`ippo`) | MAA2C (`maa2c`), MAPPO (`mappo`) |

Value-based methods currently train one network per agent; actor-critic methods can share parameters across agents, which POSIG requires.

## Configuration

Parameters are layered, and later layers win:

1. Class defaults in `Configuration/<family>_params.py`
2. Task preset `Configuration/presets/<family>_<task>.json`, loaded automatically
3. Your overrides via `--config my.json` (only the fields you change)

`<family>` is `idql` for IDQN and Hysteretic IDQN, `qmix` for VDN and QMIX, `a2c` for IA2C and MAA2C, and `ppo` for IPPO and MAPPO.

```bash
echo '{"training_episodes": 20000, "batch_size": 64}' > my.json
python main.py --env SIG --algo ippo --loc 1 --config my.json
```

Environment settings (agent count, subchannels, power levels, fast fading on/off) are in `Configuration/env_params.py`. Opt-in configurations that are not part of the baseline are in `Configuration/experimental/`.

## Outputs

Results are written to `Results/<ALGO>/` as CSV, one file per trial:

```
{algo}_{task}_{n_agent}ag_{n_sc}sc_{FF|NFF}[_{features}][_seed{s}]_trial{n}_{timestamp}.csv
```

For example, `IA2C_NFIG_loc3_4ag_4sc_NFF_MASK_NORM_seed42_trial0_20260326_153416.csv`. Model checkpoints use the same name with a `.pt` extension.

**Compute.** The environment is simulated on the CPU and dominates run time, so a GPU gives little speed-up. To run many experiments, launch independent processes (one per core) and set `OMP_NUM_THREADS=1`.

## Repository Structure

```
V2X-MARL-Bench/
├── main.py            # Entry point
├── Configuration/     # Environment and algorithm parameters, presets, experimental configs
├── Environment/       # Channel models, reward, state generation, SUMO datasets
├── Runners/           # Experiment orchestration per algorithm family
├── Trainers/          # Algorithm-specific training loops
├── Networks/          # Actor, critic and Q-networks
├── Helpers/           # GAE, replay buffers, statistics, plotting
├── Benchmarkers/      # Periodic evaluation on the test topologies
└── Results/           # Output CSVs and checkpoints (created at run time)
```

## Citation

If you find V2X-MARL-Bench useful in your research, we would appreciate a citation:

```bibtex
@misc{wang2026v2xmarlbench,
  title         = {Multi-Agent Reinforcement Learning for {V2X} Resource Allocation: Disentangling {MARL} Challenges Through Benchmarking},
  author        = {Wang, Siyuan and Lei, Lei and Maheshwari, Pranav and Bellefeuille, Sam and Zheng, Kan},
  year          = {2026},
  eprint        = {2603.06607},
  archivePrefix = {arXiv},
  primaryClass  = {cs.MA},
  url           = {https://arxiv.org/abs/2603.06607}
}
```

The benchmark extends our preliminary study at ICC 2025:

```bibtex
@inproceedings{wang2025marl,
  author    = {Wang, Siyuan and Maheshwari, Pranav and Lei, Lei and Mei, Jie and Zheng, Kan},
  title     = {Multi-Agent {DRL} for Resource Allocation in Vehicular Networks: A Comparative Study},
  booktitle = {Proc. IEEE International Conference on Communications (ICC)},
  year      = {2025},
  pages     = {1936--1941},
  doi       = {10.1109/ICC52391.2025.11161818}
}
```

## License

Released under the [MIT License](LICENSE).
