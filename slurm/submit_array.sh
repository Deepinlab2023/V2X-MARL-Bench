#!/bin/bash
# submit_array.sh — SLURM job array for the train/test-distribution validation.
#
# Matrix: 27 topologies x 2 algos (idql, ippo) x 5 seeds = 270 runs.
#   - 9 topologies  = the official SIG_SL test set        (--loc 0..8)
#   - 18 topologies = SIG_ML training-set snapshots picked to match each
#                     test topology's (density, distance-to-BS) pair, 2 per
#                     test topology — see the density_distance_match.png /
#                     sampled_18_topologies.csv analysis. Hardcoded below so
#                     this script is self-contained after a fresh clone.
#
# Usage (from inside the repo root, after slurm/setup_cc.sh has run once):
#   sbatch slurm/submit_array.sh
#
#SBATCH --account=def-leil
#SBATCH --job-name=v2x_validation
#SBATCH --array=0-269
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm_%A_%a.out
#SBATCH --error=logs/slurm_%A_%a.err

set -euo pipefail

VENV_DIR="$HOME/envs/v2x"
module load python/3.11
source "$VENV_DIR/bin/activate"

# cd to the repo root. NOTE: don't try to derive this from the script's own
# path (${BASH_SOURCE[0]}) — SLURM copies the submitted script into a
# job-specific spool dir on the compute node before running it, so the
# script's on-disk location at run time is NOT where you submitted it from.
# $SLURM_SUBMIT_DIR is the one SLURM variable that reliably reflects where
# `sbatch` was invoked from, on the real (shared) filesystem.
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs Results

# --- 27-topology table -------------------------------------------------
# TOPO_LOC[i]      : --loc value
# TOPO_IS_TRAIN[i] : 0 = test-set topology (default train_data),
#                    1 = SIG_ML training-set sample (needs --train_data override)
TOPO_LOC=(0 1 2 3 4 5 6 7 8 \
          3746 40 5557 767 6387 8057 37795 31300 29691 30223 25772 29322 \
          57650 56044 56368 57171 57512 56722)
TOPO_IS_TRAIN=(0 0 0 0 0 0 0 0 0 \
               1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
N_TOPO=${#TOPO_LOC[@]}   # 27

ALGOS=(idql ippo)
N_ALGO=${#ALGOS[@]}      # 2

# Same 5 seeds used for the paper's IDQN/IPPO runs (VE_batch), for continuity.
SEEDS=(9 10 12 87 91)
N_SEED=${#SEEDS[@]}      # 5

# --- decode SLURM_ARRAY_TASK_ID -> (topo_idx, algo_idx, seed_idx) ------
idx=$SLURM_ARRAY_TASK_ID
seed_idx=$(( idx % N_SEED ))
algo_idx=$(( (idx / N_SEED) % N_ALGO ))
topo_idx=$(( idx / (N_SEED * N_ALGO) ))

LOC=${TOPO_LOC[$topo_idx]}
IS_TRAIN=${TOPO_IS_TRAIN[$topo_idx]}
ALGO=${ALGOS[$algo_idx]}
SEED=${SEEDS[$seed_idx]}

EXTRA_ARGS=()
TAG="testloc"
if [ "$IS_TRAIN" -eq 1 ]; then
    EXTRA_ARGS+=(--train_data Environment/SUMOData/SIG_ML_k16.csv)
    TAG="trainsnap"
fi

echo "[task $idx] algo=$ALGO loc=$LOC ($TAG) seed=$SEED n_agent=16"

python main.py \
    --env SIG --loc "$LOC" --algo "$ALGO" --n_agent 16 --seed "$SEED" \
    "${EXTRA_ARGS[@]}" \
    > "logs/${ALGO}_${TAG}${LOC}_seed${SEED}.log" 2>&1
