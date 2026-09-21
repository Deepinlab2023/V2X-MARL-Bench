#!/bin/bash
# setup_cc.sh — ONE-TIME environment setup on Compute Canada.
#
# Does: clone/update the repo, pull git-lfs data, build a venv, install deps.
# Safe to re-run: if the repo/venv already exist, it just updates them
# instead of failing.
#
# Usage (run on a LOGIN node — compute nodes have no internet access):
#   bash setup_cc.sh
#
# Edit these three if your layout differs:
REPO_URL="https://github.com/Deepinlab2023/V2X-MARL-Bench.git"
REPO_DIR="V2X-MARL-Bench"          # cloned relative to the current directory
VENV_DIR="$HOME/envs/v2x"

set -euo pipefail

echo "=== 1/4  git clone / update ==="
if [ -d "$REPO_DIR/.git" ]; then
    echo "  $REPO_DIR already exists — pulling latest instead of cloning."
    cd "$REPO_DIR"
    git pull
else
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

echo "=== 2/4  git-lfs pull (large SUMO datasets) ==="
module load git-lfs/3.4.0
# --force: a pre-push hook is often already present (e.g. installed globally,
# or by the clone's own smudge filter) — safe to overwrite, it's just the
# standard lfs hook either way.
git lfs install --local --force
git lfs pull

echo "=== 3/4  Python venv ==="
module load python/3.11
if [ ! -d "$VENV_DIR" ]; then
    virtualenv --no-download "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
pip install --no-index --upgrade pip
# torch==2.6.0 per requirements.txt; falls back to closest wheel if that
# exact version isn't in the CC wheelhouse — check with `avail_wheels torch`
pip install --no-index torch numpy pandas matplotlib scipy scikit-learn

echo "=== 4/4  sanity check ==="
python -c "import torch, numpy, pandas; print('torch', torch.__version__, '| CUDA avail:', torch.cuda.is_available())"

mkdir -p logs

echo
echo "Setup complete."
echo "Repo:  $(pwd)"
echo "Venv:  $VENV_DIR"
echo "Next:  sbatch slurm/submit_array.sh   (run from inside $REPO_DIR)"
