#!/bin/bash
#SBATCH --job-name=v2x_ippo_new_topology
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --mem=10G
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --array=0-49
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#
# 50 array tasks = 10 loc values x 5 parallel trials each (index / 5 -> loc,
# index % 5 -> trial number, used only for this script's own log filenames --
# main.py itself has no --trial_run flag, so all 5 parallel copies for a
# given loc write with the same internal trial_run=0; they're only
# distinguished by output timestamp). TAG (NFF or FF) is passed in by
# run_batch.sh via --export when it submits this with sbatch. Each task sets
# fast_fading_enabled in Configuration/env_params.py itself, right before
# running python3 (see below) -- not the submitting wrapper -- since this
# task might sit in the queue for a while before it actually runs, and
# main.py only reads that file once, near startup.

set -euo pipefail

# Prefer $SLURM_SUBMIT_DIR (set by sbatch to the directory it was invoked
# from -- needed here since Slurm copies this script to a spool dir on the
# compute node, so ${BASH_SOURCE[0]} alone would point there instead) but
# only trust it if it actually looks like this project (contains main.py) --
# a stale $SLURM_SUBMIT_DIR inherited from an unrelated ancestor process
# (e.g. a JupyterHub terminal whose own session was itself a different Slurm
# job) must not silently redirect this into the wrong directory. Falls back
# to the script-relative path, correct when run directly (not via sbatch).
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "$SLURM_SUBMIT_DIR/main.py" ]]; then
    SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
cd "$SCRIPT_DIR"
if [[ ! -f main.py ]]; then
    echo "[array_task] Could not locate main.py from $(pwd) -- submit from the project root." >&2
    exit 1
fi

if [[ ! -f venv/bin/activate ]]; then
    echo "[array_task] Could not find venv/bin/activate from $(pwd) -- venv/ may not exist here, or this wasn't submitted from the project root." >&2
    exit 1
fi

# venv's own python binary is just a thin wrapper pointing back at this
# module's interpreter (see venv/pyvenv.cfg base-executable) -- an
# interactive/JupyterHub session loads this automatically as part of its own
# startup, but a plain sbatch job does not, which is why activation can work
# manually but fail here without this.
module load python/3.11

source ./venv/bin/activate

# Required on Linux + CUDA >= 10.2 because the trainers enable
# torch.use_deterministic_algorithms(True), which needs this set before any
# CuBLAS op runs or it throws RuntimeError.
export CUBLAS_WORKSPACE_CONFIG=:4096:8

ENV_NAME="${ENV_NAME:-SIG}"
ALGO="${ALGO:-ippo}"
N_AGENT="${N_AGENT:-16}"
TAG="${TAG:-batch}"

# Set fast_fading_enabled right here, immediately before running python3 --
# NOT in the submitting wrapper -- since this task might not actually execute
# until well after it was submitted (queue wait). Doing it here means the
# flag is always correct for whichever value this exact task needs,
# regardless of scheduling delay.
PARAMS_FILE="$SCRIPT_DIR/Configuration/env_params.py"
if [ "$TAG" == "NFF" ]; then
    FADING_VALUE="False"
else
    FADING_VALUE="True"
fi
sed -i -E "s/^(\s*self\.fast_fading_enabled\s*=\s*)\S+/\1${FADING_VALUE}/" "$PARAMS_FILE"
echo "Set fast_fading_enabled = $FADING_VALUE in $PARAMS_FILE"

LOCS=(0.0 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0)
LOC_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
TRIAL_RUN=$(( SLURM_ARRAY_TASK_ID % 5 ))
LOC="${LOCS[$LOC_IDX]}"

LOG_DIR="$SCRIPT_DIR/logs/$TAG"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/loc${LOC}_trial${TRIAL_RUN}.log"

# Stagger by trial number within this loc's 5-task group. SLURM can start
# several array tasks for the same loc within the same second if enough
# GPUs are free -- without this, their CSV filenames (timestamped when each
# process's own init_csv_logging() call runs, well after this sed/echo, not
# at launch) can collide and interleave writes into one file. 8s/trial keeps
# every trial's actual startup instant in a distinct second even accounting
# for module-load/torch-import jitter, without meaningfully denting an
# 8-hour+ training run.
sleep $(( TRIAL_RUN * 8 ))

echo "Task $SLURM_ARRAY_TASK_ID: $TAG loc=$LOC trial=$TRIAL_RUN -> $LOG_FILE"
python3 -u main.py --env "$ENV_NAME" --loc "$LOC" --algo "$ALGO" --n_agent "$N_AGENT" > "$LOG_FILE" 2>&1
