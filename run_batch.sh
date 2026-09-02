#!/usr/bin/env bash
#
# Orchestrator, NOT an sbatch script itself -- run this directly with
# ./run_batch.sh. It submits the NFF array job (array_task.sh, one task per
# --loc x trial), then submits the FF array job with --dependency=afterany on
# the NFF job so SLURM guarantees no FF task can start until every NFF task
# has finished. Each array_task.sh sets fast_fading_enabled itself, right
# before running python3, based on the TAG it receives -- so there's no race
# on the shared env_params.py file regardless of how long either array waits
# in the queue.
#
# Usage:
#   ./run_batch.sh
#   ./run_batch.sh --env SIG --algo ippo --n_agent 16
#   ./run_batch.sh --array 5-49          # skip loc0 (task IDs 0-4), start at loc1
#   ./run_batch.sh --start_loc 1         # same effect, expressed as a loc index instead

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="SIG"
ALGO="ippo"
N_AGENT=16
ARRAY_RANGE="0-49"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env) ENV_NAME="$2"; shift 2 ;;
        --algo) ALGO="$2"; shift 2 ;;
        --n_agent) N_AGENT="$2"; shift 2 ;;
        --array) ARRAY_RANGE="$2"; shift 2 ;;
        --start_loc)
            # Each loc = 5 consecutive array task IDs (LOC_IDX = TASK_ID / 5,
            # per array_task.sh). --start_loc N -> begin at task ID N*5, run
            # through the last loc (9), i.e. task ID 49.
            START_LOC="$2"
            ARRAY_RANGE="$(( START_LOC * 5 ))-49"
            shift 2
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# One-time venv bootstrap. Only runs if venv/ doesn't already exist -- your
# current venv is untouched. Deliberately done HERE (run once, directly, not
# via sbatch) rather than inside array_task.sh: building it from inside the
# array would mean all concurrent tasks race to build the same venv/ and
# corrupt each other.
if [[ ! -f venv/.build_complete ]]; then
    echo "venv/.build_complete not found -- building venv..."
    module load python/3.11

    if ! command -v virtualenv &>/dev/null; then
        echo "virtualenv command not found -- installing it first..."
        pip install --no-index --user \
            --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4 \
            --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
            --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
            --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
            virtualenv
    fi

    rm -rf venv
    virtualenv --no-download venv
    source venv/bin/activate
    unset PIP_PREFIX
    export PIP_DISABLE_PIP_VERSION_CHECK=1
    pip install --no-index \
        --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v4 \
        --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/x86-64-v3 \
        --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic \
        --find-links=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/generic \
        -r requirements.txt
    touch venv/.build_complete
    deactivate
    echo "venv build complete."
fi

EXPORT_VARS="ALL,ENV_NAME=$ENV_NAME,ALGO=$ALGO,N_AGENT=$N_AGENT"

NFF_JOB=$(sbatch --parsable --array="$ARRAY_RANGE" --export="$EXPORT_VARS,TAG=NFF" array_task.sh)
echo "Submitted NFF array job $NFF_JOB (array=$ARRAY_RANGE)"

FF_JOB=$(sbatch --parsable --array="$ARRAY_RANGE" --dependency=afterany:"$NFF_JOB" --export="$EXPORT_VARS,TAG=FF" array_task.sh)
echo "Submitted FF array job $FF_JOB (array=$ARRAY_RANGE, waits for $NFF_JOB to finish)"

MOVE_JOB=$(sbatch --parsable --dependency=afterany:"$FF_JOB" --job-name=v2x_move_results_new_topology --time=00:05:00 --mem=1G --cpus-per-task=1 --wrap="
    if [ -d '$SCRIPT_DIR/Results/IPPO' ]; then
        DEST_DIR='$SCRIPT_DIR/Results/linux'
        TS=\$(date +%Y%m%d_%H%M%S)
        mkdir -p \"\$DEST_DIR\"
        mv '$SCRIPT_DIR/Results/IPPO' \"\$DEST_DIR/IPPO_\$TS\"
        echo \"Moved Results/IPPO -> \$DEST_DIR/IPPO_\$TS\"
    fi
")
echo "Submitted results-move job $MOVE_JOB (waits for $FF_JOB to finish)"

echo -e "\nCheck status with: squeue -u \$USER"
echo "Watch a run live with: tail -f logs/NFF/loc0.0_trial0.log"
