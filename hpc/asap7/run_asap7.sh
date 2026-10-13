#!/bin/bash
#SBATCH --job-name=asap7
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append

# no partition or GPU here: the submitter passes --partition and --gres on the sbatch command line
SCRIPT=$1
MESC=$2
UNIT=$3
CHANNEL=$4
EXCLUDE_INITIAL_FRAMES=$5
MAD_CORRELATION_THRESHOLD=$6
DETREND_WINDOW_S=$7
DIR_SAVE=$8
[ -z "$DIR_SAVE" ] && { echo "Usage: $0 <script> <mesc> <unit> <channel> <exclude_initial_frames> <mad_correlation_threshold> <detrend_window_s> <dir_save>"; exit 1; }
[ ! -f "$SCRIPT" ] && { echo "Error: $SCRIPT not found"; exit 1; }
[ ! -f "$MESC" ] && { echo "Error: $MESC not found"; exit 1; }

PYTHON_EXEC="$HOME/envs/asap7/.venv/bin/python"
[ ! -x "$PYTHON_EXEC" ] && { echo "Error: $PYTHON_EXEC not executable; build the env (hpc/asap7/README.md)"; exit 1; }

echo "asap7 $UNIT of $MESC (job=${SLURM_JOB_ID:-local} node=$(hostname))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PYTHON_EXEC" "$SCRIPT" \
    --path_mesc "$MESC" \
    --unit "$UNIT" \
    --channel "$CHANNEL" \
    --exclude_initial_frames "$EXCLUDE_INITIAL_FRAMES" \
    --mad_correlation_threshold "$MAD_CORRELATION_THRESHOLD" \
    --detrend_window_s "$DETREND_WINDOW_S" \
    --dir_save "$DIR_SAVE"
exit $?
