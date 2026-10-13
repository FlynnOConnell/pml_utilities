#!/bin/bash
#SBATCH --job-name=asap7
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --open-mode=append

# no partition or GPU here: the submitter passes --partition and --gres on the sbatch command line
MESC=$1
UNIT=$2
CHANNEL=$3
FIRST_FRAME=$4
MAD_THRESHOLD=$5
DETREND_WINDOW=$6
DIR_SAVE=$7
[ -z "$DIR_SAVE" ] && { echo "Usage: $0 <mesc> <unit> <channel> <first_frame> <mad_threshold> <detrend_window_s> <dir_save>"; exit 1; }
[ ! -f "$MESC" ] && { echo "Error: $MESC not found"; exit 1; }

MBO="$HOME/envs/asap7/.venv/bin/mbo"
[ ! -x "$MBO" ] && { echo "Error: $MBO not executable; build the env (hpc/asap7/README.md)"; exit 1; }

echo "asap7 $UNIT of $MESC (job=${SLURM_JOB_ID:-local} node=$(hostname))"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$MBO" asap7 "$MESC" \
    --unit "$UNIT" \
    --channel "$CHANNEL" \
    --first-frame "$FIRST_FRAME" \
    --mad-threshold "$MAD_THRESHOLD" \
    --detrend-window "$DETREND_WINDOW" \
    -o "$DIR_SAVE"
exit $?
