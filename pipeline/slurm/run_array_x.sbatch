#!/bin/bash
# Array de entrenamientos exploratorios: cada línea de TASKS = "<exp> <split> [args extra de 05_train.py]".
# Salida en pipeline/runs_exploratory (06_analyze.py no lo ve; el test no se imprime).
#   sbatch --array=0-N%8 --export=ALL,TASKS=pipeline/slurm/tasks_x6.txt,CHUNK=4 pipeline/slurm/run_array_x.sbatch
#SBATCH --job-name=rev_x
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --exclude=gpu01,gpu07
#SBATCH --time=06:00:00
#SBATCH --output=pipeline/logs/x_%A_%a.out

source pipeline/slurm/env.sh
CHUNK=${CHUNK:-4}
nvidia-smi -L
start=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
mapfile -t LINES < <(sed -n "$((start + 1)),$((start + CHUNK))p" "$TASKS")
for line in "${LINES[@]}"; do
  [ -z "$line" ] && continue
  set -- $line
  echo "=== $(date +%T) $*"
  python3 pipeline/05_train.py --exp "$1" --split "$2" "${@:3}" --workers 8 --out pipeline/runs_exploratory \
    || echo "FALLO $*"
done
