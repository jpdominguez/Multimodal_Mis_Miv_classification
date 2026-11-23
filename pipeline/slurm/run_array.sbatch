#!/bin/bash
# Array de entrenamientos. Lanzar desde la raíz del proyecto, p. ej.:
#   python3 pipeline/07_make_tasks.py --tier 1      (imprime el comando sbatch exacto)
#   sbatch --array=0-37%8 --export=ALL,TASKS=pipeline/slurm/tasks_tier1.txt,CHUNK=4 pipeline/slurm/run_array.sbatch
#SBATCH --job-name=rev_train
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --exclude=gpu01,gpu07
#SBATCH --time=06:00:00
#SBATCH --output=pipeline/logs/train_%A_%a.out

source pipeline/slurm/env.sh
CHUNK=${CHUNK:-4}
nvidia-smi -L
start=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
mapfile -t LINES < <(sed -n "$((start + 1)),$((start + CHUNK))p" "$TASKS")
for line in "${LINES[@]}"; do
  [ -z "$line" ] && continue
  set -- $line
  echo "=== $(date +%T) $1 $2"
  python3 pipeline/05_train.py --exp "$1" --split "$2" --workers 8 ${EXTRA:-} || echo "FALLO $1 $2"
done
