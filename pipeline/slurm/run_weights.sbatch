#!/bin/bash
# Reentrena los modelos de la validación cruzada guardando los pesos (best_model.pth), en pipeline/runs_weights
# (no toca pipeline/runs, cuyas predicciones usan los análisis). Cada línea de TASKS = "<exp> <split>".
#   sbatch --array=0-N%8 --export=ALL,TASKS=pipeline/slurm/tasks_weights.txt,CHUNK=2 pipeline/slurm/run_weights.sbatch
#SBATCH --job-name=rev_w
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --exclude=gpu01,gpu07
#SBATCH --time=06:00:00
#SBATCH --output=pipeline/logs/w_%A_%a.out

source pipeline/slurm/env.sh
CHUNK=${CHUNK:-2}
start=$(( SLURM_ARRAY_TASK_ID * CHUNK ))
mapfile -t LINES < <(sed -n "$((start + 1)),$((start + CHUNK))p" "$TASKS")
for line in "${LINES[@]}"; do
  [ -z "$line" ] && continue
  set -- $line
  echo "=== $(date +%T) $*"
  python3 pipeline/05_train.py --exp "$1" --split "$2" --workers 8 --save-model --out pipeline/runs_weights \
    || echo "FALLO $*"
done
