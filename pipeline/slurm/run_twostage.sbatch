#!/bin/bash
# Paso exploratorio 2: una tarea del array = una partición de CV (fase A + 4 cabezas).
#   sbatch --array=0-14%8 pipeline/slurm/run_twostage.sbatch     (desde la raíz del proyecto)
#SBATCH --job-name=rev_2stage
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gpus=1
#SBATCH --exclude=gpu01,gpu07
#SBATCH --time=03:00:00
#SBATCH --output=pipeline/logs/twostage_%A_%a.out

source pipeline/slurm/env.sh
nvidia-smi -L
SPLITS=($(for r in 0 1 2; do for f in 0 1 2 3 4; do echo cv_r${r}_f${f}; done; done))
S=${SPLITS[$SLURM_ARRAY_TASK_ID]}
echo "=== $(date +%T) twostage $S"
python3 pipeline/10_train_twostage.py --split "$S" --workers 8 || echo "FALLO twostage $S"
