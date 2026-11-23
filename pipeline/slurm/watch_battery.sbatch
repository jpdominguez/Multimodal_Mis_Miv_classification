#!/bin/bash
# Vigila una batería exploratoria y deja el progreso en pipeline/results/<tag>_progress.md y <tag>_events.log
# (24_watch_battery.py; ARGS = sus argumentos).
#   sbatch --export=ALL,ARGS="--tag x8 --train-job <job> ..." pipeline/slurm/watch_battery.sbatch
#SBATCH --job-name=rev_watch
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=14:00:00
#SBATCH --output=pipeline/logs/watch_%j.out

source pipeline/slurm/env.sh
python3 pipeline/24_watch_battery.py $ARGS
