#!/bin/bash
# Batería x12 (TabPFN y MONET): embeddings + puerta en validación, con el entorno pipeline/.venv_t2.
#   sbatch pipeline/slurm/run_x12.sbatch
#SBATCH --job-name=rev_x12
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --exclude=gpu01,gpu07
#SBATCH --time=04:00:00
#SBATCH --output=pipeline/logs/x12_%j.out

source pipeline/slurm/env.sh
unset PYTHONPATH          # el módulo añade su PyTorch 1.13; el entorno trae PyTorch 2.5
source pipeline/.venv_t2/bin/activate
EV=pipeline/results/x12_events.log
echo "$(date '+%Y-%m-%d %H:%M')  Inicio del job $SLURM_JOB_ID: embeddings de MONET" >> $EV
[ -f pipeline/data/monet_emb.npz ] || python pipeline/29_x12_tabpfn_monet.py --extract || { echo "$(date '+%Y-%m-%d %H:%M')  ERROR en la extracción" >> $EV; exit 1; }
echo "$(date '+%Y-%m-%d %H:%M')  Embeddings listos; ejecutando la puerta" >> $EV
python pipeline/29_x12_tabpfn_monet.py > pipeline/results/tables/x12_gate_output.txt 2>&1 \
  && echo "$(date '+%Y-%m-%d %H:%M')  Puerta: $(tail -n 1 pipeline/results/tables/x12_gate_output.txt)" >> $EV \
  || echo "$(date '+%Y-%m-%d %H:%M')  ERROR en la puerta (ver results/tables/x12_gate_output.txt)" >> $EV
