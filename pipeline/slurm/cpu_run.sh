#!/bin/bash
# Ejecuta un comando python en el entorno del clúster: bash pipeline/slurm/cpu_run.sh script.py [args]
source "$(dirname "$0")/env.sh"
cd "$(dirname "$0")/../.." && python3 "$@"
