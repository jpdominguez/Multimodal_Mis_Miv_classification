#!/usr/bin/env python3
"""
Paso 7 - Genera la lista de tareas (experimento, partición) para SLURM.

  python pipeline/07_make_tasks.py --tier 1          # comparación principal, controles, VRUH, LOSO
  python pipeline/07_make_tasks.py --tier 2          # ablaciones (repetición 0)
  python pipeline/07_make_tasks.py --tier 2 --all-repeats
  python pipeline/07_make_tasks.py --exp image concat_meta

Escribe pipeline/slurm/tasks_<nombre>.txt (una línea "exp split") omitiendo las ya terminadas,
e imprime el comando sbatch a ejecutar.
"""
import argparse
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RUNS_DIR, SPLITS_CSV  # noqa: E402
from experiments import EXPERIMENTS, splits_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=[1, 2])
    ap.add_argument("--exp", nargs="+")
    ap.add_argument("--all-repeats", action="store_true")
    ap.add_argument("--chunk", type=int, default=4, help="tareas secuenciales por trabajo del array")
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    split_ids = sorted(pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique())
    exps = args.exp or [e for e, c in EXPERIMENTS.items() if c["tier"] == args.tier]
    tasks = []
    for e in exps:
        for s in splits_for(e, split_ids, [0, 1, 2] if args.all_repeats else None):
            if not (RUNS_DIR / e / s / "preds.csv").exists():
                tasks.append(f"{e} {s}")
    name = args.name or (f"tier{args.tier}" if args.tier else "custom")
    out = REV / "slurm" / f"tasks_{name}.txt"
    out.write_text("\n".join(tasks) + "\n")
    n_jobs = math.ceil(len(tasks) / args.chunk)
    print(f"{len(tasks)} tareas pendientes -> {out}")
    if n_jobs:
        print(f"sbatch --array=0-{n_jobs - 1}%{args.max_parallel} --export=ALL,TASKS={out.relative_to(REV.parent)},"
              f"CHUNK={args.chunk} pipeline/slurm/run_array.sbatch")


if __name__ == "__main__":
    main()
