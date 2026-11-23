#!/usr/bin/env python3
"""
Paso 3b - Añade a splits.csv la partición ORIGINAL (checkpoints_v3/N_EXP_20),
en la que el conjunto de "test" era la propia validación usada para early stopping y en la que
los pacientes ISIC estaban repartidos por imagen (fuga).

Sirve para (i) verificar que el pipeline nuevo reproduce las cifras antiguas en esas condiciones
y (ii) cuantificar cuánto inflaban el rendimiento la fuga y la selección sobre el conjunto evaluado.
split_id = legacy_N20  (train = train original; val = test = validación original)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, SPLITS_CSV, load_unified  # noqa: E402

OLD = ROOT / "checkpoints_v3" / "N_EXP_20" / "densenet121_image_only"


def main():
    df = load_unified()
    excl = set(pd.read_csv(SPLITS_CSV.parent / "exclusions.csv").image_id)
    path2id = dict(zip(df.image_path, df.image_id))
    sp = pd.read_csv(SPLITS_CSV, dtype=str)
    sp = sp[~sp.split_id.str.startswith("legacy_")]
    rows = []
    for role_old, roles_new in [("train_partition.csv", ["train"]), ("val_partition.csv", ["val", "test"])]:
        old = pd.read_csv(OLD / role_old)
        ids = [path2id[p] for p in old.image_path if p in path2id and path2id[p] not in excl]
        print(role_old, len(old), "->", len(ids))
        for r in roles_new:
            rows += [("legacy_N20", "legacy", "0", "-1", i, r) for i in ids]
    new = pd.DataFrame(rows, columns=sp.columns)
    pd.concat([sp, new]).to_csv(SPLITS_CSV, index=False)
    m = df.set_index("image_id")
    tr = set(new[new.role == "train"].image_id); va = set(new[new.role == "val"].image_id)
    shared = set(m.loc[list(tr)].patient_id) & set(m.loc[list(va)].patient_id)
    print(f"Pacientes compartidos train/val en la partición original: {len(shared)} "
          f"({sum(m.loc[list(va)].patient_id.isin(shared))} imágenes de validación afectadas)")


if __name__ == "__main__":
    main()
