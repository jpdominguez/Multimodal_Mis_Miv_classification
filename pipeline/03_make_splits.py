#!/usr/bin/env python3
"""
Paso 3 - Particiones a nivel de paciente.

  * Grupos = paciente, fusionando además imágenes casi-duplicadas (pHash<=4 y dHash<=8)
    para que nunca caigan en particiones distintas.
  * Se excluyen imágenes con conflicto de etiqueta entre duplicados (revisión clínica pendiente).
  * CV: StratifiedGroupKFold 5 folds x 3 repeticiones, estratificado por fuente x clase
    (Mis / Miv<1 mm / Miv>=1 mm). Dentro de cada fold de entrenamiento se separa un
    conjunto de VALIDACIÓN interno (~20 % de pacientes) para early stopping, selección de
    checkpoint, umbral de Youden y calibración. El fold de test no se usa para nada más.
  * LOSO (leave-one-source-out): entrenar en una fuente (con validación interna) y
    evaluar en la otra, 3 semillas.

Salida: pipeline/data/splits.csv (split_id, kind, repeat, fold, image_id, role)
        pipeline/data/exclusions.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, SPLITS_CSV, load_unified  # noqa: E402

N_FOLDS, N_REPEATS, VAL_FRAC_FOLDS = 5, 3, 5


def union_groups(df: pd.DataFrame, pairs: pd.DataFrame) -> pd.Series:
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pid = df.set_index("image_id").patient_id
    for p in pid.unique():
        find(p)
    for _, r in pairs.iterrows():
        a, b = find(pid[r.image_a]), find(pid[r.image_b])
        if a != b:
            parent[b] = a
    return df.patient_id.map(find)


def inner_val(sub: pd.DataFrame, seed: int):
    sgkf = StratifiedGroupKFold(n_splits=VAL_FRAC_FOLDS, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(sub, sub.strata, sub.group))
    return sub.image_id.values[tr], sub.image_id.values[va]


def main():
    df = load_unified()
    dup = pd.read_csv(DATA_DIR / "duplicate_candidates.csv")
    dup = dup[(dup.type != "cross_source") & (dup.phash_dist <= 4) & (dup.dhash_dist <= 8)]
    conflict = dup[dup.class3_a != dup.class3_b]
    excl = pd.DataFrame({"image_id": list(conflict.image_a) + list(conflict.image_b),
                         "reason": "identical image under different patients with conflicting labels"})
    excl.to_csv(DATA_DIR / "exclusions.csv", index=False)
    df["group"] = union_groups(df, dup)
    df = df[~df.image_id.isin(excl.image_id)].reset_index(drop=True)
    df["strata"] = df.source + "_" + df.class3.astype(str)
    print(f"Imágenes: {len(df)} (excluidas {len(excl)}) | grupos: {df.group.nunique()}")

    rows = []

    def emit(split_id, kind, rep, fold, ids, role):
        rows.extend((split_id, kind, rep, fold, i, role) for i in ids)

    for r in range(N_REPEATS):
        sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=1000 + r)
        for k, (trv, te) in enumerate(sgkf.split(df, df.strata, df.group)):
            sid = f"cv_r{r}_f{k}"
            tr_ids, va_ids = inner_val(df.iloc[trv], seed=2000 + 10 * r + k)
            emit(sid, "cv", r, k, tr_ids, "train")
            emit(sid, "cv", r, k, va_ids, "val")
            emit(sid, "cv", r, k, df.image_id.values[te], "test")

    for src, other in [("VRUH", "ISIC"), ("ISIC", "VRUH")]:
        for s in range(N_REPEATS):
            sid = f"loso_{src}2{other}_s{s}"
            tr_ids, va_ids = inner_val(df[df.source == src], seed=3000 + s)
            emit(sid, "loso", s, -1, tr_ids, "train")
            emit(sid, "loso", s, -1, va_ids, "val")
            emit(sid, "loso", s, -1, df[df.source == other].image_id.values, "test")

    sp = pd.DataFrame(rows, columns=["split_id", "kind", "repeat", "fold", "image_id", "role"])
    sp.to_csv(SPLITS_CSV, index=False)

    # ---- comprobaciones ----
    g = df.set_index("image_id").group
    pid = df.set_index("image_id").patient_id
    for sid, s in sp.groupby("split_id"):
        roles = {ro: set(x.image_id) for ro, x in s.groupby("role")}
        for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
            assert not (set(g[list(roles[a])]) & set(g[list(roles[b])])), f"fuga de grupo {sid} {a}/{b}"
            assert not (set(pid[list(roles[a])]) & set(pid[list(roles[b])])), f"fuga de paciente {sid} {a}/{b}"
    for r in range(N_REPEATS):
        te = sp[(sp.kind == "cv") & (sp.repeat == r) & (sp.role == "test")].image_id
        assert te.is_unique and len(te) == len(df), "cada imagen debe estar en test exactamente una vez por repetición"
    summ = (sp.merge(df[["image_id", "source", "label"]], on="image_id")
            .groupby(["split_id", "role"]).agg(n=("image_id", "size"), mis=("label", lambda x: (x == 0).sum()),
                                               isic=("source", lambda x: (x == "ISIC").sum())))
    summ.to_csv(DATA_DIR / "splits_summary.csv")
    print(summ.loc[["cv_r0_f0", "loso_VRUH2ISIC_s0"]].to_string())
    print("OK: sin fuga de pacientes/grupos entre train/val/test en", sp.split_id.nunique(), "particiones")


if __name__ == "__main__":
    main()
