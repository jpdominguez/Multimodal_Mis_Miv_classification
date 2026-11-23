#!/usr/bin/env python3
"""
Paso exploratorio 4 - Puerta en VALIDACIÓN para las variantes de receta (x4_*).

Para cada variante y partición: AUC en la validación interna (nivel imagen) frente a `image` y frente a
`concat_meta` en la misma partición. Tanto la variante como las referencias eligen su época con esa
validación, así que el sesgo de selección es simétrico. Regla 5: ΔAUC dentro de VRUH y dentro de ISIC.
No lee el test.

  python pipeline/11_gate_step4.py [--exps x4_a x4_b ...]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, load_unified  # noqa: E402
from experiments import EXPLORATORY  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
GATE_MEAN, GATE_POS_FRAC = 0.005, 10 / 15     # regla 3, proporcional al nº de particiones


def val(exp, split):
    base = EXP_DIR if exp in EXPLORATORY else RUNS_DIR
    f = base / exp / split / "preds.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype={"image_id": str})
    return d[d.role == "val"].set_index("image_id")


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=sorted(EXPLORATORY))
    args = ap.parse_args()
    src = load_unified().set_index("image_id").source
    rows = []
    for e in args.exps:
        for d in sorted((EXP_DIR / e).glob("cv_*")):
            s = d.name
            v, im, cm = val(e, s), val("image", s), val("concat_meta", s)
            if v is None or im is None:
                continue
            v = v.loc[im.index]
            r = {"variant": e, "split": s, "auc_val": auc(v.label, v.prob), "auc_val_image": auc(im.label, im.prob),
                 "auc_val_concat": auc(cm.label, cm.loc[im.index].prob) if cm is not None else np.nan}
            r["delta"] = r["auc_val"] - r["auc_val_image"]
            r["delta_vs_concat"] = r["auc_val"] - r["auc_val_concat"]
            for so in ["VRUH", "ISIC"]:
                m = (src.loc[im.index] == so).values
                r[f"delta_{so}"] = auc(v.label[m], v.prob[m]) - auc(im.label[m], im.prob[m])
            rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_step4.csv", index=False)
    log_rows = []
    for e, x in g.groupby("variant"):
        n = len(x)
        passed = x.delta.mean() >= GATE_MEAN and (x.delta > 0).sum() >= np.ceil(GATE_POS_FRAC * n)
        log_rows.append({"date": str(date.today()), "step": "4", "variant": e, "meta_model": "end-to-end concat",
                         "n_splits": n, "val_auc_img_mean": x.auc_val_image.mean(), "val_auc_stack_mean": x.auc_val.mean(),
                         "val_delta_mean": x.delta.mean(), "val_delta_median": x.delta.median(),
                         "val_delta_n_positive": int((x.delta > 0).sum()), "passes_gate": bool(passed),
                         "role": "candidate", "val_delta_vs_concat_mean": x.delta_vs_concat.mean(),
                         "val_delta_VRUH_mean": x.delta_VRUH.mean(), "val_delta_VRUH_n_positive": int((x.delta_VRUH > 0).sum()),
                         "val_delta_ISIC_mean": x.delta_ISIC.mean(), "val_delta_ISIC_n_positive": int((x.delta_ISIC > 0).sum())})
        print(f"{e:18s} n={n:2d} ΔAUC val vs image {x.delta.mean():+.4f} ({(x.delta > 0).sum()}/{n}) "
              f"-> {'PASA' if passed else 'no pasa'} | vs concat_meta {x.delta_vs_concat.mean():+.4f} | "
              f"VRUH {x.delta_VRUH.mean():+.4f} ({(x.delta_VRUH > 0).sum()}/{n}) | "
              f"ISIC {x.delta_ISIC.mean():+.4f} ({(x.delta_ISIC > 0).sum()}/{n})")
    # referencia: concat_meta vs image en las mismas particiones
    ref = g.drop_duplicates("split")
    print(f"{'(concat_meta)':18s} n={len(ref):2d} ΔAUC val vs image {(ref.auc_val_concat - ref.auc_val_image).mean():+.4f}")
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)


if __name__ == "__main__":
    main()
