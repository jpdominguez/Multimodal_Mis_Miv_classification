#!/usr/bin/env python3
"""
P2 + P3 - Variables nuevas y derivadas en el modelo tabular (solo VRUH, CPU).

  base      LR sobre edad, sexo, localización y diámetro (= logreg_meta_vruh, recalculado aquí)
  hx        base + antecedentes personales (personal_hx; sin indicador de faltante, imputado 0)
  derived   base + diámetro > 6 mm + edad x cabeza/cuello
  hx_derived  las dos cosas

Mismas 15 particiones de CV restringidas a VRUH; ajuste en train. Puerta 1 (solo validación): ΔAUC en val frente a
base. Las predicciones de val/test se guardan en runs_exploratory/tab_<variante>_vruh/<split>/ (test sin evaluar).
Después: python pipeline/08_stacking.py --image image_vruh --meta tab_<variante>_vruh --variants S1,S1_perm --step P2
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
META4 = ["age", "sex", "site", "diam"]
VARIANTS = {"base": (META4, False), "hx": (META4 + ["personal_hx"], False),
            "derived": (META4, True), "hx_derived": (META4 + ["personal_hx"], True)}


def design(enc, d, derived):
    X = pd.DataFrame(enc.transform(d), columns=enc.names, index=d.index)
    if derived:
        X["diam_gt6"] = (d.diam_long.astype(float) > 6).astype(float).values       # "D" del ABCD
        X["age_x_head_neck"] = X.age_z * X.site_HEAD_NECK                            # lentigo maligno
    return X.values


def main():
    df = load_unified().set_index("image_id", drop=False)
    df = df[df.source == "VRUH"]
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    rows = []
    for s in splits:
        parts = {r: df.loc[[i for i in ids if i in df.index]] for r, ids in load_split(s).items()}
        aucs = {}
        for name, (mv, derived) in VARIANTS.items():
            enc = MetaEncoder(mv, missing="indicator").fit(parts["train"])
            X = {r: design(enc, d, derived) for r, d in parts.items()}
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(X["train"], parts["train"].label)
            probs = {r: clf.predict_proba(X[r])[:, 1] for r in X}
            out = EXP_DIR / f"tab_{name}_vruh" / s
            out.mkdir(parents=True, exist_ok=True)
            pd.concat([pd.DataFrame({"image_id": parts[r].image_id.values, "role": r, "label": parts[r].label.values,
                                     "prob": probs[r], "prob_meta_permuted": np.nan}) for r in ["val", "test"]]
                      ).to_csv(out / "preds.csv", index=False)
            aucs[name] = roc_auc_score(parts["val"].label, probs["val"])
        rows += [{"variant": f"tab_{k}_vruh", "split": s, "auc_val": v, "auc_val_base": aucs["base"],
                  "delta": v - aucs["base"]} for k, v in aucs.items() if k != "base"]
    g = pd.DataFrame(rows)
    g.to_csv(RESULTS_DIR / "tables" / "exploratory_P2P3_tabular_val.csv", index=False)
    log = []
    for v, x in g.groupby("variant"):
        log.append({"date": str(date.today()), "step": "P2/P3-tab", "variant": v, "meta_model": "LR VRUH",
                    "n_splits": len(x), "val_auc_img_mean": x.auc_val_base.mean(), "val_auc_stack_mean": x.auc_val.mean(),
                    "val_delta_mean": x.delta.mean(), "val_delta_median": x.delta.median(),
                    "val_delta_n_positive": int((x.delta > 0).sum()),
                    "passes_gate": bool(x.delta.mean() >= 0.005 and (x.delta > 0).sum() >= 10), "role": "candidate"})
        print(f"{v:22s} AUC val {x.auc_val.mean():.4f} vs base {x.auc_val_base.mean():.4f}: "
              f"Δ {x.delta.mean():+.4f} ({(x.delta > 0).sum()}/{len(x)})")
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)


if __name__ == "__main__":
    main()
