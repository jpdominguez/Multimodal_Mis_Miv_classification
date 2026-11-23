#!/usr/bin/env python3
"""
Variables anteriores a la biopsia aún no usadas (solo VRUH, tabular, CPU).

  base        LR sobre edad, sexo, localización y diámetro (como 12_meta_features.py)
  loc1        base + tronco posterior (LOCATION I distingue tronco anterior/posterior)
  shape       base + log(diámetro menor) + elongación (menor/mayor) + indicador de faltante del menor
  loc1_shape  las dos cosas
Puerta 1 (solo validación): ΔAUC en val frente a base (>= +0,005 y >= 10/15). Test guardado sin evaluar.
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, ROOT, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
META4 = ["age", "sex", "site", "diam"]
VARIANTS = ["base", "loc1", "shape", "loc1_shape"]


def design(enc, d, v, st):
    X = pd.DataFrame(enc.transform(d), columns=enc.names, index=d.index)
    if "loc1" in v:
        X["post_torso"] = d.loc1.fillna("").str.strip().str.lower().eq("posterior torso").astype(float).values
    if "shape" in v:
        d2 = d.diam_short.astype(float).values
        d1 = d.diam_long.astype(float).values
        miss = np.isnan(d2) | np.isnan(d1)
        l2 = np.where(miss, st["l2_med"], np.log1p(np.nan_to_num(d2)))
        r = np.where(miss, st["r_med"], np.nan_to_num(d2) / np.where(np.isnan(d1) | (d1 == 0), 1, d1))
        X["d2_log_z"] = (l2 - st["l2_mu"]) / st["l2_sd"]
        X["elong_z"] = (r - st["r_mu"]) / st["r_sd"]
        X["d2_missing"] = miss.astype(float)
    return X.values


def main():
    df = load_unified().set_index("image_id", drop=False)
    df = df[df.source == "VRUH"]
    full = pd.read_csv(ROOT / "reports" / "lista_imagenes_VRUH.csv", usecols=["image_path", "LOCATION I"]) \
        .drop_duplicates("image_path").set_index("image_path")["LOCATION I"]
    df["loc1"] = df.image_path.map(full)
    print("LOCATION I disponible:", df.loc1.notna().mean().round(3))
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    rows = []
    for s in splits:
        parts = {r: df.loc[[i for i in ids if i in df.index]] for r, ids in load_split(s).items()}
        tr = parts["train"]
        d2, d1 = tr.diam_short.astype(float), tr.diam_long.astype(float)
        ok = d2.notna() & d1.notna() & (d1 > 0)
        l2, rr = np.log1p(d2[ok]), (d2[ok] / d1[ok])
        st = {"l2_med": l2.median(), "l2_mu": l2.mean(), "l2_sd": l2.std() or 1, "r_med": rr.median(),
              "r_mu": rr.mean(), "r_sd": rr.std() or 1}
        enc = MetaEncoder(META4, missing="indicator").fit(tr)
        aucs = {}
        for v in VARIANTS:
            X = {r: design(enc, d, v, st) for r, d in parts.items()}
            clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000).fit(X["train"], tr.label)
            probs = {r: clf.predict_proba(X[r])[:, 1] for r in X}
            out = EXP_DIR / f"tab2_{v}_vruh" / s
            out.mkdir(parents=True, exist_ok=True)
            pd.concat([pd.DataFrame({"image_id": parts[r].image_id.values, "role": r, "label": parts[r].label.values,
                                     "prob": probs[r], "prob_meta_permuted": np.nan}) for r in ["val", "test"]]
                      ).to_csv(out / "preds.csv", index=False)
            aucs[v] = roc_auc_score(parts["val"].label, probs["val"])
        rows += [{"variant": f"tab2_{k}_vruh", "split": s, "auc_val": a, "auc_val_base": aucs["base"],
                  "delta": a - aucs["base"]} for k, a in aucs.items() if k != "base"]
    g = pd.DataFrame(rows)
    g.to_csv(RESULTS_DIR / "tables" / "exploratory_tab2_val.csv", index=False)
    log = []
    for v, x in g.groupby("variant"):
        log.append({"date": str(date.today()), "step": "tab2", "variant": v, "meta_model": "LR VRUH",
                    "n_splits": len(x), "val_auc_img_mean": x.auc_val_base.mean(), "val_auc_stack_mean": x.auc_val.mean(),
                    "val_delta_mean": x.delta.mean(), "val_delta_median": x.delta.median(),
                    "val_delta_n_positive": int((x.delta > 0).sum()),
                    "passes_gate": bool(x.delta.mean() >= 0.005 and (x.delta > 0).sum() >= 10), "role": "candidate"})
        print(f"{v:22s} AUC val {x.auc_val.mean():.4f} vs base {x.auc_val_base.mean():.4f}: "
              f"Δ {x.delta.mean():+.4f} ({(x.delta > 0).sum()}/{len(x)})")
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log)
    old = pd.read_csv(logf)
    pd.concat([old[~old.variant.isin(log.variant)], log]).to_csv(logf, index=False)


if __name__ == "__main__":
    main()
