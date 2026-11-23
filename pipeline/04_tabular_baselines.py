#!/usr/bin/env python3
"""
Paso 4 - Modelos tabulares sobre las MISMAS particiones.

  * logreg_meta          : solo metadatos (edad, sexo, localización, diámetro + indicadores)
  * logreg_meta_source   : metadatos + fuente
  * logreg_source        : solo fuente (cuánto rinde "adivinar el centro")
  * logreg_meta_legacy   : codificación original (ratio=0.5, ceros)
  * logreg_meta_vruh     : solo VRUH
Guarda predicciones en pipeline/runs/<modelo>/<split>/preds.csv (mismo formato que 05_train.py).

Además: ¿se puede predecir la FUENTE a partir de los metadatos? (AUC por CV)
  -> pipeline/results/tables/source_predictability.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, RUNS_DIR, SPLITS_CSV, MetaEncoder, load_unified, write_json  # noqa: E402
from experiments import TABULAR  # noqa: E402


def fit_predict(enc_args, parts, y_col="label"):
    enc = MetaEncoder(**enc_args).fit(parts["train"])
    X = {r: enc.transform(d) for r, d in parts.items()}
    clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    clf.fit(X["train"], parts["train"][y_col].values)
    coefs = dict(zip(enc.names, clf.coef_[0].round(4)))
    return {r: clf.predict_proba(X[r])[:, 1] for r in parts}, coefs


def main():
    df = load_unified().set_index("image_id", drop=False)
    sp = pd.read_csv(SPLITS_CSV, dtype={"image_id": str})
    split_ids = sorted(sp.split_id.unique())
    coef_rows = []
    for name, cfg in TABULAR.items():
        for sid in split_ids:
            s = sp[sp.split_id == sid]
            parts = {r: df.loc[s[s.role == r].image_id] for r in ["train", "val", "test"]}
            if cfg.get("sources"):
                if sid.startswith("loso"):
                    continue
                parts = {r: d[d.source == cfg["sources"]] for r, d in parts.items()}
            if parts["train"].source.nunique() == 1 and "source" in cfg["meta_vars"]:
                continue  # en LOSO la fuente es constante en train
            probs, coefs = fit_predict(dict(meta_vars=cfg["meta_vars"], missing=cfg["missing"]), parts)
            out = RUNS_DIR / name / sid
            out.mkdir(parents=True, exist_ok=True)
            pd.concat([pd.DataFrame({"image_id": parts[r].image_id.values, "role": r, "label": parts[r].label.values,
                                     "prob": probs[r], "prob_meta_permuted": np.nan}) for r in ["val", "test"]]
                      ).to_csv(out / "preds.csv", index=False)
            write_json({"exp": name, "split": sid, "config": cfg, "coefficients": coefs}, out / "run.json")
            coef_rows.append({"model": name, "split": sid, **coefs})
        print("OK", name)
    tdir = RESULTS_DIR / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coef_rows).to_csv(tdir / "logreg_coefficients_by_split.csv", index=False)

    # ---- Predictibilidad de la fuente desde los metadatos ----
    df["is_isic"] = (df.source == "ISIC").astype(int)
    rows = []
    for label, enc_args in [("metadata (indicator coding)", dict(meta_vars=["age", "sex", "site", "diam"], missing="indicator")),
                            ("metadata without missingness indicators", dict(meta_vars=["age", "sex", "site", "diam"], missing="impute")),
                            ("metadata without diameter (age, sex, site + indicators)", dict(meta_vars=["age", "sex", "site"], missing="indicator")),
                            ("original manuscript coding (ratio=0.5, zeros)", dict(meta_vars=[], missing="legacy")),
                            ("diameter missingness indicator only", dict(meta_vars=["diam"], diam_enc="bin", missing="indicator"))]:
        aucs = []
        for sid in [s for s in split_ids if s.startswith("cv_")]:
            s = sp[sp.split_id == sid]
            parts = {r: df.loc[s[s.role == r].image_id] for r in ["train", "test"]}
            if label.startswith("diameter missingness"):
                tr = parts["train"].diam_long.isna().values[:, None].astype(float)
                te = parts["test"].diam_long.isna().values[:, None].astype(float)
                clf = LogisticRegression(max_iter=2000).fit(tr, parts["train"].is_isic)
                aucs.append(roc_auc_score(parts["test"].is_isic, clf.predict_proba(te)[:, 1]))
                continue
            probs, _ = fit_predict(enc_args, parts, y_col="is_isic")
            aucs.append(roc_auc_score(parts["test"].is_isic, probs["test"]))
        rows.append({"features": label, "auc_mean": np.mean(aucs), "auc_sd": np.std(aucs, ddof=1),
                     "auc_min": np.min(aucs), "auc_max": np.max(aucs), "n_folds": len(aucs)})
    res = pd.DataFrame(rows)
    res.to_csv(tdir / "source_predictability.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
