#!/usr/bin/env python3
"""
Paso exploratorio 3 - Modelos de metadatos más flexibles que la regresión logística (logreg_meta).
Las reglas de la puerta se fijaron antes de ver resultados.

  M1 meta_spline    LR con splines de edad y diámetro + interacción edad x localización (C por CV en train)
  M2 meta_hgb       HistGradientBoosting (faltantes nativos; profundidad/iteraciones por CV agrupada en train)
  M3 meta_hgb_nodiam  M2 sin diámetro (control de fuente)

Mismas 15 particiones de CV. Se ajusta solo con train; se guardan predicciones de val y test en
pipeline/runs_exploratory/<modelo>/<split>/preds.csv (fuera de runs/ para que 06_analyze.py no muestre su test).
Puerta 1 (solo validación): AUC en val del modelo vs AUC en val de logreg_meta, por partición.
Después: python pipeline/08_stacking.py --meta <modelo> --variants S1,S1_perm --step 3
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, SITES, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"


def enc_matrix(enc, d):
    return pd.DataFrame(enc.transform(d), columns=enc.names, index=d.index)


def fit_spline(tr, parts):
    enc = MetaEncoder(["age", "sex", "site", "diam"], missing="indicator").fit(tr)
    X = {r: enc_matrix(enc, d) for r, d in parts.items()}
    cont = ["age_z", "diam_log_z"]
    rest = [c for c in enc.names if c not in cont]
    site_cols = [f"site_{s}" for s in SITES if f"site_{s}" in enc.names]

    def add_inter(M):
        M = M.copy()
        for c in site_cols:
            M[f"age_x_{c}"] = M.age_z * M[c]
        return M
    X = {r: add_inter(M) for r, M in X.items()}
    rest += [f"age_x_{c}" for c in site_cols]
    ct = ColumnTransformer([("spl", SplineTransformer(n_knots=4, degree=3), cont)], remainder="passthrough")
    model = make_pipeline(ct, LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=5, scoring="roc_auc",
                                                   class_weight="balanced", max_iter=5000))
    model.fit(X["train"][cont + rest], tr.label.values)
    return {r: model.predict_proba(M[cont + rest])[:, 1] for r, M in X.items()}


def raw_matrix(d, use_diam):
    X = pd.DataFrame({"age": d.age.astype(float).values,
                      "sex": pd.Categorical(d.sex, categories=["FEMALE", "MALE"]).codes,
                      "site": pd.Categorical(d.site, categories=SITES).codes}, index=d.index)
    X[["sex", "site"]] = X[["sex", "site"]].replace(-1, np.nan)   # categoría ausente -> NaN (nativo)
    if use_diam:
        X["diam"] = d.diam_long.astype(float).values
    return X


def fit_hgb(tr, parts, use_diam):
    X = {r: raw_matrix(d, use_diam) for r, d in parts.items()}
    cat = [c in ("sex", "site") for c in X["train"].columns]
    base = HistGradientBoostingClassifier(categorical_features=cat, learning_rate=0.05, min_samples_leaf=20,
                                          class_weight="balanced", random_state=0)
    grid = GridSearchCV(base, {"max_depth": [2, 3], "max_iter": [50, 100, 200]}, scoring="roc_auc",
                        cv=GroupKFold(n_splits=5))
    grid.fit(X["train"], tr.label.values, groups=tr.patient_id.values)
    return {r: grid.best_estimator_.predict_proba(M)[:, 1] for r, M in X.items()}, grid.best_params_


def main():
    df = load_unified().set_index("image_id", drop=False)
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    models = {"meta_spline": lambda tr, p: (fit_spline(tr, p), {}),
              "meta_hgb": lambda tr, p: fit_hgb(tr, p, True),
              "meta_hgb_nodiam": lambda tr, p: fit_hgb(tr, p, False)}
    rows = []
    for s in splits:
        sp = load_split(s)
        parts = {r: df.loc[ids] for r, ids in sp.items()}
        ref = pd.read_csv(RUNS_DIR / "logreg_meta" / s / "preds.csv", dtype={"image_id": str})
        ref_val = ref[ref.role == "val"].set_index("image_id").prob.loc[parts["val"].image_id]
        auc_ref = roc_auc_score(parts["val"].label, ref_val)
        for name, fit in models.items():
            probs, params = fit(parts["train"], parts)
            out = EXP_DIR / name / s
            out.mkdir(parents=True, exist_ok=True)
            pd.concat([pd.DataFrame({"image_id": parts[r].image_id.values, "role": r, "label": parts[r].label.values,
                                     "prob": probs[r], "prob_meta_permuted": np.nan}) for r in ["val", "test"]]
                      ).to_csv(out / "preds.csv", index=False)
            rows.append({"model": name, "split": s, "auc_val": roc_auc_score(parts["val"].label, probs["val"]),
                         "auc_val_logreg_meta": auc_ref, **params})
        print("OK", s, flush=True)
    r = pd.DataFrame(rows)
    r["delta"] = r.auc_val - r.auc_val_logreg_meta
    r.to_csv(RESULTS_DIR / "tables" / "exploratory_step3_meta_models_val.csv", index=False)
    print(r.groupby("model").agg(auc_val=("auc_val", "mean"), auc_val_logreg=("auc_val_logreg_meta", "mean"),
                                 delta_mean=("delta", "mean"), n_positive=("delta", lambda x: int((x > 0).sum())))
          .round(4).to_string())


if __name__ == "__main__":
    main()
