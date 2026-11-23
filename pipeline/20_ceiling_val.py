#!/usr/bin/env python3
"""
Techo de la aportación de los metadatos (solo VALIDACIÓN; el test no se toca). 2026-10-10.

Para cada repetición se juntan las validaciones de sus 5 particiones (~1.900 imágenes). Imagen = ensemble de 5
semillas (image, x6_image_s1..s4), logit estandarizado dentro de cada partición. Con cross-fitting (10 folds
agrupados por paciente) se compara:
  lineal:   LR[imagen, fuente]            vs  LR[imagen, fuente, metadatos codificados]
  flexible: HGB[imagen, fuente]           vs  HGB[imagen, fuente, edad, sexo, localización, log-diámetro]
  (y las mismas sin fuente)
AUC por imagen y por lesión de las predicciones fuera de fold. Es un techo OPTIMISTA (combinador entrenado con más
datos que en cualquier partición real). Además, descomposición por variable (LR por lesión ajustada por imagen y
fuente; test LR de cada grupo de variables), descriptiva.

  python pipeline/20_ceiling_val.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, SITES, MetaEncoder, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
IMG = ["image"] + [f"x6_image_s{k}" for k in range(1, 5)]
META4 = ["age", "sex", "site", "diam"]


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def val_logit(exp, split):
    base = RUNS_DIR if (RUNS_DIR / exp / split).exists() else EXP_DIR
    d = pd.read_csv(base / exp / split / "preds.csv", dtype={"image_id": str})
    d = d[d.role == "val"]
    return pd.Series(logit(d.prob.values), index=d.image_id.values)


def les_auc(y, p, les):
    q = pd.Series(p).groupby(les).mean()
    yl = pd.Series(y).groupby(les).first()
    return roc_auc_score(yl.loc[q.index], q)


def main():
    u = load_unified().set_index("image_id")
    out = []
    for r in range(3):
        parts = []
        for f in range(5):
            s = f"cv_r{r}_f{f}"
            L = pd.concat([val_logit(e, s) for e in IMG], axis=1).mean(1)
            parts.append(pd.DataFrame({"li": (L - L.mean()) / L.std(), "split": s}))
        d = pd.concat(parts)
        n_dup = d.index.duplicated().sum()
        d = d.groupby(level=0).agg(li=("li", "mean"))
        d = d.join(u[["label", "source", "lesion_id", "patient_id", "age", "sex", "site", "diam_long", "diam_short"]])
        y, les, g = d.label.values, d.lesion_id.values, d.patient_id.values
        src = (d.source == "ISIC").astype(float).values[:, None]
        enc = MetaEncoder(META4, missing="indicator").fit(d)
        M = enc.transform(d)
        dpol = MetaEncoder._isic_policy(d)
        raw = np.c_[d.age.astype(float).values, (d.sex == "MALE").astype(float).values,
                    np.array([SITES.index(x) if x in SITES else np.nan for x in d.site.fillna("")], float),
                    np.log1p(dpol.diam_long.astype(float).values)]
        li = d.li.values[:, None]
        specs = {
            "lineal + fuente": (np.c_[li, src], np.c_[li, src, M], "lr"),
            "lineal sin fuente": (li, np.c_[li, M], "lr"),
            "flexible + fuente": (np.c_[li, src], np.c_[li, src, raw], "hgb"),
            "flexible sin fuente": (li, np.c_[li, raw], "hgb"),
        }
        folds = list(StratifiedGroupKFold(10, shuffle=True, random_state=r).split(li, y, groups=g))
        for name, (Xb, Xm, kind) in specs.items():
            pb, pm = np.zeros(len(y)), np.zeros(len(y))
            def mk(X, with_meta):
                if kind == "lr":
                    return LogisticRegression(C=1.0, max_iter=5000)
                cat = [X.shape[1] - 2] if with_meta else None          # localización como categórica
                return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                                      min_samples_leaf=30, l2_regularization=1.0, early_stopping=True,
                                                      validation_fraction=0.2, random_state=0,
                                                      categorical_features=cat)
            for a, b in folds:
                pb[b] = mk(Xb, False).fit(Xb[a], y[a]).predict_proba(Xb[b])[:, 1]
                pm[b] = mk(Xm, kind == "hgb").fit(Xm[a], y[a]).predict_proba(Xm[b])[:, 1]
            row = {"repeat": r, "combiner": name, "n_img": len(y), "dup_val": int(n_dup),
                   "dAUC_img": roc_auc_score(y, pm) - roc_auc_score(y, pb),
                   "dAUC_lesion": les_auc(y, pm, les) - les_auc(y, pb, les)}
            for so in ["VRUH", "ISIC"]:
                k = (d.source == so).values
                row[f"dAUC_lesion_{so}"] = les_auc(y[k], pm[k], les[k]) - les_auc(y[k], pb[k], les[k])
            out.append(row)
        # descomposición por variable (descriptiva, por lesión, ajustada por imagen y fuente)
        Ld = pd.DataFrame(M, columns=enc.names, index=d.index).assign(li=d.li.values, isic=src[:, 0], y=y, les=les)
        Ld = Ld.groupby("les").mean()
        groups = {"edad": ["age_z", "age_missing"], "sexo": ["sex_MALE", "sex_missing"],
                  "localización": [c for c in enc.names if c.startswith("site_") and c != "site_HEAD_NECK"],
                  "diámetro": ["diam_log_z"]}
        keep = [c for c in Ld.columns if Ld[c].std() > 0]                 # fuera indicadores constantes
        groups = {k: [c for c in v if c in keep] for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if v}
        allc = ["li", "isic"] + sum(groups.values(), [])
        full = sm.Logit(Ld.y, sm.add_constant(Ld[allc])).fit(disp=0, maxiter=300)
        txt = []
        for gname, cols in groups.items():
            red = sm.Logit(Ld.y, sm.add_constant(Ld[[c for c in allc if c not in cols]])).fit(disp=0, maxiter=300)
            lr = 2 * (full.llf - red.llf)
            txt.append(f"{gname} p={chi2.sf(lr, len(cols)):.3f}")
        print(f"r{r}: descomposición (por lesión, ajustada por imagen y fuente): " + " | ".join(txt)
              + f" | coef diámetro {full.params.get('diam_log_z', np.nan):+.2f}", flush=True)
    o = pd.DataFrame(out)
    o.to_csv(RESULTS_DIR / "tables" / "ceiling_val.csv", index=False)
    print(o.groupby("combiner")[["dAUC_img", "dAUC_lesion", "dAUC_lesion_VRUH", "dAUC_lesion_ISIC"]].mean().round(4))
    print(o[["repeat", "combiner", "dAUC_img", "dAUC_lesion", "dup_val"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
