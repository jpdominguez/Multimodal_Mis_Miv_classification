#!/usr/bin/env python3
"""
Análisis SECUNDARIO POST HOC (declarado el 2026-10-10 antes de calcularlo):
valor incremental de los metadatos sobre la salida del modelo de imagen, con el test recomendado para modelos
anidados (Vickers 2011; Demler 2012; Pepe 2013): razón de verosimilitudes del término de metadatos, y estimación
(no test) del ΔAUC con IC bootstrap por paciente.

Datos: predicciones de TEST de los modelos primarios (media de logits de las 3 repeticiones de CV), nivel lesión.
  principal:   label ~ z(logit image) + fuente   vs   + z(logit logreg_meta)
  VRUH:        label ~ z(logit image_vruh)        vs   + z(logit logreg_meta_vruh)
  sensibilidad: sin término de fuente; nivel imagen con EE por clúster de paciente.

  python pipeline/19_incremental_value.py
Salida: results/tables/incremental_value.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, RUNS_DIR, load_unified  # noqa: E402

B = 2000


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def test_logits(exp):
    out = []
    for r in range(3):
        for f in range(5):
            d = pd.read_csv(RUNS_DIR / exp / f"cv_r{r}_f{f}" / "preds.csv", dtype={"image_id": str})
            d = d[d.role == "test"]
            out.append(pd.DataFrame({"image_id": d.image_id.values, "rep": r, "lg": logit(d.prob.values)}))
    o = pd.concat(out)
    assert o.groupby(["image_id", "rep"]).size().max() == 1          # cada imagen, una vez por repetición
    return o.groupby("image_id").lg.mean()


def fit(y, X):
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0, maxiter=200)


def analyse(name, d, cols_base, unit, groups):
    y = d.label.values.astype(int)
    Xb, Xe = d[cols_base].values, d[cols_base + ["lm"]].values
    mb, me = fit(y, Xb), fit(y, Xe)
    lr = 2 * (me.llf - mb.llf)
    p_lr = chi2.sf(lr, 1)
    mc = sm.Logit(y, sm.add_constant(Xe, has_constant="add")).fit(
        disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    beta, se_cl = me.params[-1], mc.bse[-1]
    auc_b = roc_auc_score(y, mb.fittedvalues if hasattr(mb, "fittedvalues") else mb.predict())
    auc_e = roc_auc_score(y, me.fittedvalues if hasattr(me, "fittedvalues") else me.predict())
    # bootstrap por paciente
    rng = np.random.default_rng(12345)
    pid = pd.Series(groups)
    idx_by = {p: np.where(groups == p)[0] for p in pid.unique()}
    pats = np.array(list(idx_by))
    bb, dd = [], []
    for _ in range(B):
        ii = np.concatenate([idx_by[p] for p in rng.choice(pats, len(pats))])
        try:
            b_ = fit(y[ii], Xb[ii]); e_ = fit(y[ii], Xe[ii])
            bb.append(e_.params[-1])
            dd.append(roc_auc_score(y[ii], e_.fittedvalues) - roc_auc_score(y[ii], b_.fittedvalues))
        except Exception:
            continue
    r = {"analysis": name, "unit": unit, "n": len(y), "events": int(y.sum()), "covariates_base": "+".join(cols_base),
         "beta_meta_per_SD": beta, "OR_meta_per_SD": np.exp(beta),
         "beta_boot_CI_low": np.percentile(bb, 2.5), "beta_boot_CI_high": np.percentile(bb, 97.5),
         "se_cluster": se_cl, "p_wald_cluster": 2 * (1 - __import__("scipy").stats.norm.cdf(abs(beta / se_cl))),
         "LR_stat": lr, "p_LR": p_lr, "AUC_base_apparent": auc_b, "AUC_ext_apparent": auc_e,
         "dAUC_apparent": auc_e - auc_b, "dAUC_boot_CI_low": np.percentile(dd, 2.5),
         "dAUC_boot_CI_high": np.percentile(dd, 97.5), "n_boot_ok": len(bb)}
    print(f"{name:34s} [{unit}] n={r['n']} β_meta/DE={beta:+.3f} (OR {np.exp(beta):.2f}; IC boot "
          f"{r['beta_boot_CI_low']:+.3f} a {r['beta_boot_CI_high']:+.3f}) | LR={lr:.2f} p={p_lr:.4f} | "
          f"Wald clúster p={r['p_wald_cluster']:.4f} | ΔAUC aparente {r['dAUC_apparent']:+.4f} "
          f"(IC {r['dAUC_boot_CI_low']:+.4f} a {r['dAUC_boot_CI_high']:+.4f})", flush=True)
    return r


def main():
    u = load_unified().set_index("image_id")
    rows = []
    for name, img, meta, vruh in [("todas las fuentes", "image", "logreg_meta", False),
                                  ("solo VRUH", "image_vruh", "logreg_meta_vruh", True)]:
        li, lm = test_logits(img), test_logits(meta)
        d = u.loc[li.index].assign(li=li.values, lm=lm.loc[li.index].values)
        d["isic"] = (d.source == "ISIC").astype(float)
        # nivel lesión (los metadatos son de lesión; la imagen se promedia)
        L = d.groupby("lesion_id").agg(li=("li", "mean"), lm=("lm", "mean"), label=("label", "first"),
                                       isic=("isic", "first"), patient_id=("patient_id", "first"))
        for D in (L, d):
            for c in ("li", "lm"):
                D[c] = (D[c] - D[c].mean()) / D[c].std()
        base = ["li"] if vruh else ["li", "isic"]
        rows.append(analyse(name + (" (principal)" if not vruh else ""), L, base, "lesión", L.patient_id.values))
        if not vruh:
            rows.append(analyse(name + " sin fuente", L, ["li"], "lesión", L.patient_id.values))
        rows.append(analyse(name, d, base, "imagen (EE clúster)", d.patient_id.values))
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "tables" / "incremental_value.csv", index=False)


if __name__ == "__main__":
    main()
