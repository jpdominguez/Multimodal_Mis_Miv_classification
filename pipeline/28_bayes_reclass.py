#!/usr/bin/env python3
"""
Análisis SECUNDARIOS POST HOC (declarados el 2026-10-21 antes de calcularlos).
Mismos datos que 19_incremental_value.py: predicciones de TEST de los modelos primarios, nivel lesión.

  A. Bayesiano del término de metadatos (β por DE), aproximación normal con EE robusto por clúster de paciente y
     tres a priori fijadas de antemano: escéptica N(0, 0.25²), débil N(0, 1), favorable N(log 1.5, 0.25²).
  B. NRI sin categorías, IDI y ΔBrier (IC bootstrap por paciente, 2000; Holm sobre las 9 pruebas):
       cv_all:  combinación con CV por paciente (10 folds) [imagen, fuente] vs [imagen, fuente, metadatos];
       cv_vruh: lo mismo en VRUH (sin fuente);
       e2e:     concat_meta vs image (probabilidades de extremo a extremo).

  python pipeline/28_bayes_reclass.py
Salida: results/tables/bayes_metadata.csv, results/tables/reclassification.csv
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, load_unified  # noqa: E402

spec = importlib.util.spec_from_file_location("iv", REV / "19_incremental_value.py")
iv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(iv)
B = 2000
PRIORS = {"escéptica N(0; 0,25²)": (0.0, 0.25), "débil N(0; 1)": (0.0, 1.0),
          "favorable N(log 1,5; 0,25²)": (np.log(1.5), 0.25)}


def lesion_table(img, meta, extra=None):
    u = load_unified().set_index("image_id")
    li, lm = iv.test_logits(img), iv.test_logits(meta)
    d = u.loc[li.index].assign(li=li.values, lm=lm.loc[li.index].values)
    agg = dict(li=("li", "mean"), lm=("lm", "mean"), label=("label", "first"),
               source=("source", "first"), patient_id=("patient_id", "first"))
    if extra:
        le = iv.test_logits(extra)
        d["le"] = le.loc[d.index].values
        agg["le"] = ("le", "mean")
    L = d.groupby("lesion_id").agg(**agg)
    L["isic"] = (L.source == "ISIC").astype(float)
    L["li_raw"] = L.li                       # logit sin estandarizar (comparación e2e)
    for c in ("li", "lm"):
        L[c] = (L[c] - L[c].mean()) / L[c].std()
    return L


def bayes(name, L, base):
    y = L.label.values.astype(int)
    X = sm.add_constant(L[base + ["lm"]].values, has_constant="add")
    m = sm.Logit(y, X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(L.patient_id)[0]})
    b, se = m.params[-1], m.bse[-1]
    rows = []
    for pn, (mu0, s0) in PRIORS.items():
        prec = 1 / s0 ** 2 + 1 / se ** 2
        mu, sd = (mu0 / s0 ** 2 + b / se ** 2) / prec, prec ** -0.5
        rows.append({"analysis": name, "prior": pn, "beta_hat": b, "se_cluster": se, "post_mean": mu, "post_sd": sd,
                     "post_OR": np.exp(mu), "post_OR_CI_low": np.exp(mu - 1.96 * sd),
                     "post_OR_CI_high": np.exp(mu + 1.96 * sd), "P_beta_gt_0": norm.sf(0, mu, sd)})
        print(f"[A] {name:18s} a priori {pn:28s} OR posterior {np.exp(mu):.2f} "
              f"(IC {np.exp(mu - 1.96 * sd):.2f}-{np.exp(mu + 1.96 * sd):.2f}); P(β>0) = {norm.sf(0, mu, sd):.3f}")
    return rows


def crossfit(L, base):
    y = L.label.values.astype(int)
    pb, pe = np.zeros(len(L)), np.zeros(len(L))
    cv = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=0)
    for tr, te in cv.split(L, y, groups=L.patient_id):
        for cols, out in ((base, pb), (base + ["lm"], pe)):
            lr = LogisticRegression(C=1e6, max_iter=1000).fit(L[cols].values[tr], y[tr])
            out[te] = lr.predict_proba(L[cols].values[te])[:, 1]
    return y, pb, pe


def metrics(y, pb, pe):
    ev, ne = y == 1, y == 0
    up, down = pe > pb, pe < pb
    nri = (up[ev].mean() - down[ev].mean()) + (down[ne].mean() - up[ne].mean())
    idi = (pe[ev].mean() - pe[ne].mean()) - (pb[ev].mean() - pb[ne].mean())
    dbrier = np.mean((pe - y) ** 2) - np.mean((pb - y) ** 2)       # negativo = mejor con metadatos
    return {"NRI": nri, "IDI": idi, "dBrier": dbrier}


def reclass(name, y, pb, pe, groups):
    est = metrics(y, pb, pe)
    rng = np.random.default_rng(2026)
    idx_by = pd.Series(np.arange(len(y))).groupby(groups).apply(np.array).to_dict()
    pats = np.array(list(idx_by))
    boot = []
    for _ in range(B):
        ii = np.concatenate([idx_by[p] for p in rng.choice(pats, len(pats))])
        boot.append(metrics(y[ii], pb[ii], pe[ii]))
    boot = pd.DataFrame(boot)
    rows = []
    for k, v in est.items():
        b = boot[k].values
        p = min(1.0, 2 * min((b <= 0).mean(), (b >= 0).mean()))
        rows.append({"comparison": name, "metric": k, "estimate": v, "CI_low": np.percentile(b, 2.5),
                     "CI_high": np.percentile(b, 97.5), "p_boot": p})
    return rows


def main():
    LA = lesion_table("image", "logreg_meta", extra="concat_meta")
    LV = lesion_table("image_vruh", "logreg_meta_vruh")
    brows = bayes("todas las fuentes", LA, ["li", "isic"]) + bayes("solo VRUH", LV, ["li"])
    pd.DataFrame(brows).to_csv(RESULTS_DIR / "tables" / "bayes_metadata.csv", index=False)

    rrows = []
    y, pb, pe = crossfit(LA, ["li", "isic"])
    rrows += reclass("cv_all: [imagen, fuente] + metadatos", y, pb, pe, LA.patient_id.values)
    y, pb, pe = crossfit(LV, ["li"])
    rrows += reclass("cv_vruh: imagen + metadatos (VRUH)", y, pb, pe, LV.patient_id.values)
    sig = lambda z: 1 / (1 + np.exp(-z))
    rrows += reclass("e2e: concat_meta vs image", LA.label.values.astype(int), sig(LA.li_raw.values),
                     sig(LA["le"].values), LA.patient_id.values)
    R = pd.DataFrame(rrows)
    R["p_holm"] = multipletests(R.p_boot, method="holm")[1]
    R.to_csv(RESULTS_DIR / "tables" / "reclassification.csv", index=False)
    for _, r in R.iterrows():
        print(f"[B] {r.comparison:40s} {r.metric:6s} {r.estimate:+.4f} (IC {r.CI_low:+.4f} a {r.CI_high:+.4f}); "
              f"p {r.p_boot:.3f}, Holm {r.p_holm:.3f}")


if __name__ == "__main__":
    main()
