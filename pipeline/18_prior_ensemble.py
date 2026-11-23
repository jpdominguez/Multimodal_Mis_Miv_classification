#!/usr/bin/env python3
"""
Prior clínico (M3) y su versión por lesión (P4a) sobre un ENSEMBLE de semillas de imagen.
Definido el 2026-10-10 antes de ver sus resultados.

  ens_m3      logit(media de K semillas de imagen) + w·logit(p_meta); w por cross-fitting en validación (5 folds por
              paciente, como 13_prior_lesion.py). Referencia: el mismo ensemble de imagen (sin efecto ensemble).
  ens_p4a     igual a nivel lesión (logit medio de la lesión; LR de metadatos ajustada en las lesiones de train)
  *_perm      metadatos permutados.
  Todas las fuentes: K = 5 (image, x6_image_s1..s4) con logreg_meta. Solo VRUH: K = 3 (image_vruh,
  x6_image_vruh_s1..s2) con logreg_meta_vruh.

  python pipeline/18_prior_ensemble.py [--vruh]
"""
import argparse
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

_s = importlib.util.spec_from_file_location("p13", REV / "13_prior_lesion.py")
p13 = importlib.util.module_from_spec(_s); _s.loader.exec_module(p13)
logit, auc, best_w, crossfit, gate_stats = p13.logit, p13.auc, p13.best_w, p13.crossfit, p13.gate_stats
EXP_DIR, TAB = REV / "runs_exploratory", RESULTS_DIR / "tables"
META4 = ["age", "sex", "site", "diam"]


def preds(exp, split):
    for base in (RUNS_DIR, EXP_DIR):
        f = base / exp / split / "preds.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"image_id": str})
            return {r: g.set_index("image_id").prob for r, g in d.groupby("role")}
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vruh", action="store_true")
    a = ap.parse_args()
    img = ["image_vruh", "x6_image_vruh_s1", "x6_image_vruh_s2"] if a.vruh else \
        ["image"] + [f"x6_image_s{k}" for k in range(1, 5)]
    meta_exp = "logreg_meta_vruh" if a.vruh else "logreg_meta"
    tag = "_vruh" if a.vruh else ""
    df = load_unified().set_index("image_id", drop=False)
    if a.vruh:
        df = df[df.source == "VRUH"]
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    rows = []
    for k, s in enumerate(splits):
        seed = 5000 + k
        rng = np.random.default_rng(seed)
        P = [preds(e, s) for e in img]
        pm = preds(meta_exp, s)
        sp = load_split(s)
        ids = {r: [i for i in sp[r] if i in P[0][r].index] for r in ["val", "test"]}
        d = {r: df.loc[ids[r]] for r in ids}
        li = {r: np.mean([logit(p[r].loc[ids[r]].values) for p in P], axis=0) for r in ids}
        lm = {r: logit(pm[r].loc[ids[r]].values) for r in ids}
        c = lm["val"].mean(); lm = {r: lm[r] - c for r in lm}
        yv, gv, srcv = d["val"].label.values, d["val"].patient_id.values, d["val"].source.values
        # lesión
        tr = df.loc[[i for i in sp["train"] if i in df.index]]
        Ltr = tr.groupby("lesion_id").first()
        enc = MetaEncoder(META4, missing="indicator").fit(Ltr)
        mlr = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], scoring="roc_auc", max_iter=5000,
                                   cv=list(GroupKFold(5).split(Ltr, groups=Ltr.patient_id.values)))
        mlr.fit(enc.transform(Ltr), Ltr.label.values)
        Lv = d["val"].assign(li=li["val"]).groupby("lesion_id")
        lv_li, lv_first = Lv.li.mean(), Lv.first()
        lml = mlr.decision_function(enc.transform(lv_first)); lml = lml - lml.mean()
        ylv, glv, srclv = lv_first.label.values, lv_first.patient_id.values, lv_first.source.values
        for perm in [False, True]:
            sfx = ("_perm" if perm else "") + tag
            lmv = lm["val"][rng.permutation(len(yv))] if perm else lm["val"]
            cf = crossfit(yv, gv, seed, lambda trn, te: li["val"][te] + best_w(li["val"][trn], lmv[trn], yv[trn]) * lmv[te])
            r = gate_stats(f"ens_m3{sfx}", yv, srcv, cf, li["val"]); r.update(split=s, unit="image", w=best_w(li["val"], lmv, yv))
            # AUC por lesión de la versión por imagen
            les = d["val"].lesion_id.values
            qa = pd.Series(1 / (1 + np.exp(-cf))).groupby(les).mean(); qb = pd.Series(1 / (1 + np.exp(-li["val"]))).groupby(les).mean()
            yl = pd.Series(yv).groupby(les).first()
            r["delta_lesion"] = auc(yl.loc[qa.index], qa) - auc(yl.loc[qb.index], qb)
            rows.append(r)
            ml = lml[rng.permutation(len(ylv))] if perm else lml
            x = lv_li.loc[lv_first.index].values
            cf = crossfit(ylv, glv, seed, lambda trn, te: x[te] + best_w(x[trn], ml[trn], ylv[trn]) * ml[te])
            r = gate_stats(f"ens_p4a{sfx}", ylv, srclv, cf, x); r.update(split=s, unit="lesion", w=best_w(x, ml, ylv))
            r["delta_lesion"] = r["delta"]
            rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(TAB / f"exploratory_gate_ensprior{tag}.csv", index=False)
    log = []
    for v, x in g.groupby("variant", sort=False):
        n = len(x)
        passed = x.delta.mean() >= 0.005 and (x.delta > 0).sum() >= 10
        log.append({"date": str(date.today()), "step": "ens-prior", "variant": v, "meta_model": meta_exp, "n_splits": n,
                    "val_auc_img_mean": x.auc_val_ref.mean(), "val_auc_stack_mean": x.auc_val_new.mean(),
                    "val_delta_mean": x.delta.mean(), "val_delta_median": x.delta.median(),
                    "val_delta_n_positive": int((x.delta > 0).sum()), "passes_gate": bool(passed),
                    "val_delta_VRUH_mean": x.delta_VRUH.mean(), "val_delta_VRUH_n_positive": int((x.delta_VRUH > 0).sum()),
                    "role": "control" if "_perm" in v else "candidate"})
        print(f"{v:22s} [{x.unit.iloc[0]}] ΔAUC val {x.delta.mean():+.4f} ({(x.delta > 0).sum()}/{n}) -> "
              f"{'PASA' if passed else 'no pasa'} | lesión {x.delta_lesion.mean():+.4f} ({(x.delta_lesion > 0).sum()}/{n}) | "
              f"VRUH {x.delta_VRUH.mean():+.4f} ({(x.delta_VRUH > 0).sum()}/{n}) | w {x.w.mean():.2f}")
    logf = TAB / "exploratory_log.csv"
    old = pd.read_csv(logf); log = pd.DataFrame(log)
    pd.concat([old[~old.variant.isin(log.variant)], log]).to_csv(logf, index=False)


if __name__ == "__main__":
    main()
