#!/usr/bin/env python3
"""
M1: fusión tardía con salidas de imagen "honestas" en train (cross-fitting).

Dentro del train de cada partición se entrenan 5 modelos de imagen (05_train.py --inner K, folds por paciente).
Cada imagen de train recibe un logit FUERA DE FOLD (role=oof) y val/test reciben la media de los logits de los
5 modelos internos ("bag"). Así el combinador [logit imagen, metadatos] se ajusta con TODO el train (~1.500
imágenes) y salidas de imagen realistas, y la validación queda limpia para la puerta (sin cross-fitting).

Combinadores (ajustados solo con train):
  C1   LR (C por CV agrupada) sobre [logit, metadatos]
  C2   C1 + interacciones logit x metadatos
  C3   HistGradientBoosting sobre [logit, metadatos]
  C1L  C1 a nivel lesión (logit medio de la lesión; una fila por lesión)
  C4   prior: logit + w * logit(p_meta), con p_meta = LR de metadatos fuera de fold en train y w ajustado en train
  C5   C4 con un w por fuente (w_VRUH, w_ISIC), cada uno ajustado en el train de su fuente: en ISIC no hay
       diámetro (política de la corrección) y la edad/sexo/localización se imputan, así que su fiabilidad es otra.
       La fuente solo modula el peso de los metadatos; no entra como predictor.
  C6   C4 con el modelo de metadatos y el w ponderados por lesión (peso 1/n.º de imágenes de la lesión): las
       lesiones con muchas imágenes repiten el mismo vector y exageran la señal de los metadatos en train.
  C7   C5 + C6.
  C8   C4 con modelo de metadatos HistGradientBoosting (fuera de fold en train), en vez de LR.
  C9   prior SOLO donde los metadatos están completos (diámetro observado; en la práctica VRUH, porque en ISIC el
       diámetro se descarta por política). El modelo de metadatos y w se ajustan con esas filas de train; el resto
       de imágenes conserva su logit de imagen. (Añadido tras ver que C4 mejora dentro de VRUH y empeora en ISIC.)
  C10  C4 con el logit de metadatos centrado dentro de cada fuente (medias de train): quita el desplazamiento
       entre fuentes que introducen los metadatos imputados de ISIC y conserva el efecto dentro de cada fuente.
  C11  modelo de metadatos y w de C4 (todo el train), aplicados solo con diámetro observado (centrado en esas filas).
       (C10 y C11 añadidos tras ver la AUC por lesión y por fuente de C4.)
  (C5-C7 se añadieron el 2026-10-10 tras ver C4 en 14/15 particiones; quedan en el registro como tales.)
  *_perm  mismos combinadores con los metadatos permutados (train, val y test)
Referencia: el propio logit bag (el combinador sin metadatos es monótono en él). También se informa la
referencia `image` (modelo de imagen entrenado con todo el train) por si el bag cambia la AUC por sí solo.

Uso:   python pipeline/14_crossfit_fusion.py [--inner-exp image] [--tag ""]
Salida: runs_exploratory/m1_<comb>/<split>/preds.csv, results/tables/exploratory_gate_m1.csv, exploratory_log.csv
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
TAB = RESULTS_DIR / "tables"
META4 = ["age", "sex", "site", "diam"]
GATE_MEAN, GATE_POS = 0.005, 10
W_GRID = np.round(np.arange(0, 2.0001, 0.05), 2)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731


def gkf(n_or_X, groups, k=5):
    return list(GroupKFold(k).split(np.zeros(len(groups)), groups=groups))


def lrcv(groups):
    return LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3, 10], cv=gkf(None, groups), scoring="roc_auc",
                                max_iter=5000)


def load_inner(inner_exp, split, n_seeds=1):
    """n_seeds > 1: promedia las semillas de los modelos internos (mismos folds). El logit fuera de fold de train es
    la media de las semillas del modelo de su fold; single_sd = dispersión media de esa misma media en val/test
    (escala comparable con la de train para el reescalado _sc)."""
    oof, outer = [], []
    for sd in range(n_seeds):
        dname = f"{inner_exp}_inner" + (f"_s{sd}" if sd else "")
        for k in range(5):
            f = EXP_DIR / dname / f"{split}_in{k}" / "preds.csv"
            if not f.exists():
                return None
            d = pd.read_csv(f, dtype={"image_id": str})
            d["lg"] = logit(d.prob)
            oof.append(d[d.role == "oof"].assign(sd=sd))
            outer.append(d[d.role.isin(["val", "test"])].assign(k=k, sd=sd))
    oof = pd.concat(oof).groupby("image_id").lg.mean()
    allo = pd.concat(outer)
    W = allo.pivot_table(index="image_id", columns="k", values="lg", aggfunc="mean")   # media de semillas por fold
    single_sd = {r: W.loc[allo[allo.role == r].image_id.unique()].std().mean() for r in ["val", "test"]}
    return oof, W.mean(1), single_sd


def run_split(split, k, df, inner_exp, rng):
    got = load_inner(inner_exp, split)
    if got is None:
        return None
    oof, bag, single_sd = got
    sp = load_split(split)
    tr = df.loc[[i for i in sp["train"] if i in oof.index]]
    va = df.loc[[i for i in sp["val"] if i in bag.index]]
    te = df.loc[[i for i in sp["test"] if i in bag.index]]
    full = pd.read_csv(RUNS_DIR / inner_exp / split / "preds.csv", dtype={"image_id": str}).set_index("image_id")
    li = {"train": oof.loc[tr.index].values, "val": bag.loc[va.index].values, "test": bag.loc[te.index].values}
    # _sc: el bag (media de 5 modelos) tiene menos dispersión que el logit de UN modelo con el que se ajusta el
    # combinador; se reescala a la dispersión media de los modelos individuales en ese mismo conjunto (sin etiquetas)
    li_sc = {"train": li["train"]}
    for r in ["val", "test"]:
        li_sc[r] = (li[r] - li[r].mean()) / li[r].std() * single_sd[r] + li[r].mean()
    y = {"train": tr.label.values, "val": va.label.values, "test": te.label.values}
    D = {"train": tr, "val": va, "test": te}
    enc = MetaEncoder(META4, missing="indicator").fit(tr)
    M0 = {r: enc.transform(D[r]) for r in D}
    g_tr = tr.patient_id.values
    src_v = va.source.values
    rows = []
    base = {"split": split, "auc_val_bag": auc(y["val"], li["val"]),
            "auc_val_image": auc(y["val"], full.loc[va.index].prob.values),
            "auc_train_oof": auc(y["train"], li["train"])}

    perms = {r: rng.permutation(len(M0[r])) for r in M0}
    li_ref = li
    for perm, sc in [(False, False), (True, False), (False, True), (True, True)]:
        li = li_sc if sc else li_ref
        M = {r: (M0[r][perms[r]] if perm else M0[r]) for r in M0}
        sfx = ("_sc" if sc else "") + ("_perm" if perm else "")
        scores = {}
        # C1
        X = {r: np.c_[li[r], M[r]] for r in D}
        m = lrcv(g_tr).fit(X["train"], y["train"])
        scores["C1"] = (m.decision_function(X["val"]), m.decision_function(X["test"]))
        # C2: interacciones
        X2 = {r: np.c_[li[r], M[r], li[r][:, None] * M[r]] for r in D}
        m = lrcv(g_tr).fit(X2["train"], y["train"])
        scores["C2"] = (m.decision_function(X2["val"]), m.decision_function(X2["test"]))
        # C3: HGB (profundidad pequeña, early stopping interno)
        if not sc:      # HGB: invariante a la escala del logit
            m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                               early_stopping=True, validation_fraction=0.2, random_state=0)
            m.fit(X["train"], y["train"])
            scores["C3"] = (m.predict_proba(X["val"])[:, 1], m.predict_proba(X["test"])[:, 1])
        # C4: prior con p_meta fuera de fold en train
        lm_tr = np.zeros(len(tr))
        for a, b in gkf(None, g_tr):
            lm_tr[b] = lrcv(g_tr[a]).fit(M["train"][a], y["train"][a]).decision_function(M["train"][b])
        mm = lrcv(g_tr).fit(M["train"], y["train"])
        lm_v, lm_t = mm.decision_function(M["val"]), mm.decision_function(M["test"])
        w = float(W_GRID[int(np.nanargmax([auc(y["train"], li["train"] + w * lm_tr) for w in W_GRID]))])
        scores["C4"] = (li["val"] + w * lm_v, li["test"] + w * lm_t)
        # C5-C7
        sw = 1.0 / tr.groupby("lesion_id").image_id.transform("size").values
        lmw_tr = np.zeros(len(tr))
        for a, b in gkf(None, g_tr):
            lmw_tr[b] = lrcv(g_tr[a]).fit(M["train"][a], y["train"][a], sample_weight=sw[a]).decision_function(M["train"][b])
        mmw = lrcv(g_tr).fit(M["train"], y["train"], sample_weight=sw)
        lmw_v, lmw_t = mmw.decision_function(M["val"]), mmw.decision_function(M["test"])
        src_tr, src_te = tr.source.values, te.source.values

        def pick_w(lmt, weights=None, mask=None):
            k = np.ones(len(lmt), bool) if mask is None else mask
            if k.sum() < 20 or len(np.unique(y["train"][k])) < 2:
                return 0.0
            ww = None if weights is None else weights[k]
            a = [roc_auc_score(y["train"][k], li["train"][k] + w_ * lmt[k], sample_weight=ww) for w_ in W_GRID]
            return float(W_GRID[int(np.nanargmax(a))])

        def by_src(lmt, lv, lt, weights=None):
            ws = {so: pick_w(lmt, weights, src_tr == so) for so in ["VRUH", "ISIC"]}
            wv = np.array([ws[x] for x in src_v]); wt = np.array([ws[x] for x in src_te])
            return (li["val"] + wv * lv, li["test"] + wt * lt), ws
        scores["C5"], ws5 = by_src(lm_tr, lm_v, lm_t)
        w6 = pick_w(lmw_tr, sw)
        scores["C6"] = (li["val"] + w6 * lmw_v, li["test"] + w6 * lmw_t)
        scores["C7"], ws7 = by_src(lmw_tr, lmw_v, lmw_t, sw)
        # C8: modelo de metadatos HGB
        def hgb():
            return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=40,
                                                  l2_regularization=1.0, early_stopping=True, validation_fraction=0.2,
                                                  random_state=0)
        lmh_tr = np.zeros(len(tr))
        for a, b in gkf(None, g_tr):
            lmh_tr[b] = logit(hgb().fit(M["train"][a], y["train"][a]).predict_proba(M["train"][b])[:, 1])
        mh = hgb().fit(M["train"], y["train"])
        lmh_v, lmh_t = logit(mh.predict_proba(M["val"])[:, 1]), logit(mh.predict_proba(M["test"])[:, 1])
        c8 = lmh_tr.mean(); lmh_tr, lmh_v, lmh_t = lmh_tr - c8, lmh_v - c8, lmh_t - c8
        w8 = pick_w(lmh_tr)
        scores["C8"] = (li["val"] + w8 * lmh_v, li["test"] + w8 * lmh_t)
        # C9: prior solo con diámetro observado
        jd = enc.names.index("diam_missing")
        ob = {r: M0[r][:, jd] == 0 for r in D}                  # disponibilidad real (no se permuta)
        k9 = ob["train"]
        lm9_tr = np.zeros(len(tr))
        for a, b in gkf(None, g_tr[k9]):
            ia, ib = np.where(k9)[0][a], np.where(k9)[0][b]
            lm9_tr[ib] = lrcv(g_tr[ia]).fit(M["train"][ia], y["train"][ia]).decision_function(M["train"][ib])
        m9 = lrcv(g_tr[k9]).fit(M["train"][k9], y["train"][k9])
        c9 = lm9_tr[k9].mean()
        w9 = pick_w(np.where(k9, lm9_tr - c9, 0.0), mask=k9)
        # C10: centrado por fuente
        mu_s = {so: lm_tr[src_tr == so].mean() if (src_tr == so).any() else 0.0 for so in ["VRUH", "ISIC"]}
        cen = lambda lmx, srcx: lmx - np.array([mu_s[x] for x in srcx])  # noqa: E731
        w10 = pick_w(cen(lm_tr, src_tr))
        scores["C10"] = (li["val"] + w10 * cen(lm_v, src_v), li["test"] + w10 * cen(lm_t, src_te))
        # C11: C4 aplicado solo con diámetro observado
        jd0 = enc.names.index("diam_missing")
        ob0 = {r: M0[r][:, jd0] == 0 for r in D}
        mu_o = lm_tr[ob0["train"]].mean()
        w11 = pick_w(np.where(ob0["train"], lm_tr - mu_o, 0.0), mask=ob0["train"])
        scores["C11"] = (li["val"] + np.where(ob0["val"], w11 * (lm_v - mu_o), 0.0),
                         li["test"] + np.where(ob0["test"], w11 * (lm_t - mu_o), 0.0))
        scores["C9"] = tuple(li[r] + np.where(ob[r], w9 * (m9.decision_function(M[r]) - c9), 0.0) for r in ["val", "test"])
        # C1L: lesión
        def les(r):
            dd = D[r].assign(li=li[r])
            g = dd.groupby("lesion_id", sort=False)
            L = g.first().assign(li=g.li.mean())
            Ml = pd.DataFrame(M[r], index=dd.lesion_id.values).groupby(level=0, sort=False).first().loc[L.index].values
            return L, np.c_[L.li.values, Ml]
        (Ltr, XLtr), (Lv, XLv), (Lt, XLt) = les("train"), les("val"), les("test")
        m = lrcv(Ltr.patient_id.values).fit(XLtr, Ltr.label.values)
        bc = lambda L, s, dd: pd.Series(s, index=L.index).loc[dd.lesion_id.values].values  # noqa: E731
        scores["C1L"] = (bc(Lv, m.decision_function(XLv), va), bc(Lt, m.decision_function(XLt), te))
        ref_les = bc(Lv, Lv.li.values, va)          # referencia de C1L: el mismo logit promediado por lesión

        for c, (sv, st) in scores.items():
            name = f"m1_{c}{sfx}" + ("" if inner_exp == "image" else f"_{inner_exp}")
            ref = ref_les if c == "C1L" else li["val"]
            r = {**base, "variant": name, "auc_val": auc(y["val"], sv), "auc_val_ref": auc(y["val"], ref)}
            r["delta"] = r["auc_val"] - r["auc_val_ref"]
            r["delta_vs_image"] = r["auc_val"] - base["auc_val_image"]
            les_v = va.lesion_id.values                      # informativo: AUC por lesión (media de prob.)
            pa = pd.Series(sig(sv) if c != "C3" else sv).groupby(les_v).mean()
            pb = pd.Series(sig(ref)).groupby(les_v).mean()
            yl = pd.Series(y["val"]).groupby(les_v).first()
            r["delta_lesion"] = auc(yl.loc[pa.index], pa) - auc(yl.loc[pb.index], pb)
            for s_ in ["VRUH", "ISIC"]:
                kk = src_v == s_
                if kk.sum() > 10:
                    qa = pd.Series((sig(sv) if c != "C3" else sv)[kk]).groupby(les_v[kk]).mean()
                    qb = pd.Series(sig(ref)[kk]).groupby(les_v[kk]).mean()
                    ql = pd.Series(y["val"][kk]).groupby(les_v[kk]).first()
                    r[f"delta_lesion_{s_}"] = auc(ql.loc[qa.index], qa) - auc(ql.loc[qb.index], qb)
            for s_ in ["VRUH", "ISIC"]:
                kk = src_v == s_
                r[f"delta_{s_}"] = auc(y["val"][kk], sv[kk]) - auc(y["val"][kk], ref[kk]) if kk.sum() > 10 else np.nan
            if c == "C4":
                r["w"] = w
            elif c == "C6":
                r["w"] = w6
            elif c == "C8":
                r["w"] = w8
            elif c == "C9":
                r["w"] = w9
            elif c == "C10":
                r["w"] = w10
            elif c == "C11":
                r["w"] = w11
            elif c in ("C5", "C7"):
                r["w_VRUH"], r["w_ISIC"] = (ws5 if c == "C5" else ws7)["VRUH"], (ws5 if c == "C5" else ws7)["ISIC"]
            rows.append(r)
            o = EXP_DIR / name / split
            o.mkdir(parents=True, exist_ok=True)
            pv = sv if c == "C3" else sig(sv)
            pt = st if c == "C3" else sig(st)
            pd.concat([pd.DataFrame({"image_id": va.index, "role": "val", "label": y["val"], "prob": pv,
                                     "prob_meta_permuted": np.nan}),
                       pd.DataFrame({"image_id": te.index, "role": "test", "label": y["test"], "prob": pt,
                                     "prob_meta_permuted": np.nan})]).to_csv(o / "preds.csv", index=False)
    li = li_ref
    # referencia bag (para 06_analyze / ensembles): logit medio de los modelos internos
    name = "m1_bag" + ("" if inner_exp == "image" else f"_{inner_exp}")
    o = EXP_DIR / name / split
    o.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.DataFrame({"image_id": va.index, "role": "val", "label": y["val"], "prob": sig(li["val"]),
                             "prob_meta_permuted": np.nan}),
               pd.DataFrame({"image_id": te.index, "role": "test", "label": y["test"], "prob": sig(li["test"]),
                             "prob_meta_permuted": np.nan})]).to_csv(o / "preds.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-exp", default="image")
    a = ap.parse_args()
    df = load_unified().set_index("image_id", drop=False)
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    rows = []
    for k, s in enumerate(splits):
        r = run_split(s, k, df, a.inner_exp, np.random.default_rng(3000 + k))
        if r is None:
            print("falta", s); continue
        rows += r
        print("OK", s, f"bag {r[0]['auc_val_bag']:.4f} image {r[0]['auc_val_image']:.4f} oof {r[0]['auc_train_oof']:.4f}",
              flush=True)
    g = pd.DataFrame(rows)
    g.to_csv(TAB / f"exploratory_gate_m1_{a.inner_exp}.csv", index=False)
    print(f"\nbag vs image (sin metadatos): {(g.auc_val_bag - g.auc_val_image).groupby(g.split).first().mean():+.4f}")
    log_rows = []
    for v, x in g.groupby("variant", sort=False):
        n = len(x)
        passed = x.delta.mean() >= GATE_MEAN and (x.delta > 0).sum() >= np.ceil(GATE_POS / 15 * n)
        log_rows.append({"date": str(date.today()), "step": "M1", "variant": v, "meta_model": "combinador en train OOF",
                         "n_splits": n, "val_auc_img_mean": x.auc_val_ref.mean(), "val_auc_stack_mean": x.auc_val.mean(),
                         "val_delta_mean": x.delta.mean(), "val_delta_median": x.delta.median(),
                         "val_delta_n_positive": int((x.delta > 0).sum()), "passes_gate": bool(passed),
                         "val_delta_VRUH_mean": x.delta_VRUH.mean(), "val_delta_VRUH_n_positive": int((x.delta_VRUH > 0).sum()),
                         "val_delta_ISIC_mean": x.delta_ISIC.mean(), "val_delta_ISIC_n_positive": int((x.delta_ISIC > 0).sum()),
                         "role": "control" if "_perm" in v else "candidate"})
        print(f"{v:18s} n={n:2d} ΔAUC val vs bag {x.delta.mean():+.4f} (med {x.delta.median():+.4f}; "
              f"{(x.delta > 0).sum()}/{n}) -> {'PASA' if passed else 'no pasa'} | vs image {x.delta_vs_image.mean():+.4f} | "
              f"VRUH {x.delta_VRUH.mean():+.4f} ({(x.delta_VRUH > 0).sum()}/{n}) | ISIC {x.delta_ISIC.mean():+.4f}"
              f" | lesión {x.delta_lesion.mean():+.4f} ({(x.delta_lesion > 0).sum()}/{n})"
              + (f" [V {x.delta_lesion_VRUH.mean():+.4f} ({(x.delta_lesion_VRUH > 0).sum()}/{n})"
                 f" I {x.delta_lesion_ISIC.mean():+.4f}]" if "delta_lesion_ISIC" in x and x.delta_lesion_ISIC.notna().any()
                 else (f" [V {x.delta_lesion_VRUH.mean():+.4f}]" if "delta_lesion_VRUH" in x else ""))
              + (f" | w {x.w.mean():.2f}" if "w" in x and x.w.notna().any() else "")
              + (f" | w_V {x.w_VRUH.mean():.2f} w_I {x.w_ISIC.mean():.2f}" if "w_VRUH" in x and x.w_VRUH.notna().any() else ""),
              flush=True)
    logf = TAB / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)


if __name__ == "__main__":
    main()
