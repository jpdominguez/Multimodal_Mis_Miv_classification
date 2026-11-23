#!/usr/bin/env python3
"""
Preprocesados alternativos de los metadatos dentro de M1 (CPU; reutiliza los modelos de
imagen internos de 14_crossfit_fusion.py). Definidos el 2026-10-10 ANTES de ver sus resultados.

Motivo: el problema observado es el ruido al estimar los pesos de los metadatos (y su desplazamiento entre fuentes),
así que se prueban codificaciones con menos parámetros o más robustas:
  E0  codificación actual (MetaEncoder, 13 columnas)                                  [referencia de codificación]
  E1  gruesa: log-diámetro z (+ indicador de faltante), edad >= 65, cabeza/cuello      (3-4 columnas)
  E2  cuantiles: edad y diámetro -> puntuación normal de su rango en train; categóricas como E0
  E3  continuas estandarizadas DENTRO de cada fuente (estadísticos de train por fuente); categóricas como E0
  E4  puntuación clínica fija, sin ajustar pesos: z(log-diámetro) − 1·(edad >= 65 y cabeza/cuello); 0 si falta
  E5  LR con RESTRICCIONES DE SIGNO tomadas de la literatura sobre [z(log-diámetro), indicador de faltante,
      edad >= 65, cabeza/cuello]: diámetro >= 0, edad >= 65 <= 0, cabeza/cuello <= 0 (in situ: más edad y
      cabeza/cuello, patrón de lentigo maligno). Regularización por conocimiento clínico. (Añadido el 2026-10-10
      antes de ver ningún resultado suyo.)
Combinadores (como en 14_crossfit_fusion.py, con logit bag reescalado "_sc"):
  C4  logit + w·logit(p_meta), p_meta = LR fuera de fold en train, w por AUC en train
  C6  igual con modelo de metadatos y w ponderados por lesión (1/n.º de imágenes)
  (E4 no tiene modelo de metadatos: solo w)
Controles *_perm: metadatos permutados. Puerta (regla 3) por imagen; se informa también por lesión y por fuente.

  python pipeline/17_meta_preproc.py [--inner-exp image|image_vruh]
"""
import argparse
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

_spec = importlib.util.spec_from_file_location("m1", REV / "14_crossfit_fusion.py")
m1 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m1)
load_inner, lrcv, gkf, logit, auc, sig = m1.load_inner, m1.lrcv, m1.gkf, m1.logit, m1.auc, m1.sig
EXP_DIR, TAB = REV / "runs_exploratory", RESULTS_DIR / "tables"
META4 = ["age", "sex", "site", "diam"]
W_GRID = np.round(np.arange(0, 2.0001, 0.05), 2)
GATE_MEAN, GATE_POS = 0.005, 10


def diam_log(d):
    return np.log1p(MetaEncoder._isic_policy(d).diam_long.astype(float).values)


class Enc:
    """Codificaciones E0-E4 ajustadas en train."""
    def __init__(self, kind, tr):
        self.kind = kind
        self.base = MetaEncoder(META4, missing="indicator").fit(tr)
        dl = diam_log(tr)
        self.dl_mu, self.dl_sd = np.nanmean(dl), np.nanstd(dl) or 1.0
        self.age_tr = np.sort(tr.age.astype(float).dropna().values)
        self.dl_tr = np.sort(dl[~np.isnan(dl)])
        self.src_stats = {}
        for so, g in tr.groupby("source"):
            a, d = g.age.astype(float).values, diam_log(g)
            self.src_stats[so] = (np.nanmean(a), np.nanstd(a) or 1.0,
                                  np.nanmean(d) if np.isfinite(d).any() else 0.0,
                                  np.nanstd(d) if np.isfinite(d).sum() > 1 else 1.0)

    def _qnorm(self, x, ref):
        r = (np.searchsorted(ref, x, side="left") + np.searchsorted(ref, x, side="right")) / 2
        return norm.ppf(np.clip((r + 0.5) / (len(ref) + 1), 1e-3, 1 - 1e-3))

    def transform(self, d):
        X = pd.DataFrame(self.base.transform(d), columns=self.base.names, index=d.index)
        dl = diam_log(d)
        miss = np.isnan(dl)
        age = d.age.astype(float).values
        hn = (d.site.fillna("").values == "HEAD_NECK").astype(float)
        old = (np.nan_to_num(age, nan=0) >= 65).astype(float)
        dz = np.where(miss, 0.0, (np.nan_to_num(dl) - self.dl_mu) / self.dl_sd)
        if self.kind == "E0":
            return X.values
        if self.kind in ("E1", "E5"):
            return np.c_[dz, miss.astype(float), old, hn]
        if self.kind == "E2":
            X["age_z"] = np.where(np.isnan(age), 0.0, self._qnorm(np.nan_to_num(age), self.age_tr))
            X["diam_log_z"] = np.where(miss, 0.0, self._qnorm(np.nan_to_num(dl), self.dl_tr))
            return X.values
        if self.kind == "E3":
            st = np.array([self.src_stats.get(s, (0, 1, 0, 1)) for s in d.source.values])
            X["age_z"] = np.where(np.isnan(age), 0.0, (np.nan_to_num(age) - st[:, 0]) / st[:, 1])
            X["diam_log_z"] = np.where(miss, 0.0, (np.nan_to_num(dl) - st[:, 2]) / st[:, 3])
            return X.values
        if self.kind == "E4":
            return (dz - old * hn)[:, None]
        raise ValueError(self.kind)


FINAL = {("E0", "C4"), ("E0", "C6"), ("E2", "C6"), ("E4", "fixed"), ("E5", "signs")}   # conjunto final a priori


def fit_signed_lr(X, y, lower, upper, l2=1.0, weights=None):
    """LR con cotas en los coeficientes (L-BFGS-B); devuelve (coef, intercept)."""
    from scipy.optimize import minimize
    w_ = np.ones(len(y)) if weights is None else weights
    w_ = w_ / w_.mean()

    def f(b):
        z = X @ b[1:] + b[0]
        ll = np.sum(w_ * (np.logaddexp(0, z) - y * z)) + 0.5 * l2 * np.sum(b[1:] ** 2)
        p = 1 / (1 + np.exp(-z))
        g = np.r_[np.sum(w_ * (p - y)), X.T @ (w_ * (p - y)) + l2 * b[1:]]
        return ll, g
    b0 = np.zeros(X.shape[1] + 1)
    res = minimize(f, b0, jac=True, method="L-BFGS-B", bounds=[(None, None)] + list(zip(lower, upper)))
    return res.x[1:], res.x[0]


def run_split(split, df, inner_exp, rng, n_seeds=1, final=False):
    got = load_inner(inner_exp, split, n_seeds)
    if got is None:
        return None
    oof, bag, single_sd = got
    sp = load_split(split)
    tr = df.loc[[i for i in sp["train"] if i in oof.index]]
    va = df.loc[[i for i in sp["val"] if i in bag.index]]
    te = df.loc[[i for i in sp["test"] if i in bag.index]]
    D = {"train": tr, "val": va, "test": te}
    y = {r: D[r].label.values for r in D}
    li = {"train": oof.loc[tr.index].values}
    for r in ["val", "test"]:                       # reescalado _sc (ver 14_crossfit_fusion.py)
        b = bag.loc[D[r].index].values
        li[r] = (b - b.mean()) / b.std() * single_sd[r] + b.mean()
    g_tr = tr.patient_id.values
    sw = 1.0 / tr.groupby("lesion_id").image_id.transform("size").values
    src_v, les_v = va.source.values, va.lesion_id.values
    perm = {r: rng.permutation(len(D[r])) for r in D}

    def pick_w(lmt, weights=None):
        a = [roc_auc_score(y["train"], li["train"] + w * lmt, sample_weight=weights) for w in W_GRID]
        return float(W_GRID[int(np.nanargmax(a))])

    def les_auc(score, k=None):
        k = np.ones(len(score), bool) if k is None else k
        q = pd.Series(sig(score[k])).groupby(les_v[k]).mean()
        yl = pd.Series(y["val"][k]).groupby(les_v[k]).first()
        return auc(yl.loc[q.index], q)

    rows = []
    kinds = ["E0", "E2", "E4", "E5"] if final else ["E0", "E1", "E2", "E3", "E4", "E5"]
    for kind in kinds:
        enc = Enc(kind, tr)
        M0 = {r: enc.transform(D[r]) for r in D}
        for is_perm in [False, True]:
            M = {r: (M0[r][perm[r]] if is_perm else M0[r]) for r in D}
            combos = {}
            if kind == "E4":
                lm = {r: M[r][:, 0] - M["train"][:, 0].mean() for r in D}
                w = pick_w(lm["train"])
                combos["fixed"] = (li["val"] + w * lm["val"], li["test"] + w * lm["test"], w)
            elif kind == "E5":
                lo, up = [0, None, None, None], [None, None, 0, 0]     # diam >= 0, faltante libre, edad>=65 <= 0, cab/cue <= 0
                lm_tr = np.zeros(len(tr))
                for a, b in gkf(None, g_tr):
                    cf, ic = fit_signed_lr(M["train"][a], y["train"][a], lo, up)
                    lm_tr[b] = M["train"][b] @ cf + ic
                cf, ic = fit_signed_lr(M["train"], y["train"], lo, up)
                c = lm_tr.mean()
                w = pick_w(lm_tr - c)
                combos["signs"] = (li["val"] + w * (M["val"] @ cf + ic - c), li["test"] + w * (M["test"] @ cf + ic - c), w)
            else:
                for cname, weights in [("C4", None), ("C6", sw)]:
                    lm_tr = np.zeros(len(tr))
                    for a, b in gkf(None, g_tr):
                        kw = {} if weights is None else {"sample_weight": weights[a]}
                        lm_tr[b] = lrcv(g_tr[a]).fit(M["train"][a], y["train"][a], **kw).decision_function(M["train"][b])
                    kw = {} if weights is None else {"sample_weight": weights}
                    mm = lrcv(g_tr).fit(M["train"], y["train"], **kw)
                    c = lm_tr.mean()
                    w = pick_w(lm_tr - c, weights)
                    combos[cname] = (li["val"] + w * (mm.decision_function(M["val"]) - c),
                                     li["test"] + w * (mm.decision_function(M["test"]) - c), w)
            for cname, (sv, st, w) in combos.items():
                if final and (kind, cname) not in FINAL:
                    continue
                name = f"pp_{kind}_{cname}" + ("_perm" if is_perm else "") + ("" if inner_exp == "image" else f"_{inner_exp}") \
                    + (f"_is{n_seeds}" if n_seeds > 1 else "")
                r = {"variant": name, "split": split, "w": w, "auc_val_ref": auc(y["val"], li["val"]),
                     "auc_val": auc(y["val"], sv)}
                r["delta"] = r["auc_val"] - r["auc_val_ref"]
                r["delta_lesion"] = les_auc(sv) - les_auc(li["val"])
                for so in ["VRUH", "ISIC"]:
                    k = src_v == so
                    r[f"delta_{so}"] = auc(y["val"][k], sv[k]) - auc(y["val"][k], li["val"][k]) if k.sum() > 10 else np.nan
                    r[f"delta_lesion_{so}"] = les_auc(sv, k) - les_auc(li["val"], k) if k.sum() > 10 else np.nan
                rows.append(r)
                o = EXP_DIR / name / split
                o.mkdir(parents=True, exist_ok=True)
                pd.concat([pd.DataFrame({"image_id": va.index, "role": "val", "label": y["val"], "prob": sig(sv),
                                         "prob_meta_permuted": np.nan}),
                           pd.DataFrame({"image_id": te.index, "role": "test", "label": y["test"], "prob": sig(st),
                                         "prob_meta_permuted": np.nan})]).to_csv(o / "preds.csv", index=False)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-exp", default="image")
    ap.add_argument("--inner-seeds", type=int, default=1)
    ap.add_argument("--final", action="store_true", help="solo el conjunto final fijado a priori")
    a = ap.parse_args()
    df = load_unified().set_index("image_id", drop=False)
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    rows = []
    for k, s in enumerate(splits):
        r = run_split(s, df, a.inner_exp, np.random.default_rng(4000 + k), a.inner_seeds, a.final)
        if r is None:
            print("falta", s); continue
        rows += r
    g = pd.DataFrame(rows)
    sfx = (f"_is{a.inner_seeds}" if a.inner_seeds > 1 else "") + ("_final" if a.final else "")
    g.to_csv(TAB / f"exploratory_gate_preproc_{a.inner_exp}{sfx}.csv", index=False)
    log_rows = []
    for v, x in g.groupby("variant", sort=False):
        n = len(x)
        passed = x.delta.mean() >= GATE_MEAN and (x.delta > 0).sum() >= np.ceil(GATE_POS / 15 * n)
        log_rows.append({"date": str(date.today()), "step": "preproc", "variant": v, "meta_model": "M1 + preprocesado",
                         "n_splits": n, "val_auc_img_mean": x.auc_val_ref.mean(), "val_auc_stack_mean": x.auc_val.mean(),
                         "val_delta_mean": x.delta.mean(), "val_delta_median": x.delta.median(),
                         "val_delta_n_positive": int((x.delta > 0).sum()), "passes_gate": bool(passed),
                         "val_delta_VRUH_mean": x.delta_VRUH.mean(), "val_delta_VRUH_n_positive": int((x.delta_VRUH > 0).sum()),
                         "val_delta_ISIC_mean": x.delta_ISIC.mean(), "val_delta_ISIC_n_positive": int((x.delta_ISIC > 0).sum()),
                         "role": "control" if "_perm" in v else "candidate"})
        print(f"{v:30s} ΔAUC img {x.delta.mean():+.4f} ({(x.delta > 0).sum()}/{n}) -> {'PASA' if passed else 'no pasa'} | "
              f"lesión {x.delta_lesion.mean():+.4f} ({(x.delta_lesion > 0).sum()}/{n}) | "
              f"VRUH img {x.delta_VRUH.mean():+.4f} ({(x.delta_VRUH > 0).sum()}/{n}) les {x.delta_lesion_VRUH.mean():+.4f} | "
              f"ISIC img {x.delta_ISIC.mean():+.4f} | w {x.w.mean():.2f}", flush=True)
    logf = TAB / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    old = pd.read_csv(logf)
    pd.concat([old[~old.variant.isin(log.variant)], log]).to_csv(logf, index=False)


if __name__ == "__main__":
    main()
