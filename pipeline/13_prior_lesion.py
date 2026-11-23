#!/usr/bin/env python3
"""
M3 (prior clínico fijo) y P4 (fusión a nivel lesión). Solo CPU.
Mismas reglas que el resto de la exploración: puerta en validación (regla 3) y comprobación dentro de VRUH (regla 5).

  M3  score = logit p_img + w * logit p_meta      (un único parámetro w >= 0; w = 0 es el modelo de imagen)
      p_meta = logreg_meta (ajustado en train). Referencia: logit p_img sin tocar.
  P4  unidad = lesión (se promedian los logits / embeddings de las imágenes de cada lesión; los metadatos
      entran una sola vez por lesión):
      P4a  M3 a nivel lesión, con un modelo de metadatos LR ajustado en las lesiones de train
      P4b  S1 a nivel lesión: LR sobre [logit img medio, logit p_meta lesión] ajustada en validación
           (referencia S0 = mismo cross-fitting con solo el logit de imagen)
      P4c  cabeza LR (C por CV agrupada) sobre [embedding medio, metadatos] ajustada en las lesiones de train,
           frente a la misma cabeza sin metadatos (P4c0). Embeddings de twostage_emb (modelo twostage_imgref).
  Controles *_perm: metadatos (o p_meta) permutados entre unidades.
  --image image_vruh: M3/P4a/P4b dentro de VRUH (image_vruh + metadatos ajustados solo con VRUH).

Puerta: cross-fitting en validación (5 folds agrupados por paciente, semillas 1000+k como 08_stacking.py).
El test se predice una vez (parámetros ajustados con toda la validación) y se guarda SIN evaluar.
Las predicciones de lesión se repiten en cada imagen de la lesión, para que 06_analyze.py las agregue igual.

Uso:   python pipeline/13_prior_lesion.py [--image image|image_vruh]
Salida: pipeline/runs_exploratory/<variante>/<split>/preds.csv, results/tables/exploratory_gate_m3p4_<image>.csv,
        fila por variante en results/tables/exploratory_log.csv
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
TAB = RESULTS_DIR / "tables"
META4 = ["age", "sex", "site", "diam"]
GATE_MEAN, GATE_POS = 0.005, 10        # regla 3 (puerta en validación)
W_GRID = np.round(np.arange(0, 2.0001, 0.05), 2)

# importa read_preds/logit/auc de 08_stacking.py (el nombre empieza por dígito)
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location("stacking", REV / "08_stacking.py")
_st = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_st)
logit, auc, read_preds = _st.logit, _st.auc, _st.read_preds


def best_w(li, lm, y):
    a = [auc(y, li + w * lm) for w in W_GRID]
    return float(W_GRID[int(np.nanargmax(a))])          # empate -> el w más pequeño


def crossfit(y, groups, seed, fit_predict):
    """fit_predict(tr_idx, te_idx) -> predicciones para te_idx. Devuelve predicción fuera de fold."""
    cf = np.full(len(y), np.nan)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed % (2 ** 31))
    for tr, te in skf.split(np.zeros(len(y)), y, groups=groups):
        cf[te] = fit_predict(tr, te)
    return cf


def lr_small():
    return LogisticRegression(C=1.0, max_iter=5000)


def lesion_table(d, s_img=None, H=None):
    """d: filas de imagen (con lesion_id); s_img: logit de imagen alineado; H: embeddings alineados."""
    d = d.copy()
    if s_img is not None:
        d["li"] = s_img
    g = d.groupby("lesion_id", sort=False)
    les = g.first()
    if s_img is not None:
        les["li"] = g.li.mean()
    les["n_img"] = g.size()
    Hl = None
    if H is not None:
        Hl = pd.DataFrame(H, index=d.lesion_id.values).groupby(level=0, sort=False).mean().loc[les.index].values
    return les, Hl


def gate_stats(name, y, src, score, ref):
    r = {"variant": name, "n_val": len(y), "auc_val_ref": auc(y, ref), "auc_val_new": auc(y, score)}
    r["delta"] = r["auc_val_new"] - r["auc_val_ref"]
    for s in ["VRUH", "ISIC"]:
        k = src == s
        r[f"delta_{s}"] = auc(y[k], score[k]) - auc(y[k], ref[k]) if k.sum() > 10 else np.nan
    return r


def save(name, split, ids_val, y_val, p_val, ids_test, y_test, p_test):
    o = EXP_DIR / name / split
    o.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.DataFrame({"image_id": ids_val, "role": "val", "label": y_val, "prob": p_val,
                             "prob_meta_permuted": np.nan}),
               pd.DataFrame({"image_id": ids_test, "role": "test", "label": y_test, "prob": p_test,
                             "prob_meta_permuted": np.nan})]).to_csv(o / "preds.csv", index=False)


sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731


def run_split(split, k, df, image_exp, meta_exp, vruh):
    seed = 1000 + k
    rng = np.random.default_rng(seed)
    sp = load_split(split)
    pi, pm = read_preds(image_exp, split), read_preds(meta_exp, split)
    ids = {r: [i for i in sp[r] if i in pi[r].index] for r in ["val", "test"]}
    tr_ids = [i for i in sp["train"] if i in df.index]          # con --image image_vruh, df ya es solo VRUH
    d = {r: df.loc[ids[r]] for r in ids}
    li = {r: logit(pi[r].loc[ids[r]].values) for r in ids}
    lm = {r: logit(pm[r].loc[ids[r]].values) for r in ids}
    mu = lm["val"].mean()
    lm = {r: lm[r] - mu for r in lm}                      # centrado (no cambia la AUC dentro de un fold)
    yv, yt = d["val"].label.values, d["test"].label.values
    gv, srcv = d["val"].patient_id.values, d["val"].source.values
    rows, tag = [], ("" if image_exp == "image" else f"_on_{image_exp}")

    # ---------------- M3 (imagen) ----------------
    for perm in [False, True]:
        name = f"m3_prior{'_perm' if perm else ''}{tag}"
        lmv, lmt = (lm["val"][rng.permutation(len(yv))], lm["test"][rng.permutation(len(yt))]) if perm \
            else (lm["val"], lm["test"])
        ws = []

        def fp(tr, te):
            w = best_w(li["val"][tr], lmv[tr], yv[tr]); ws.append(w)
            return li["val"][te] + w * lmv[te]
        cf = crossfit(yv, gv, seed, fp)
        r = gate_stats(name, yv, srcv, cf, li["val"])
        w = best_w(li["val"], lmv, yv)
        r.update(w_full=w, w_folds_mean=np.mean(ws), unit="image")
        rows.append(r)
        save(name, split, ids["val"], yv, sig(cf), ids["test"], yt, sig(li["test"] + w * lmt))

    # ---------------- P4: tablas de lesión ----------------
    Lv, _ = lesion_table(d["val"], li["val"])
    Lt, _ = lesion_table(d["test"], li["test"])
    Ltr, _ = lesion_table(df.loc[tr_ids])
    enc = MetaEncoder(META4, missing="indicator").fit(Ltr)
    Mtr, Mv, Mt = enc.transform(Ltr), enc.transform(Lv), enc.transform(Lt)
    meta_lr = LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], scoring="roc_auc", max_iter=5000,
                                   cv=list(GroupKFold(5).split(Mtr, groups=Ltr.patient_id.values)))
    meta_lr.fit(Mtr, Ltr.label.values)
    lmlv, lmlt = logit(meta_lr.predict_proba(Mv)[:, 1]), logit(meta_lr.predict_proba(Mt)[:, 1])
    c = lmlv.mean(); lmlv, lmlt = lmlv - c, lmlt - c
    ylv, ylt, glv, srclv = Lv.label.values, Lt.label.values, Lv.patient_id.values, Lv.source.values
    liv, lit = Lv.li.values, Lt.li.values
    bcast = lambda L, s, dd: pd.Series(s, index=L.index).loc[dd.lesion_id.values].values  # noqa: E731

    for perm in [False, True]:
        pv, pt = rng.permutation(len(ylv)), rng.permutation(len(ylt))
        mv, mt = (lmlv[pv], lmlt[pt]) if perm else (lmlv, lmlt)
        sfx = ("_perm" if perm else "") + tag
        # P4a: prior a nivel lesión
        cf = crossfit(ylv, glv, seed, lambda tr, te: liv[te] + best_w(liv[tr], mv[tr], ylv[tr]) * mv[te])
        w = best_w(liv, mv, ylv)
        r = gate_stats(f"p4a_lesion_prior{sfx}", ylv, srclv, cf, liv); r.update(w_full=w, unit="lesion")
        rows.append(r)
        save(f"p4a_lesion_prior{sfx}", split, ids["val"], yv, sig(bcast(Lv, cf, d["val"])),
             ids["test"], yt, sig(bcast(Lt, lit + w * mt, d["test"])))
        # P4b: S1 a nivel lesión (ref. S0 con los mismos folds)
        Xv, Xt = np.c_[liv, mv], np.c_[lit, mt]
        cf = crossfit(ylv, glv, seed, lambda tr, te: lr_small().fit(Xv[tr], ylv[tr]).predict_proba(Xv[te])[:, 1])
        cf0 = crossfit(ylv, glv, seed, lambda tr, te: lr_small().fit(Xv[tr, :1], ylv[tr]).predict_proba(Xv[te, :1])[:, 1])
        r = gate_stats(f"p4b_lesion_stack{sfx}", ylv, srclv, cf, cf0); r.update(unit="lesion")
        rows.append(r)
        save(f"p4b_lesion_stack{sfx}", split, ids["val"], yv, bcast(Lv, cf, d["val"]),
             ids["test"], yt, bcast(Lt, lr_small().fit(Xv, ylv).predict_proba(Xt)[:, 1], d["test"]))

    # ---------------- P4c: cabeza sobre embeddings medios por lesión (solo todas las fuentes) ----------------
    f = EXP_DIR / "twostage_emb" / split / "emb.npz"
    if not vruh and f.exists():
        z = np.load(f, allow_pickle=False)
        H = {r: z[f"H_{r}"] for r in ["train", "val", "test"]}
        eid = {r: z[f"id_{r}"].astype(str) for r in H}
        mu_h, sd_h = H["train"].mean(0), H["train"].std(0) + 1e-6
        E = {}
        for r in H:
            dd = df.loc[eid[r]]
            E[r] = lesion_table(dd, H=(H[r] - mu_h) / sd_h)
        (Etr, Htr), (Ev, Hv), (Et, Ht) = E["train"], E["val"], E["test"]
        Me = {"train": enc.transform(Etr), "val": enc.transform(Ev), "test": enc.transform(Et)}
        ye = {"train": Etr.label.values, "val": Ev.label.values, "test": Et.label.values}
        did = {r: df.loc[eid[r]] for r in ["val", "test"]}
        perm_idx = {r: rng.permutation(len(ye[r])) for r in ye}

        def head(X):
            m = LogisticRegressionCV(Cs=[0.001, 0.003, 0.01, 0.03, 0.1], scoring="roc_auc", max_iter=5000,
                                     class_weight="balanced",
                                     cv=list(GroupKFold(5).split(X["train"], groups=Etr.patient_id.values)))
            return m.fit(X["train"], ye["train"])
        feats = {"p4c0_lesion_head": {r: h for r, h in [("train", Htr), ("val", Hv), ("test", Ht)]},
                 "p4c_lesion_head": {r: np.c_[h, Me[r]] for r, h in [("train", Htr), ("val", Hv), ("test", Ht)]},
                 "p4c_lesion_head_perm": {r: np.c_[h, Me[r][perm_idx[r]]]
                                          for r, h in [("train", Htr), ("val", Hv), ("test", Ht)]}}
        sc = {}
        for name, X in feats.items():
            m = head(X)
            sc[name] = (m.decision_function(X["val"]), m.decision_function(X["test"]))
            save(name, split, eid["val"], did["val"].label.values, sig(bcast(Ev, sc[name][0], did["val"])),
                 eid["test"], did["test"].label.values, sig(bcast(Et, sc[name][1], did["test"])))
        for name in ["p4c_lesion_head", "p4c_lesion_head_perm"]:
            r = gate_stats(name, ye["val"], Ev.source.values, sc[name][0], sc["p4c0_lesion_head"][0])
            r.update(unit="lesion"); rows.append(r)
    return [{"split": split, **r} for r in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="image", choices=["image", "image_vruh"])
    a = ap.parse_args()
    vruh = a.image == "image_vruh"
    meta_exp = "logreg_meta_vruh" if vruh else "logreg_meta"
    df = load_unified().set_index("image_id", drop=False)
    if vruh:
        df = df[df.source == "VRUH"]
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    rows = []
    for k, s in enumerate(splits):
        rows += run_split(s, k, df, a.image, meta_exp, vruh)
        print("OK", s, flush=True)
    g = pd.DataFrame(rows)
    TAB.mkdir(parents=True, exist_ok=True)
    g.to_csv(TAB / f"exploratory_gate_m3p4_{a.image}.csv", index=False)

    log_rows = []
    for name, x in g.groupby("variant", sort=False):
        passed = (x.delta.mean() >= GATE_MEAN) and ((x.delta > 0).sum() >= GATE_POS)
        log_rows.append({"date": str(date.today()), "step": "M3/P4", "variant": name,
                         "meta_model": meta_exp if name.startswith(("m3", "p4a", "p4b")) else "MetaEncoder (lesión)",
                         "n_splits": len(x), "val_auc_img_mean": x.auc_val_ref.mean(),
                         "val_auc_stack_mean": x.auc_val_new.mean(), "val_delta_mean": x.delta.mean(),
                         "val_delta_median": x.delta.median(), "val_delta_n_positive": int((x.delta > 0).sum()),
                         "passes_gate": bool(passed),
                         "val_delta_VRUH_mean": x.delta_VRUH.mean(),
                         "val_delta_VRUH_n_positive": int((x.delta_VRUH > 0).sum()),
                         "val_delta_ISIC_mean": x.delta_ISIC.mean(),
                         "val_delta_ISIC_n_positive": int((x.delta_ISIC > 0).sum()),
                         "role": "control" if "_perm" in name else "candidate"})
        extra = f" | w medio {x.w_full.mean():.2f}" if "w_full" in x and x.w_full.notna().any() else ""
        print(f"{name:34s} [{x.unit.iloc[0]}] ΔAUC val {x.delta.mean():+.4f} (mediana {x.delta.median():+.4f}; "
              f"{(x.delta > 0).sum()}/{len(x)}) -> {'PASA' if passed else 'no pasa'} | "
              f"VRUH {x.delta_VRUH.mean():+.4f} ({(x.delta_VRUH > 0).sum()}/{len(x)}) | "
              f"ISIC {x.delta_ISIC.mean():+.4f}{extra}", flush=True)
    logf = TAB / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)


if __name__ == "__main__":
    main()
