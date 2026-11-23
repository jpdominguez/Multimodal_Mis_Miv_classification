#!/usr/bin/env python3
"""
Apertura del TEST de las variantes exploratorias (protocolo declarado ANTES de
calcular, 2026-10-10). Se informa todo; Holm sobre la familia A.

  Familia A: conjunto final M1 con 5 semillas internas (17_meta_preproc.py --final --inner-seeds 5) frente a su
             propia puntuación de imagen (bag de modelos internos reescalado, igual que en 17_meta_preproc.py).
  Contexto:  las mismas frente al modelo `image` / `image_vruh` primario (incluye efecto ensemble).
  Familia B: ensemble de 5 semillas concat_meta vs 5 de image; ídem VRUH.
  AUC por lesión: media de las 3 repeticiones por imagen -> media por lesión. IC y p bootstrap por paciente (2000).

  python pipeline/21_test_final.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, load_split, load_unified  # noqa: E402

_s = importlib.util.spec_from_file_location("m1", REV / "14_crossfit_fusion.py")
m1 = importlib.util.module_from_spec(_s); _s.loader.exec_module(m1)
EXP_DIR = REV / "runs_exploratory"
SPLITS = [f"cv_r{r}_f{f}" for r in range(3) for f in range(5)]
B = 2000
sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731


def test_prob(exp, split):
    for base in (RUNS_DIR, EXP_DIR):
        f = base / exp / split / "preds.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"image_id": str})
            d = d[d.role == "test"]
            return pd.Series(d.prob.values, index=d.image_id.values)
    raise FileNotFoundError(f"{exp}/{split}")


def bag_test(inner_exp, split):
    """Puntuación de imagen de referencia de 17_meta_preproc.py (bag de 5 semillas internas, reescalado _sc)."""
    oof, bag, single_sd = m1.load_inner(inner_exp, split, 5)
    ids = [i for i in load_split(split)["test"] if i in bag.index]
    b = bag.loc[ids].values
    return pd.Series(sig((b - b.mean()) / b.std() * single_sd["test"] + b.mean()), index=ids)


def per_image(fn):
    """Media por imagen de las 3 repeticiones (cada imagen está en test una vez por repetición)."""
    return pd.concat([fn(s) for s in SPLITS]).groupby(level=0).mean()


def lesion(p, u):
    d = u.loc[p.index].assign(p=p.values)
    return d.groupby("lesion_id").agg(p=("p", "mean"), y=("label", "first"), pid=("patient_id", "first"),
                                      src=("source", "first"))


def compare(pa, pb, u, rng, subset=None):
    La, Lb = lesion(pa, u), lesion(pb, u)
    idx = La.index.intersection(Lb.index)
    if subset is not None:
        idx = idx[(La.loc[idx].src == subset).values]
    La, Lb = La.loc[idx], Lb.loc[idx]
    y, a, b, pid = La.y.values, La.p.values, Lb.p.values, La.pid.values
    d0 = roc_auc_score(y, a) - roc_auc_score(y, b)
    groups = pd.Series(np.arange(len(pid))).groupby(pid).apply(np.array).values
    ds = []
    for _ in range(B):
        ii = np.concatenate([groups[k] for k in rng.integers(0, len(groups), len(groups))])
        if len(np.unique(y[ii])) < 2:
            continue
        ds.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    ds = np.array(ds)
    p = min(1.0, 2 * min((ds <= 0).mean(), (ds >= 0).mean()))
    return {"n_lesions": len(y), "AUC_A": roc_auc_score(y, a), "AUC_B": roc_auc_score(y, b), "dAUC": d0,
            "CI_low": np.percentile(ds, 2.5), "CI_high": np.percentile(ds, 97.5), "p_boot": p}


def holm(ps):
    ps = np.asarray(ps); order = np.argsort(ps); adj = np.empty_like(ps); m = len(ps); run = 0
    for k, i in enumerate(order):
        run = max(run, (m - k) * ps[i]); adj[i] = min(1.0, run)
    return adj


def main():
    u = load_unified().set_index("image_id")
    rng = np.random.default_rng(2026)
    rows = []
    ref_bag = {"image": per_image(lambda s: bag_test("image", s)),
               "image_vruh": per_image(lambda s: bag_test("image_vruh", s))}
    ref_primary = {"image": per_image(lambda s: test_prob("image", s)),
                   "image_vruh": per_image(lambda s: test_prob("image_vruh", s))}
    for inner in ["image", "image_vruh"]:
        tag = "" if inner == "image" else "_image_vruh"
        for v in ["E0_C4", "E0_C6", "E2_C6", "E4_fixed", "E5_signs"]:
            for perm in [False, True]:
                name = f"pp_{v}" + ("_perm" if perm else "") + f"{tag}_is5"
                pa = per_image(lambda s, n=name: test_prob(n, s))
                fam = "control" if perm else "A"
                rows.append({"family": fam, "A": name, "B": f"bag {inner} (misma imagen)", "subgroup": "all",
                             **compare(pa, ref_bag[inner], u, rng)})
                if not perm:
                    if inner == "image":
                        for so in ["VRUH", "ISIC"]:
                            rows.append({"family": "A-subgrupo", "A": name, "B": f"bag {inner} (misma imagen)",
                                         "subgroup": so, **compare(pa, ref_bag[inner], u, rng, so)})
                    rows.append({"family": "contexto", "A": name, "B": f"{inner} (primario)", "subgroup": "all",
                                 **compare(pa, ref_primary[inner], u, rng)})
    # Familia B: ensembles de 5 semillas
    for img, cat, tag in [("image", "concat_meta", ""), ("image_vruh", "concat_meta_vruh", "_vruh")]:
        si = [img] + [f"x6_{img}_s{k}" for k in range(1, 5)]
        sc = [cat] + [f"x6_{'concat' if cat == 'concat_meta' else cat}_s{k}" for k in range(1, 5)]
        pi = per_image(lambda s: pd.concat([test_prob(e, s) for e in si], axis=1).mean(1))
        pc = per_image(lambda s: pd.concat([test_prob(e, s) for e in sc], axis=1).mean(1))
        rows.append({"family": "B", "A": f"ens5 {cat}", "B": f"ens5 {img}", "subgroup": "all", **compare(pc, pi, u, rng)})
    g = pd.DataFrame(rows)
    a = g.family == "A"
    g.loc[a, "p_Holm"] = holm(g.loc[a, "p_boot"].values)
    g.to_csv(RESULTS_DIR / "tables" / "test_final_exploratory.csv", index=False)
    pd.set_option("display.width", 250)
    g["ΔAUC (IC95)"] = g.apply(lambda r: f"{r.dAUC:+.3f} ({r.CI_low:+.3f} a {r.CI_high:+.3f})", axis=1)
    print(g[["family", "A", "B", "subgroup", "n_lesions", "AUC_A", "AUC_B", "ΔAUC (IC95)", "p_boot", "p_Holm"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
