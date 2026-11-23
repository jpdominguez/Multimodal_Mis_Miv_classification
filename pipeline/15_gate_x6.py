#!/usr/bin/env python3
"""
Puerta en VALIDACIÓN de la batería x6 (protocolo fijado antes de entrenar). No lee el test.

  1) Modelo único: ΔAUC = AUC(variante) − media de AUC de las semillas de imagen disponibles
     (image, x6_image_s1, x6_image_s2) en la misma partición. Promediar la referencia reduce el ruido de semilla
     sin dar ventaja de ensemble a nadie (se compara un modelo con el modelo de imagen "esperado").
  2) Ensemble contra ensemble, con el mismo nº de semillas: media de probabilidades de
     {concat_meta, x6_concat_s1, x6_concat_s2} frente a {image, x6_image_s1, x6_image_s2}.
  Regla 3: media >= +0,005 y positiva en >= 2/3 de las particiones. Regla 5: dentro de VRUH.

  python pipeline/15_gate_x6.py [--exps x6_film x6_add_lin ...]
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
GATE_MEAN, GATE_POS_FRAC = 0.005, 10 / 15
SEEDS = {"image": ["image"] + [f"x6_image_s{k}" for k in range(1, 5)],
         "concat_meta": ["concat_meta"] + [f"x6_concat_s{k}" for k in range(1, 5)],
         "image_vruh": ["image_vruh"] + [f"x6_image_vruh_s{k}" for k in range(1, 5)],
         "concat_meta_vruh": ["concat_meta_vruh"] + [f"x6_concat_meta_vruh_s{k}" for k in range(1, 5)]}


def val(exp, split):
    for base in (EXP_DIR, RUNS_DIR):
        f = base / exp / split / "preds.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"image_id": str})
            return d[d.role == "val"].set_index("image_id")
    return None


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def aucs_by_src(v, idx, src):
    out = {"all": auc(v.label.loc[idx], v.prob.loc[idx])}
    for s in ["VRUH", "ISIC"]:
        k = idx[(src.loc[idx] == s).values]
        out[s] = auc(v.label.loc[k], v.prob.loc[k]) if len(k) > 10 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", nargs="+", default=None)
    ap.add_argument("--ref", default="image", help="'image' o 'image_vruh' (variantes *_vruh)")
    ap.add_argument("--tag", default="x6")
    ap.add_argument("--family", nargs="*", default=[],
                    help="familias de semillas <nombre>, <nombre>_s1, ... -> ensemble de K frente a K semillas de imagen")
    args = ap.parse_args()
    vr = args.ref == "image_vruh"
    exps = args.exps or sorted(p.name for p in EXP_DIR.glob("x6_*")
                               if not p.name.startswith("x6_image") and (p.name.endswith("_vruh") or "_vruh_s" in p.name) == vr)
    img_seeds = SEEDS[args.ref]
    cat_seeds = SEEDS["concat_meta_vruh" if vr else "concat_meta"]
    uni = load_unified().set_index("image_id")
    src, lesion = uni.source, uni.lesion_id

    def les_auc(lab, p, idx):
        q = pd.Series(p.loc[idx].values, index=lesion.loc[idx].values).groupby(level=0).mean()
        yl = pd.Series(lab.loc[idx].values, index=lesion.loc[idx].values).groupby(level=0).first()
        return auc(yl.loc[q.index], q)
    splits = sorted(p.name for p in (RUNS_DIR / args.ref).glob("cv_*"))
    rows = []
    for s in splits:
        refs = {e: val(e, s) for e in img_seeds}
        refs = {e: v for e, v in refs.items() if v is not None}
        idx = refs[args.ref].index
        ref_auc = {k: np.nanmean([aucs_by_src(v, idx, src)[k] for v in refs.values()]) for k in ["all", "VRUH", "ISIC"]}
        for e in exps:
            v = val(e, s)
            if v is None:
                continue
            a = aucs_by_src(v, idx, src)
            rl = np.nanmean([les_auc(vv.label, vv.prob, idx) for vv in refs.values()])
            rows.append({"variant": e, "split": s, "n_img_seeds": len(refs), "auc_val": a["all"],
                         "auc_val_ref": ref_auc["all"], "delta": a["all"] - ref_auc["all"],
                         "delta_lesion": les_auc(v.label, v.prob, idx) - rl,
                         "delta_VRUH": a["VRUH"] - ref_auc["VRUH"], "delta_ISIC": a["ISIC"] - ref_auc["ISIC"]})
        # ensemble contra ensemble, con el MISMO nº de semillas en los dos brazos (3 y, si están, 5)
        rs = list(refs.values())
        fams = {"concat": [v for v in (val(e, s) for e in cat_seeds) if v is not None]}
        for fam in args.family:
            fams[fam] = [v for v in (val(e, s) for e in [fam] + [f"{fam}_s{k}" for k in range(1, 5)]) if v is not None]
        for fam, cs in fams.items():
          for K in (3, 5):
            if len(rs) >= K and len(cs) >= K:
                pi = pd.concat([v.prob.loc[idx] for v in rs[:K]], axis=1).mean(1)
                pc = pd.concat([v.prob.loc[idx] for v in cs[:K]], axis=1).mean(1)
                lab = refs[args.ref].label.loc[idx]
                r = {"variant": f"ens{K}_{fam}_vs_ens{K}_image" + ("_vruh" if vr and not fam.endswith("vruh") else ""),
                     "split": s, "n_img_seeds": K, "auc_val": auc(lab, pc), "auc_val_ref": auc(lab, pi)}
                r["delta"] = r["auc_val"] - r["auc_val_ref"]
                r["delta_lesion"] = les_auc(lab, pc, idx) - les_auc(lab, pi, idx)
                for so in ["VRUH", "ISIC"]:
                    k = idx[(src.loc[idx] == so).values]
                    r[f"delta_{so}"] = auc(lab.loc[k], pc.loc[k]) - auc(lab.loc[k], pi.loc[k]) if len(k) > 10 else np.nan
                rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(RESULTS_DIR / "tables" / f"exploratory_gate_{args.tag}_{args.ref}.csv", index=False)
    log_rows = []
    for e, x in g.groupby("variant"):
        n = len(x)
        passed = x.delta.mean() >= GATE_MEAN and (x.delta > 0).sum() >= np.ceil(GATE_POS_FRAC * n)
        log_rows.append({"date": str(date.today()), "step": args.tag, "variant": e if args.ref == "image" else f"{e}__ref_{args.ref}",
                         "meta_model": "end-to-end", "n_splits": n, "val_auc_img_mean": x.auc_val_ref.mean(),
                         "val_auc_stack_mean": x.auc_val.mean(), "val_delta_mean": x.delta.mean(),
                         "val_delta_median": x.delta.median(), "val_delta_n_positive": int((x.delta > 0).sum()),
                         "passes_gate": bool(passed),
                         "role": "control" if ("perm" in e or e.startswith(("x6_concat_s", "x6_image", "x6_concat_meta_vruh_s"))) else "candidate",
                         "val_delta_VRUH_mean": x.delta_VRUH.mean(), "val_delta_VRUH_n_positive": int((x.delta_VRUH > 0).sum()),
                         "val_delta_ISIC_mean": x.delta_ISIC.mean(), "val_delta_ISIC_n_positive": int((x.delta_ISIC > 0).sum())})
        print(f"{e:28s} n={n:2d} (semillas ref {x.n_img_seeds.min()}-{x.n_img_seeds.max()}) ΔAUC val {x.delta.mean():+.4f} "
              f"(med {x.delta.median():+.4f}; {(x.delta > 0).sum()}/{n}) -> {'PASA' if passed else 'no pasa'} | "
              f"VRUH {x.delta_VRUH.mean():+.4f} ({(x.delta_VRUH > 0).sum()}/{n}) | ISIC {x.delta_ISIC.mean():+.4f} | "
              f"lesión {x.delta_lesion.mean():+.4f} ({(x.delta_lesion > 0).sum()}/{n})")
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)


if __name__ == "__main__":
    main()
