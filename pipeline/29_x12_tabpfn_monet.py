#!/usr/bin/env python3
"""
Batería x12 (protocolo fijado el 2026-10-21): TabPFN y embeddings de MONET. Solo VALIDACIÓN; no lee el test.
Se ejecuta con el entorno aparte pipeline/.venv_t2 (PyTorch 2.5, TabPFN 2.0.9, transformers).

  python pipeline/29_x12_tabpfn_monet.py --extract   # embeddings MONET de todas las imágenes (GPU)
  python pipeline/29_x12_tabpfn_monet.py             # puerta de los 3 candidatos

Candidatos (mismo modelo en los dos brazos; train de la partición -> validación interna, repetición 0):
  tabpfn_fusion: TabPFN [logit imagen (cross-fitting), fuente] vs + [edad, sexo, localización, log-diámetro]
  monet_lr:      LR (C por CV agrupada) [PCA64 de MONET] vs + metadatos (MetaEncoder)
  monet_tabpfn:  TabPFN [PCA32 de MONET] vs + [edad, sexo, localización, log-diámetro]
Reglas 3 y 5; candidato final = mayor ΔAUC media entre los que pasan (results/tables/x12_decision.json).
"""
import argparse
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, SITES, MetaEncoder, load_split, load_unified, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("m1", REV / "14_crossfit_fusion.py")
m1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m1)
EMB = REV / "data" / "monet_emb.npz"
GATE_MEAN, GATE_POS_FRAC = 0.005, 2 / 3
SPLITS = [f"cv_r0_f{k}" for k in range(5)]
META4 = ["age", "sex", "site", "diam"]


def extract():
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor
    df = load_unified()
    model = CLIPModel.from_pretrained("suinleelab/monet").cuda().eval()
    proc = CLIPProcessor.from_pretrained("suinleelab/monet")
    embs = []
    for i in range(0, len(df), 64):
        ims = [Image.open(REV / p).convert("RGB") for p in df.cache_path.iloc[i:i + 64]]
        with torch.no_grad():
            x = proc(images=ims, return_tensors="pt")["pixel_values"].cuda()
            e = model.get_image_features(pixel_values=x)
        embs.append(torch.nn.functional.normalize(e, dim=1).cpu().numpy())
        print(f"{i + len(ims)}/{len(df)}", flush=True)
    np.savez(EMB, image_id=df.image_id.values.astype(str), emb=np.concatenate(embs))


def raw_meta(d, tr):
    """[edad, sexo, localización, log-diámetro] con la política de MetaEncoder: en ISIC sin diámetro y con edad,
    sexo y localización imputados (mediana/moda de train, sin indicador); en VRUH los faltantes quedan NaN."""
    d = MetaEncoder._isic_policy(d)
    X = pd.DataFrame({"age": d.age.astype(float).values,
                      "sex": d.sex.map({"MALE": 1.0, "FEMALE": 0.0}).values,
                      "site": d.site.map({s: float(k) for k, s in enumerate(SITES)}).values,
                      "logdiam": np.log1p(d.diam_long.astype(float)).values}, index=d.index)
    trm = raw_meta_stats(tr)
    isic = (d.source == "ISIC").values
    for c, v in trm.items():
        X.loc[isic & X[c].isna().values, c] = v
    return X


def raw_meta_stats(tr):
    return {"age": tr.age.astype(float).median(), "sex": float(tr.sex.map({"MALE": 1.0, "FEMALE": 0.0}).mode()[0]),
            "site": float(tr.site.map({s: float(k) for k, s in enumerate(SITES)}).mode()[0])}


def tabpfn(Xtr, ytr, Xva, cat_idx):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device="cuda", random_state=0, categorical_features_indices=cat_idx or None)
    return clf.fit(Xtr, ytr).predict_proba(Xva)[:, 1]


def passes(d):
    d = np.asarray(d, float)
    return bool(np.nanmean(d) >= GATE_MEAN and (d > 0).sum() >= np.ceil(GATE_POS_FRAC * len(d)))


def gate():
    uni = load_unified().set_index("image_id", drop=False)
    z = np.load(EMB, allow_pickle=True)
    emb = pd.DataFrame(z["emb"], index=z["image_id"])
    res = {c: [] for c in ("tabpfn_fusion", "monet_lr", "monet_tabpfn")}
    for s in SPLITS:
        sp = load_split(s)
        tr, va = uni.loc[sp["train"]], uni.loc[sp["val"]]
        ytr, yva = tr.label.values, va.label.values
        Mtr, Mva = raw_meta(tr, tr), raw_meta(va, tr)
        isic = lambda d: (d.source == "ISIC").astype(float).values[:, None]
        preds = {}
        # 1) TabPFN sobre el logit de imagen de cross-fitting
        oof, bag, _ = m1.load_inner("image", s, n_seeds=3)
        lt, lv = oof.loc[tr.index].values[:, None], bag.loc[va.index].values[:, None]
        preds["tabpfn_fusion"] = (tabpfn(np.hstack([lt, isic(tr)]), ytr, np.hstack([lv, isic(va)]), []),
                                  tabpfn(np.hstack([lt, isic(tr), Mtr.values]), ytr,
                                         np.hstack([lv, isic(va), Mva.values]), [4]))
        # 2) y 3) embeddings de MONET
        Etr, Eva = emb.loc[tr.index].values, emb.loc[va.index].values
        p64 = PCA(64, random_state=0).fit(Etr)
        sc = StandardScaler().fit(p64.transform(Etr))
        Ptr, Pva = sc.transform(p64.transform(Etr)), sc.transform(p64.transform(Eva))
        enc = MetaEncoder(META4, "cont", "cont", "indicator").fit(tr)
        Gtr, Gva = enc.transform(tr), enc.transform(va)
        lr_a = m1.lrcv(tr.patient_id.values).fit(Ptr, ytr)
        lr_b = m1.lrcv(tr.patient_id.values).fit(np.hstack([Ptr, Gtr]), ytr)
        preds["monet_lr"] = (lr_a.predict_proba(Pva)[:, 1], lr_b.predict_proba(np.hstack([Pva, Gva]))[:, 1])
        q = PCA(32, random_state=0).fit(Etr)
        Qtr, Qva = q.transform(Etr), q.transform(Eva)
        preds["monet_tabpfn"] = (tabpfn(Qtr, ytr, Qva, []),
                                 tabpfn(np.hstack([Qtr, Mtr.values]), ytr, np.hstack([Qva, Mva.values]), [34]))
        for c, (pa, pb) in preds.items():
            r = {"split": s, "auc_a": roc_auc_score(yva, pa), "auc_b": roc_auc_score(yva, pb)}
            for so in ("VRUH", "ISIC"):
                k = (va.source == so).values
                r[f"d_{so}"] = roc_auc_score(yva[k], pb[k]) - roc_auc_score(yva[k], pa[k])
            les = va.lesion_id.values
            la = pd.Series(pa, index=les).groupby(level=0).mean()
            lb = pd.Series(pb, index=les).groupby(level=0).mean()
            yl = pd.Series(yva, index=les).groupby(level=0).first()
            r["d_lesion"] = roc_auc_score(yl.loc[lb.index], lb) - roc_auc_score(yl.loc[la.index], la)
            r["d_all"] = r["auc_b"] - r["auc_a"]
            res[c].append(r)
            print(f"{s} {c:13s} AUC sin {r['auc_a']:.4f} con {r['auc_b']:.4f} Δ {r['d_all']:+.4f} "
                  f"VRUH {r['d_VRUH']:+.4f}", flush=True)

    rows, log_rows = [], []
    print()
    for c, rs in res.items():
        d = pd.DataFrame(rs)
        r3, r5 = passes(d.d_all), passes(d.d_VRUH)
        rows.append({"candidate": c, "auc_val_without": d.auc_a.mean(), "auc_val_with": d.auc_b.mean(),
                     "val_delta_mean": d.d_all.mean(), "val_delta_n_positive": int((d.d_all > 0).sum()),
                     "val_delta_VRUH_mean": d.d_VRUH.mean(), "val_delta_VRUH_n_positive": int((d.d_VRUH > 0).sum()),
                     "val_delta_ISIC_mean": d.d_ISIC.mean(), "val_delta_lesion_mean": d.d_lesion.mean(),
                     "rule3": r3, "rule5": r5, "passes": r3 and r5})
        print(f"{c:13s} AUC val sin {d.auc_a.mean():.4f} con {d.auc_b.mean():.4f} | ΔAUC {d.d_all.mean():+.4f} "
              f"({(d.d_all > 0).sum()}/5) | VRUH {d.d_VRUH.mean():+.4f} ({(d.d_VRUH > 0).sum()}/5) | "
              f"ISIC {d.d_ISIC.mean():+.4f} | lesión {d.d_lesion.mean():+.4f} -> {'PASA' if r3 and r5 else 'no pasa'}")
        log_rows.append({"date": str(date.today()), "step": "x12", "variant": f"x12_{c}", "meta_model": c,
                         "n_splits": len(d), "val_auc_img_mean": d.auc_a.mean(), "val_auc_stack_mean": d.auc_b.mean(),
                         "val_delta_mean": d.d_all.mean(), "val_delta_median": d.d_all.median(),
                         "val_delta_n_positive": int((d.d_all > 0).sum()), "passes_gate": bool(r3 and r5),
                         "role": "candidate", "val_delta_VRUH_mean": d.d_VRUH.mean(),
                         "val_delta_VRUH_n_positive": int((d.d_VRUH > 0).sum()), "val_delta_ISIC_mean": d.d_ISIC.mean(),
                         "val_delta_ISIC_n_positive": int((d.d_ISIC > 0).sum())})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_x12.csv", index=False)
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)
    ok = R[R.passes]
    final = ok.loc[ok.val_delta_mean.idxmax()] if len(ok) else None
    print("\n" + (f"Candidato final: {final.candidate} (ΔAUC val {final.val_delta_mean:+.4f})" if final is not None
                  else "Ningún candidato pasa la puerta: la batería termina aquí."))
    write_json({"date": str(date.today()), "passes": final is not None,
                "final_candidate": None if final is None else final.candidate,
                "val_delta_mean": None if final is None else float(final.val_delta_mean),
                "candidates": R.to_dict("records")}, RESULTS_DIR / "tables" / "x12_decision.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    extract() if ap.parse_args().extract else gate()
