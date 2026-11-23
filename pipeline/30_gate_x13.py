#!/usr/bin/env python3
"""
Puerta en VALIDACIÓN de la batería x13 (métodos que dan más peso a los metadatos; protocolo fijado el 2026-10-21). No lee el test.

Candidatos (6), ensemble de 3 semillas, repetición 0:
  ogm, gblend, staged, contrast: x13_<m>_concat frente a x11_base_image (misma receta);
  breslow: x13_breslow_concat frente a x13_breslow_image;
  mil: atención multiinstancia por lesión sobre twostage_emb (30 épocas fijas), con frente a sin metadatos de la
       lesión; ΔAUC por lesión.
ΔAUC por partición, emparejada (red: por imagen). Regla 3: media >= +0,005 y positiva en >= 2/3 de las particiones;
regla 5: el mismo criterio dentro de VRUH. Candidato final: el de mayor ΔAUC entre los que pasan
(results/tables/x13_decision.json). Como contexto se da también x13_<m>_concat frente a x11_base_concat.

  python pipeline/30_gate_x13.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, MetaEncoder, load_unified, write_json  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
GATE_MEAN, GATE_POS_FRAC = 0.005, 2 / 3
SPLITS = [f"cv_r0_f{k}" for k in range(5)]
META4 = ["age", "sex", "site", "diam"]
NET = {"ogm": ("x13_ogm_concat", "x11_base_image"), "gblend": ("x13_gblend_concat", "x11_base_image"),
       "staged": ("x13_staged_concat", "x11_base_image"), "contrast": ("x13_contrast_concat", "x11_base_image"),
       "breslow": ("x13_breslow_concat", "x13_breslow_image")}


def seeds(e):
    return [e, f"{e}_s1", f"{e}_s2"]


def val(exp, split):
    f = EXP_DIR / exp / split / "preds.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f, dtype={"image_id": str})
    return d[d.role == "val"].set_index("image_id")


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def passes(d):
    d = np.asarray(d, float)
    return bool(np.nanmean(d) >= GATE_MEAN and (d > 0).sum() >= np.ceil(GATE_POS_FRAC * len(d)))


def ens(exps, s):
    vs = [val(e, s) for e in exps]
    if any(v is None for v in vs):
        return None
    idx = vs[0].index
    return vs[0].label.loc[idx], pd.concat([v.prob.loc[idx] for v in vs], axis=1).mean(1)


def mil_split(s, uni, seed):
    """Atención MIL (Ilse 2018) por lesión: devuelve (lesiones de val, y, p sin metadatos, p con metadatos)."""
    import torch
    import torch.nn as nn
    z = np.load(EXP_DIR / "twostage_emb" / s / "emb.npz", allow_pickle=False)
    tr_ids, va_ids = z["id_train"].astype(str), z["id_val"].astype(str)
    enc = MetaEncoder(META4, "cont", "cont", "indicator").fit(uni.loc[tr_ids])

    def bags(ids, H):
        d = uni.loc[ids]
        M = enc.transform(d)
        les = d.lesion_id.values
        out = []
        for l in pd.unique(les):
            k = np.where(les == l)[0]
            out.append((l, H[k], M[k[0]], int(d.label.values[k[0]])))
        return out

    btr, bva = bags(tr_ids, z["H_train"]), bags(va_ids, z["H_val"])

    def pad(bs):
        n = max(len(b[1]) for b in bs)
        X = np.zeros((len(bs), n, bs[0][1].shape[1]), np.float32)
        mk = np.zeros((len(bs), n), bool)
        for i, b in enumerate(bs):
            X[i, :len(b[1])], mk[i, :len(b[1])] = b[1], True
        return (torch.tensor(X), torch.tensor(mk), torch.tensor(np.stack([b[2] for b in bs])).float(),
                torch.tensor([b[3] for b in bs]))

    Xtr, Mtr, Gtr, ytr = pad(btr)
    Xva, Mva, Gva, yva = pad(bva)

    class MIL(nn.Module):
        def __init__(self, d, md):
            super().__init__()
            self.att = nn.Sequential(nn.Linear(d, 64), nn.Tanh(), nn.Linear(64, 1))
            self.cls = nn.Sequential(nn.Linear(d + md, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 2))
            self.md = md

        def forward(self, X, mk, G):
            a = self.att(X).squeeze(-1).masked_fill(~mk, -1e9).softmax(1)
            h = (a.unsqueeze(-1) * X).sum(1)
            return self.cls(torch.cat([h, G], 1) if self.md else h)

    out = {}
    cnt = np.bincount(ytr.numpy(), minlength=2)
    w = torch.tensor(len(ytr) / (2 * cnt), dtype=torch.float32)
    for arm, md in (("without", 0), ("with", Gtr.shape[1])):
        torch.manual_seed(seed)
        m = MIL(Xtr.shape[2], md)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
        g = torch.Generator().manual_seed(seed)
        for _ in range(30):
            m.train()
            for b in torch.randperm(len(ytr), generator=g).split(32):
                opt.zero_grad()
                nn.functional.cross_entropy(m(Xtr[b], Mtr[b], Gtr[b]), ytr[b], weight=w).backward()
                opt.step()
        m.eval()
        with torch.no_grad():
            out[arm] = m(Xva, Mva, Gva).softmax(1)[:, 1].numpy()
    return [b[0] for b in bva], yva.numpy(), out["without"], out["with"]


def main():
    uni = load_unified().set_index("image_id", drop=False)
    src = uni.source
    les_src = uni.groupby("lesion_id").source.first()
    rows, log_rows, missing = [], [], []

    def record(name, d, extra=None):
        d = pd.DataFrame(d, index=SPLITS)
        r3, r5 = passes(d["all"]), passes(d["VRUH"])
        rows.append({"candidate": name, "val_delta_mean": d["all"].mean(), "val_delta_n_positive": int((d["all"] > 0).sum()),
                     "val_delta_VRUH_mean": d["VRUH"].mean(), "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()),
                     "val_delta_ISIC_mean": d["ISIC"].mean(), "rule3": r3, "rule5": r5, "passes": r3 and r5,
                     **(extra or {})})
        print(f"{name:9s} ΔAUC val {d['all'].mean():+.4f} ({(d['all'] > 0).sum()}/5) | VRUH {d['VRUH'].mean():+.4f} "
              f"({(d['VRUH'] > 0).sum()}/5) | ISIC {d['ISIC'].mean():+.4f}"
              + "".join(f" | {k} {v:+.4f}" for k, v in (extra or {}).items())
              + f" -> {'PASA' if r3 and r5 else 'no pasa'}")
        log_rows.append({"date": str(date.today()), "step": "x13", "variant": f"x13_{name}", "meta_model": name,
                         "n_splits": len(d), "val_delta_mean": d["all"].mean(), "val_delta_median": d["all"].median(),
                         "val_delta_n_positive": int((d["all"] > 0).sum()), "passes_gate": bool(r3 and r5),
                         "role": "candidate", "val_delta_VRUH_mean": d["VRUH"].mean(),
                         "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()), "val_delta_ISIC_mean": d["ISIC"].mean(),
                         "val_delta_ISIC_n_positive": int((d["ISIC"] > 0).sum())})

    for name, (cat, img) in NET.items():
        d, ctx = [], []
        for s in SPLITS:
            a, b, c = ens(seeds(cat), s), ens(seeds(img), s), ens(seeds("x11_base_concat"), s)
            if a is None or b is None or c is None:
                missing.append((name, s))
                continue
            (lab, pc), (_, pi), (_, pb) = a, b, c
            idx = lab.index
            r = {"all": auc(lab, pc) - auc(lab, pi.loc[idx])}
            for so in ("VRUH", "ISIC"):
                k = idx[(src.loc[idx] == so).values]
                r[so] = auc(lab.loc[k], pc.loc[k]) - auc(lab.loc[k], pi.loc[k])
            d.append(r)
            ctx.append(auc(lab, pc) - auc(lab, pb.loc[idx]))
        if missing:
            sys.exit(f"Faltan entrenamientos: {missing[:3]}")
        record(name, d, {"vs_base_concat": float(np.mean(ctx))})

    d = []
    for s in SPLITS:
        P = {}
        for sd in range(3):
            les, y, pa, pb = mil_split(s, uni, 1000 + sd)
            P.setdefault("a", []).append(pa)
            P.setdefault("b", []).append(pb)
        pa, pb = np.mean(P["a"], 0), np.mean(P["b"], 0)
        so = les_src.loc[les].values
        r = {"all": auc(y, pb) - auc(y, pa)}
        for k in ("VRUH", "ISIC"):
            mk = so == k
            r[k] = auc(y[mk], pb[mk]) - auc(y[mk], pa[mk])
        d.append(r)
        print(f"  MIL {s}: AUC sin {auc(y, pa):.4f} con {auc(y, pb):.4f}", flush=True)
    record("mil", d)

    R = pd.DataFrame(rows)
    R.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_x13.csv", index=False)
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
                "candidates": R.to_dict("records")}, RESULTS_DIR / "tables" / "x13_decision.json")


if __name__ == "__main__":
    main()
