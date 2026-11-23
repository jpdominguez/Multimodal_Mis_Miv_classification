#!/usr/bin/env python3
"""
Paso exploratorio 2 - Backbone de imagen congelado + cabeza de fusión.

  python pipeline/10_train_twostage.py --split cv_r0_f0          # entrena (GPU)
  python pipeline/10_train_twostage.py --summarize               # puerta en validación (CPU)

Fase A: modelo 'image' con la MISMA receta que 05_train.py (focal, Mixup, sampler balanceado, AdamW,
        early stopping por AUC en la validación interna) -> referencia emparejada 'twostage_imgref'.
Fase B: backbone congelado. Se extrae h = penúltima capa del clasificador (128-d, modo eval, sin aumentos)
        para train/val/test, y se entrenan cabezas sobre h (+ metadatos codificados con train):
          T0  lineal sobre h                       (control: el reajuste sin metadatos)
          T1  lineal sobre [h, metadatos]
          T2  MLP (64, dropout 0.3) sobre [h, metadatos]
          T3  residual: logit_img (fijo) + lineal(metadatos)
        Todas con CE ponderada, early stopping por AUC en validación.
El test se predice una vez y se guarda en runs_exploratory/ SIN evaluarlo (se abre solo en el paso 5).
Puerta (regla 3): ΔAUC en val de T1-T3 frente a imgref; se informa también frente a T0 y dentro de cada fuente.
"""
import argparse
import copy
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REV, RESULTS_DIR, SPLITS_CSV, MetaEncoder, aux_targets, build_torch_components,  # noqa: E402
                    load_cached_images, load_split, load_unified, set_seed, stable_seed, write_json)

EXP_DIR = REV / "runs_exploratory"
META4 = ["age", "sex", "site", "diam"]
HEADS = ["T0", "T1", "T2", "T3"]
GATE_MEAN, GATE_POS = 0.005, 10


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def save_preds(name, split, parts, y, p_val, p_test, p_perm=None):
    out = EXP_DIR / name / split
    out.mkdir(parents=True, exist_ok=True)
    pd.concat([
        pd.DataFrame({"image_id": parts["val"].image_id.values, "role": "val", "label": y["val"], "prob": p_val,
                      "prob_meta_permuted": np.nan}),
        pd.DataFrame({"image_id": parts["test"].image_id.values, "role": "test", "label": y["test"], "prob": p_test,
                      "prob_meta_permuted": np.nan if p_perm is None else p_perm})]).to_csv(out / "preds.csv", index=False)


def train(args):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler

    split = args.split
    if (EXP_DIR / "twostage_T3" / split / "preds.csv").exists() and not args.overwrite:
        print(f"Ya existe {split} -> se omite")
        return
    seed = stable_seed("twostage", split)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        sys.exit("ERROR: CUDA no disponible en este nodo")
    T = build_torch_components()
    df = load_unified().set_index("image_id", drop=False)
    parts = {r: df.loc[ids] for r, ids in load_split(split).items()}
    enc = MetaEncoder(META4, missing="indicator").fit(parts["train"])
    t0 = time.time()
    imgs = {r: load_cached_images(d) for r, d in parts.items()}
    meta = {r: enc.transform(d) for r, d in parts.items()}
    y = {r: d.label.values.astype(np.int64) for r, d in parts.items()}
    empty = {r: np.zeros((len(d), 0), np.float32) for r, d in parts.items()}
    aux = {r: aux_targets(d, enc) for r, d in parts.items()}

    # ---------------- Fase A: modelo de imagen (receta de 05_train.py) ----------------
    ds_train = T["CachedImageDataset"](imgs["train"], empty["train"], y["train"], aux["train"], train=True)
    counts = np.bincount(y["train"], minlength=2)
    sampler = WeightedRandomSampler(torch.as_tensor(1.0 / counts[y["train"]], dtype=torch.double),
                                    num_samples=len(y["train"]), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    dl_train = DataLoader(ds_train, batch_size=16, sampler=sampler, num_workers=args.workers, pin_memory=True,
                          drop_last=True)

    def loader(role):
        ds = T["CachedImageDataset"](imgs[role], empty[role], y[role], aux[role], train=False)
        return DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = T["MultimodalNet"]("image", 0).to(device)
    class_w = torch.tensor(len(y["train"]) / (2 * counts), dtype=torch.float32, device=device)
    crit = T["FocalLoss"](alpha=class_w, gamma=2.0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    rng = np.random.default_rng(seed)

    @torch.no_grad()
    def embed(role):
        """Devuelve (h 128-d, logit de la clase 1 - logit clase 0) en modo eval."""
        model.eval()
        hs, ls = [], []
        for img, *_ in loader(role):
            with torch.autocast("cuda"):
                f = torch.nn.functional.relu(model.features(img.to(device)))
                pooled = torch.nn.functional.adaptive_avg_pool2d(f, 1).flatten(1)
                h = model.classifier[:4](model.img_proj(pooled))
                lg = model.classifier[4](h)
            hs.append(h.float().cpu().numpy()); ls.append((lg[:, 1] - lg[:, 0]).float().cpu().numpy())
        return np.concatenate(hs), np.concatenate(ls)

    best = {"auc": -1, "epoch": 0, "state": None}
    bad = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        for bi, (img, _, lab, *_r) in enumerate(dl_train):
            img, lab = img.to(device, non_blocking=True), lab.to(device)
            mix = rng.random() > 0.5
            with torch.autocast("cuda"):
                if mix:
                    lam = float(rng.beta(0.2, 0.2))
                    idx = torch.randperm(img.size(0), device=device)
                    logits = model(lam * img + (1 - lam) * img[idx])
                    loss = lam * crit(logits.float(), lab) + (1 - lam) * crit(logits.float(), lab[idx])
                else:
                    loss = crit(model(img).float(), lab)
            scaler.scale(loss / 2).backward()
            if (bi + 1) % 2 == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
        sched.step()
        _, lv = embed("val")
        a = auc(y["val"], lv)
        print(f"A ep {epoch:3d} val_auc {a:.4f}", flush=True)
        if a > best["auc"]:
            best = {"auc": a, "epoch": epoch, "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
            bad = 0
        else:
            bad += 1
            if bad >= 10:
                break
    model.load_state_dict(best["state"])
    H, L = {}, {}
    for r in ["train", "val", "test"]:
        H[r], L[r] = embed(r)
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    save_preds("twostage_imgref", split, parts, y, sig(L["val"]), sig(L["test"]))
    print(f"Fase A: best_epoch={best['epoch']} val_auc={best['auc']:.4f} ({(time.time() - t0) / 60:.1f} min)")

    out = EXP_DIR / "twostage_emb" / split
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "emb.npz", **{f"H_{r}": H[r] for r in H}, **{f"L_{r}": L[r] for r in L},
                        **{f"M_{r}": meta[r] for r in meta}, **{f"y_{r}": y[r] for r in y},
                        **{f"id_{r}": parts[r].image_id.values.astype(str) for r in parts},
                        **{f"pid_{r}": parts[r].patient_id.values.astype(str) for r in parts})
    write_json({"split": split, "seed": seed, "phaseA_best_epoch": best["epoch"], "phaseA_val_auc": best["auc"],
                "encoder": enc.to_json(), "minutes": (time.time() - t0) / 60,
                "device": torch.cuda.get_device_name(0), "date": str(date.today())}, out / "run.json")
    print(f"FIN twostage {split} ({(time.time() - t0) / 60:.1f} min)")


def heads(split, seed):
    """Fase B en CPU a partir de los embeddings guardados.

    Para la PUERTA, cada cabeza se entrena dos veces con early stopping en una mitad de la validación
    (agrupada por paciente) y predice la otra mitad -> AUC sin sesgo de selección de época.
    Para el TEST, se entrena con early stopping en toda la validación (se guarda sin evaluar).
    Controles: T*_perm = misma cabeza con los metadatos permutados entre imágenes en train, val y test.
    """
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    torch.set_num_threads(4)
    z = np.load(EXP_DIR / "twostage_emb" / split / "emb.npz", allow_pickle=False)
    R = ["train", "val", "test"]
    H, L, M, y = ({r: z[f"{k}_{r}"] for r in R} for k in ["H", "L", "M", "y"])
    ids = {r: z[f"id_{r}"] for r in R}
    # Metadatos recodificados con el MetaEncoder ACTUAL: emb.npz guarda la codificación del momento del
    # entrenamiento (anterior a la corrección de 2026-10-09). Los embeddings de imagen no dependen de ella.
    dfu = load_unified().set_index("image_id")
    enc = MetaEncoder(META4, missing="indicator").fit(dfu.loc[ids["train"].astype(str)])
    M = {r: enc.transform(dfu.loc[ids[r].astype(str)]) for r in R}
    counts = np.bincount(y["train"], minlength=2)
    mu, sd = H["train"].mean(0), H["train"].std(0) + 1e-6
    Hz = {r: (H[r] - mu) / sd for r in R}
    rng = np.random.default_rng(seed)
    Mp = {r: M[r][rng.permutation(len(M[r]))] for r in R}
    zero = {r: np.zeros(len(y[r]), np.float32) for r in R}
    sig = lambda v: 1 / (1 + np.exp(-v))  # noqa: E731
    specs = {"T0": ("linear", Hz, zero),
             "T1": ("linear", {r: np.hstack([Hz[r], M[r]]) for r in R}, zero),
             "T2": ("mlp", {r: np.hstack([Hz[r], M[r]]) for r in R}, zero),
             "T3": ("residual", M, L),
             "T1_perm": ("linear", {r: np.hstack([Hz[r], Mp[r]]) for r in R}, zero),
             "T2_perm": ("mlp", {r: np.hstack([Hz[r], Mp[r]]) for r in R}, zero),
             "T3_perm": ("residual", Mp, L)}

    def fit(kind, X, off, stop_idx):
        torch.manual_seed(seed)
        d_in = X["train"].shape[1]
        if kind == "mlp":
            net = torch.nn.Sequential(torch.nn.Linear(d_in, 64), torch.nn.ReLU(), torch.nn.Dropout(0.3),
                                      torch.nn.Linear(64, 1)); lr = 1e-3
        else:
            net = torch.nn.Linear(d_in, 1); lr = 1e-2
            if kind == "residual":          # empieza exactamente en el modelo de imagen
                torch.nn.init.zeros_(net.weight); torch.nn.init.zeros_(net.bias)
        o = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-3)
        t = {r: torch.tensor(X[r], dtype=torch.float32) for r in R}
        of = {r: torch.tensor(off[r], dtype=torch.float32) for r in R}
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counts[0] / counts[1], dtype=torch.float32))
        yt = torch.tensor(y["train"], dtype=torch.float32)

        def pred(r):
            net.eval()
            with torch.no_grad():
                return (net(t[r]).squeeze(1) + of[r]).numpy()
        bst, bad, state = -1, 0, copy.deepcopy(net.state_dict())
        for ep in range(2000):
            net.train(); o.zero_grad()
            lossf(net(t["train"]).squeeze(1) + of["train"], yt).backward(); o.step()
            if ep % 5 == 4:
                a = auc(y["val"][stop_idx], pred("val")[stop_idx])
                if a > bst:
                    bst, bad, state = a, 0, copy.deepcopy(net.state_dict())
                else:
                    bad += 1
                    if bad >= 40:
                        break
        net.load_state_dict(state)
        return pred

    halves = list(StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed % (2 ** 31)).split(
        y["val"], y["val"], groups=z["pid_val"]))
    for name, (kind, X, off) in specs.items():
        cf = np.full(len(y["val"]), np.nan)
        for a_idx, b_idx in halves:                 # early stopping en una mitad, predicción en la otra
            cf[b_idx] = fit(kind, X, off, a_idx)("val")[b_idx]
        p_test = fit(kind, X, off, np.arange(len(y["val"])))("test")
        o = EXP_DIR / f"twostage_{name}" / split
        o.mkdir(parents=True, exist_ok=True)
        pd.concat([pd.DataFrame({"image_id": ids["val"], "role": "val", "label": y["val"], "prob": sig(cf),
                                 "prob_meta_permuted": np.nan}),
                   pd.DataFrame({"image_id": ids["test"], "role": "test", "label": y["test"], "prob": sig(p_test),
                                 "prob_meta_permuted": np.nan})]).to_csv(o / "preds.csv", index=False)


def summarize():
    df = load_unified().set_index("image_id")
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    rd = lambda n, s: pd.read_csv(EXP_DIR / n / s / "preds.csv", dtype={"image_id": str}).query("role == 'val'").set_index("image_id")  # noqa: E731
    rows = []
    for k, s in enumerate(splits):
        if not (EXP_DIR / "twostage_emb" / s / "emb.npz").exists():
            continue
        heads(s, seed=2000 + k)
        ref, t0 = rd("twostage_imgref", s), rd("twostage_T0", s)
        src = df.loc[ref.index].source.values
        for h in HEADS[1:] + [f"{x}_perm" for x in HEADS[1:]]:
            d = rd(f"twostage_{h}", s).loc[ref.index]
            r = {"variant": f"twostage_{h}", "split": s, "auc_val_imgref": auc(ref.label, ref.prob),
                 "auc_val_T0": auc(ref.label, t0.loc[ref.index].prob), "auc_val": auc(ref.label, d.prob)}
            r["delta"] = r["auc_val"] - r["auc_val_imgref"]
            r["delta_vs_T0"] = r["auc_val"] - r["auc_val_T0"]
            if not h.endswith("_perm"):
                pr = rd(f"twostage_{h}_perm", s).loc[ref.index]
                r["delta_vs_perm"] = r["auc_val"] - auc(ref.label, pr.prob)
            for so in ["VRUH", "ISIC"]:
                m = src == so
                r[f"delta_{so}"] = auc(ref.label[m], d.prob[m]) - auc(ref.label[m], ref.prob[m])
            rows.append(r)
    g = pd.DataFrame(rows)
    g.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_step2.csv", index=False)
    log_rows = []
    for v, x in g.groupby("variant"):
        is_ctrl = v.endswith("_perm")
        # Puerta: regla 3 frente a imgref Y, además (añadido antes de abrir test, más estricto), frente a su
        # control permutado con el mismo procedimiento.
        passed = (not is_ctrl and x.delta.mean() >= GATE_MEAN and (x.delta > 0).sum() >= GATE_POS
                  and x.delta_vs_perm.mean() >= GATE_MEAN and (x.delta_vs_perm > 0).sum() >= GATE_POS)
        log_rows.append({"date": str(date.today()), "step": "2", "variant": v, "meta_model": "end-to-end head",
                         "n_splits": len(x), "val_auc_img_mean": x.auc_val_imgref.mean(),
                         "val_auc_stack_mean": x.auc_val.mean(), "val_delta_mean": x.delta.mean(),
                         "val_delta_median": x.delta.median(), "val_delta_n_positive": int((x.delta > 0).sum()),
                         "passes_gate": bool(passed), "role": "control" if is_ctrl else "candidate",
                         "val_delta_vs_perm_mean": x.delta_vs_perm.mean() if not is_ctrl else np.nan,
                         "val_delta_vs_perm_n_positive": int((x.delta_vs_perm > 0).sum()) if not is_ctrl else np.nan,
                         "val_delta_vs_T0_mean": x.delta_vs_T0.mean(),
                         "val_delta_VRUH_mean": x.delta_VRUH.mean(), "val_delta_VRUH_n_positive": int((x.delta_VRUH > 0).sum()),
                         "val_delta_ISIC_mean": x.delta_ISIC.mean(), "val_delta_ISIC_n_positive": int((x.delta_ISIC > 0).sum())})
        print(f"{v:16s} n={len(x)} ΔAUC val vs imgref {x.delta.mean():+.4f} ({(x.delta > 0).sum()}/{len(x)}) "
              f"-> {'PASA' if passed else ('control' if is_ctrl else 'no pasa')} | vs T0 {x.delta_vs_T0.mean():+.4f} | "
              + (f"vs perm {x.delta_vs_perm.mean():+.4f} ({(x.delta_vs_perm > 0).sum()}/{len(x)}) | " if not is_ctrl else "") +
              f"VRUH {x.delta_VRUH.mean():+.4f} ({(x.delta_VRUH > 0).sum()}/{len(x)}) | "
              f"ISIC {x.delta_ISIC.mean():+.4f} ({(x.delta_ISIC > 0).sum()}/{len(x)})")
    print(f"T0 (reajuste sin metadatos) vs imgref: {(g.auc_val_T0 - g.auc_val_imgref).groupby(g.split).first().mean():+.4f}")
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    summarize() if a.summarize else train(a)
