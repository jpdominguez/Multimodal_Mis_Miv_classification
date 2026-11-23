#!/usr/bin/env python3
"""
Paso 5 - Entrena UN experimento en UNA partición y guarda predicciones.

  python pipeline/05_train.py --exp concat_meta --split cv_r0_f0

Diseño:
  * Codificador de metadatos y pesos de clase ajustados SOLO con el train del fold.
  * Early stopping y selección del checkpoint por AUC en la validación INTERNA.
  * El test del fold solo se predice una vez, con el checkpoint ya elegido.
  * Se guardan predicciones de val y test -> el umbral (Youden) se fija en val
    y se aplica a test en 06_analyze.py.
  * Para modelos con metadatos se guarda también la predicción en test con los
    metadatos PERMUTADOS entre pacientes (control de dependencia).

Salida: pipeline/runs/<exp>/<split>/{preds.csv, run.json}
"""
import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (META_MODELS, RUNS_DIR, MetaEncoder, aux_targets, build_torch_components, load_cached_images,  # noqa: E402
                    load_split, load_unified, set_seed, stable_seed, write_json)
from experiments import EXPERIMENTS  # noqa: E402


def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=sorted(EXPERIMENTS))
    ap.add_argument("--split", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--aux-weight", type=float, default=0.3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="prueba rápida: 2 épocas y subconjunto")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--out", default=str(RUNS_DIR))
    ap.add_argument("--inner", type=int, default=None,
                    help="M1 (cross-fitting): entrena en 4/5 del train externo (folds por paciente) y predice el 1/5 "
                         "restante (role=oof) + val/test externos. Salida en <out>/<exp>_inner/<split>_in<K>/")
    ap.add_argument("--inner-seed", type=int, default=0,
                    help="semilla adicional de los modelos internos (mismos folds); salida en <exp>_inner_s<S>/")
    return ap.parse_args()


def main():
    args = parse()
    import torch
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import DataLoader, WeightedRandomSampler

    cfg = EXPERIMENTS[args.exp]
    out_dir = Path(args.out) / args.exp / args.split
    if args.inner is not None:
        out_dir = Path(args.out) / (f"{args.exp}_inner" + (f"_s{args.inner_seed}" if args.inner_seed else "")) \
            / f"{args.split}_in{args.inner}"
    if (out_dir / "preds.csv").exists() and not args.overwrite and not args.smoke:
        print(f"Ya existe {out_dir}/preds.csv -> se omite")
        return
    seed = stable_seed(args.exp, args.split) if args.inner is None else (
        stable_seed(args.exp, args.split, args.inner) if not args.inner_seed
        else stable_seed(args.exp, args.split, args.inner, args.inner_seed))
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not (args.smoke or args.allow_cpu):
        # Evita entrenar en CPU sin darse cuenta (p. ej. nodo con el driver CUDA caído)
        sys.exit("ERROR: CUDA no disponible en este nodo; usa --allow-cpu para forzar CPU")
    T = build_torch_components()

    df = load_unified().set_index("image_id", drop=False)
    sp = load_split(args.split)
    parts = {r: df.loc[ids] for r, ids in sp.items()}
    if cfg["sources"]:
        parts = {r: d[d.source == cfg["sources"]] for r, d in parts.items()}
    outer_role = None
    if args.inner is not None:
        from sklearn.model_selection import StratifiedGroupKFold
        tr = parts["train"]
        folds = list(StratifiedGroupKFold(n_splits=5, shuffle=True,
                                          random_state=stable_seed("inner", args.split) % (2 ** 31))
                     .split(tr, tr.label, groups=tr.patient_id))
        a, b = folds[args.inner]
        outer_role = pd.concat([pd.Series("val", index=parts["val"].image_id.values),
                                pd.Series("test", index=parts["test"].image_id.values)])
        parts = {"train": tr.iloc[a], "val": tr.iloc[b], "test": pd.concat([parts["val"], parts["test"]])}
    if args.smoke:
        parts = {r: d.sample(min(len(d), 96 if r == "train" else 64), random_state=0) for r, d in parts.items()}
        args.epochs, args.patience = 2, 2

    enc = MetaEncoder(cfg["meta_vars"], cfg["age_enc"], cfg["diam_enc"], cfg["missing"]).fit(parts["train"])
    use_meta = cfg["model"] in META_MODELS and enc.dim > 0
    print(f"[{args.exp} | {args.split}] train={len(parts['train'])} val={len(parts['val'])} "
          f"test={len(parts['test'])} meta_dim={enc.dim} ({', '.join(enc.names)}) device={device}")

    t0 = time.time()
    imgs = {r: load_cached_images(d) for r, d in parts.items()}
    meta = {r: (enc.transform(d) if use_meta else np.zeros((len(d), 0), np.float32)) for r, d in parts.items()}
    aux = {r: aux_targets(d, enc) for r, d in parts.items()}
    y = {r: d.label.values.astype(np.int64) for r, d in parts.items()}
    # batería x11: clase de grosor (Mis / Miv < 1 mm / Miv >= 1 mm) y fuente como objetivos extra, solo en train
    # batería x13: log(1 + Breslow) como tercer objetivo (Mis = 0; NaN si falta)
    thick_w, adv_w = cfg.get("thick_weight", 0.0), cfg.get("adv_weight", 0.0)
    bres_w = cfg.get("breslow_weight", 0.0)
    kd_b = cfg.get("kd_beta", 0.0)
    if thick_w or adv_w or bres_w or kd_b:
        tr = parts["train"]
        lb = np.where(tr.class3.values == 0, 0.0, np.log1p(tr.breslow_mm.astype(float).values))
        teacher = np.full(len(tr), np.nan)
        if kd_b:
            # x15: profesor = LR en train sobre el logit de imagen fuera de fold (image_inner) [+ logit fuera de fold
            # de un modelo logístico de metadatos, 5 folds por paciente]
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import GroupKFold
            inner = Path(__file__).resolve().parent / "runs_exploratory" / "image_inner"
            oof = pd.concat([pd.read_csv(inner / f"{args.split}_in{k}" / "preds.csv", dtype={"image_id": str})
                             .query("role == 'oof'") for k in range(5)]).set_index("image_id").prob
            q = np.clip(oof.loc[tr.image_id.values].values, 1e-6, 1 - 1e-6)
            X = np.log(q / (1 - q))[:, None]
            if use_meta:
                lm = np.zeros(len(tr))
                for a, b in GroupKFold(5).split(meta["train"], groups=tr.patient_id.values):
                    lm[b] = LogisticRegression(max_iter=2000, class_weight="balanced").fit(
                        meta["train"][a], y["train"][a]).decision_function(meta["train"][b])
                X = np.column_stack([X, lm])
            teacher = LogisticRegression(C=1e6, max_iter=2000).fit(X, y["train"]).predict_proba(X)[:, 1]
        aux["train"]["extra"] = np.stack([tr.class3.values, (tr.source.values == "ISIC"), lb, teacher],
                                         1).astype(np.float32)
    print(f"Imágenes cargadas en {time.time() - t0:.1f}s")

    img_size = cfg.get("img_size", 224)
    ds_train = T["CachedImageDataset"](imgs["train"], meta["train"], y["train"], aux["train"], train=True,
                                       img_size=img_size, meta_noise=cfg["meta_noise"],
                                       cont_mask=enc.cont_mask if use_meta else None)
    counts = np.bincount(y["train"], minlength=2)
    sw = 1.0 / counts[y["train"]]
    if cfg.get("lesion_weighted"):
        # x16: cada lesión pesa lo mismo (1 / nº de imágenes) y cada clase suma lo mismo
        lw = 1.0 / parts["train"].lesion_id.map(parts["train"].lesion_id.value_counts()).values
        sw = lw / np.array([lw[y["train"] == c].sum() for c in (0, 1)])[y["train"]]
    sampler = WeightedRandomSampler(torch.as_tensor(sw, dtype=torch.double),
                                    num_samples=len(y["train"]), replacement=True,
                                    generator=torch.Generator().manual_seed(seed))
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                          pin_memory=True, drop_last=True)

    def eval_loader(role, meta_arr=None):
        m = meta[role] if meta_arr is None else meta_arr
        ds = T["CachedImageDataset"](imgs[role], m, y[role], aux[role], train=False, img_size=img_size)
        return DataLoader(ds, batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = T["MultimodalNet"](cfg["model"] if use_meta else "image", enc.dim, gate=cfg["gate"],
                               xattn=cfg["xattn"], aux=cfg["aux"], mod_drop_img=cfg.get("mod_drop_img", 0.0),
                               mod_drop_meta=cfg.get("mod_drop_meta", 0.0),
                               meta_dropout=cfg.get("meta_dropout"), aux_diam=cfg.get("aux_diam", False),
                               backbone=cfg.get("backbone", "densenet121"),
                               fmap_film=cfg.get("fmap_film", False)).to(device)
    # cabezas extra de la x11 sobre las características agrupadas de la imagen (capturadas con un hook)
    extra_heads, feat = None, {}
    if thick_w or adv_w:
        fd = 2048 if cfg.get("backbone") == "resnet50" else 1024
        head = lambda k: torch.nn.Sequential(torch.nn.Linear(fd, 64), torch.nn.ReLU(), torch.nn.Dropout(0.5),
                                             torch.nn.Linear(64, k))
        extra_heads = torch.nn.ModuleDict({"thick": head(3), "src": head(2)}).to(device)
        model.features.register_forward_hook(lambda mod, inp, out: feat.__setitem__("f", out))

    class GradRev(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, lam_):
            ctx.lam = lam_
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return -ctx.lam * g, None

    # batería x13: cabezas unimodales (ogm, gblend), proyecciones contrastivas y regresión de Breslow
    ogm_a, gb_w, ct_w = cfg.get("ogm_alpha", 0.0), cfg.get("gblend_weight", 0.0), cfg.get("contrast_weight", 0.0)
    x13 = None
    if ogm_a or gb_w or ct_w or bres_w:
        zd = 256 + 64 if use_meta and cfg["model"] == "concat" else 256
        x13 = torch.nn.ModuleDict({
            "uni_img": torch.nn.Linear(256, 2), "uni_meta": torch.nn.Linear(64, 2),
            "p_img": torch.nn.Linear(256, 64), "p_meta": torch.nn.Linear(64, 64),
            "bres": torch.nn.Sequential(torch.nn.Linear(zd, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))}).to(device)
    ogm_ratios = []

    aux_w = cfg.get("aux_weight", args.aux_weight)
    class_w = torch.tensor(len(y["train"]) / (2 * counts), dtype=torch.float32, device=device)
    if cfg["loss"] == "focal":
        crit = T["FocalLoss"](alpha=class_w, gamma=2.0)
    else:
        crit = torch.nn.CrossEntropyLoss(weight=class_w)
    # x15: SWA de los pesos desde la época swa_start, con label smoothing (CE ponderada en lugar de focal)
    swa_start, swa_model = cfg.get("swa_start", 0), None
    if swa_start:
        args.epochs = cfg.get("epochs", args.epochs) if not args.smoke else 2
        swa_start = 1 if args.smoke else swa_start
        crit = torch.nn.CrossEntropyLoss(weight=class_w, label_smoothing=cfg.get("label_smoothing", 0.0))
        swa_model = torch.optim.swa_utils.AveragedModel(model)
    # meta_lr_mult: lr propia para la rama de metadatos (recién inicializada y con poca señal)
    meta_prefix = ("meta_mlp", "meta_proj", "meta_out", "film")
    pm = [p for n, p in model.named_parameters() if n.startswith(meta_prefix)]
    pr = [p for n, p in model.named_parameters() if not n.startswith(meta_prefix)]
    if extra_heads is not None:
        pr += list(extra_heads.parameters())
    if x13 is not None:
        pr += list(x13.parameters())
    # lr y wd: la configuración del experimento (batería x8) tiene prioridad sobre los argumentos
    lr, wd = cfg.get("lr", args.lr), cfg.get("wd", args.wd)
    groups = [{"params": pr}] + ([{"params": pm, "lr": lr * cfg.get("meta_lr_mult", 1.0)}] if pm else [])
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    rng = np.random.default_rng(seed)
    staged = cfg.get("staged", False) and use_meta
    if staged:
        # x13: la rama de metadatos se preentrena sola (50 épocas, solo train) y se congela
        head_m = torch.nn.Linear(64, 2).to(device)
        o2 = torch.optim.Adam(list(model.meta_mlp.parameters()) + list(head_m.parameters()), lr=1e-3)
        Mt = torch.as_tensor(meta["train"], device=device)
        Yt = torch.as_tensor(y["train"], device=device)
        ce = torch.nn.CrossEntropyLoss(weight=class_w)
        model.meta_mlp.train()
        for _ in range(50):
            for b in torch.randperm(len(Yt), device=device).split(64):
                if len(b) < 2:
                    continue
                o2.zero_grad()
                ce(head_m(model.meta_mlp(Mt[b])), Yt[b]).backward()
                o2.step()
        for p_ in model.meta_mlp.parameters():
            p_.requires_grad_(False)

    def aux_loss(aux_out, a_age, a_sex, a_site, a_diam):
        p_age, p_sex, p_site, p_diam = aux_out
        loss = torch.zeros((), device=device)
        if cfg.get("aux_diam", False):
            md = ~torch.isnan(a_diam)
            if md.any():
                loss = loss + F.mse_loss(p_diam[md].float(), a_diam[md])
        m = ~torch.isnan(a_age)
        if m.any():
            loss = loss + F.mse_loss(p_age[m].float(), a_age[m])
        if (a_sex >= 0).any():
            loss = loss + F.cross_entropy(p_sex.float(), a_sex, ignore_index=-1)
        if (a_site >= 0).any():
            loss = loss + F.cross_entropy(p_site.float(), a_site, ignore_index=-1)
        return loss

    @torch.no_grad()
    def predict(loader):
        model.eval()
        ps = []
        for img, m, *_ in loader:
            with torch.autocast("cuda", enabled=amp):
                logits = model(img.to(device, non_blocking=True), m.to(device) if use_meta else None)
            ps.append(torch.softmax(logits.float(), 1)[:, 1].cpu().numpy())
        return np.concatenate(ps)

    @torch.no_grad()
    def predict_tta(loader):
        """Media de la probabilidad sobre la imagen original y sus volteos horizontal y vertical (x11)."""
        model.eval()
        ps = []
        for img, m, *_ in loader:
            img, mm = img.to(device, non_blocking=True), (m.to(device) if use_meta else None)
            acc = 0
            for v in (img, torch.flip(img, [3]), torch.flip(img, [2])):
                with torch.autocast("cuda", enabled=amp):
                    acc = acc + torch.softmax(model(v, mm).float(), 1)[:, 1]
            ps.append((acc / 3).cpu().numpy())
        return np.concatenate(ps)

    dl_val = eval_loader("val")
    best = {"auc": -1, "epoch": 0, "state": None}
    history, bad = [], 0
    for epoch in range(1, args.epochs + 1):
        if cfg["permute_train"] and use_meta:
            ds_train.meta = meta["train"][rng.permutation(len(y["train"]))]
        model.train()
        if cfg.get("lpft_epochs"):
            # x16 (LP-FT): red de imagen congelada (pesos y BatchNorm) las primeras épocas
            frozen = epoch <= cfg["lpft_epochs"]
            for p_ in model.features.parameters():
                p_.requires_grad_(not frozen)
            if frozen:
                model.features.eval()
        if staged:
            model.meta_mlp.eval()        # rama congelada: sin actualizar las estadísticas de BatchNorm
        opt.zero_grad(set_to_none=True)
        run_loss, nb = 0.0, 0
        for bi, (img, m, lab, a_age, a_sex, a_site, a_diam, *extra) in enumerate(dl_train):
            img, m, lab = img.to(device, non_blocking=True), m.to(device), lab.to(device)
            a_age, a_sex, a_site, a_diam = a_age.to(device), a_sex.to(device), a_site.to(device), a_diam.to(device)
            mix = cfg["mixup"] and rng.random() > 0.5
            if mix:
                lam = float(rng.beta(0.2, 0.2))
                idx = torch.randperm(img.size(0), device=device)
                img = lam * img + (1 - lam) * img[idx]
                if use_meta:
                    m = lam * m + (1 - lam) * m[idx]      # one-hot -> mezcla convexa (categoría "suave")
            with torch.autocast("cuda", enabled=amp):
                out = model(img, m if use_meta else None, return_aux=True)
                logits, aux_out = (out if isinstance(out, tuple) else (out, None))
                if mix:
                    loss = lam * crit(logits.float(), lab) + (1 - lam) * crit(logits.float(), lab[idx])
                else:
                    loss = crit(logits.float(), lab)
                if kd_b:
                    tp = extra[0].to(device)[:, 3]
                    if mix:
                        tp = lam * tp + (1 - lam) * tp[idx]
                    qt = torch.stack([1 - tp, tp], 1).clamp(1e-6, 1)
                    loss = (1 - kd_b) * loss + kd_b * F.kl_div(F.log_softmax(logits.float(), 1), qt,
                                                               reduction="batchmean")
                if aux_out is not None:
                    if mix:
                        la = lam * aux_loss(aux_out, a_age, a_sex, a_site, a_diam) + \
                            (1 - lam) * aux_loss(aux_out, a_age[idx], a_sex[idx], a_site[idx], a_diam[idx])
                    else:
                        la = aux_loss(aux_out, a_age, a_sex, a_site, a_diam)
                    loss = loss + aux_w * la
                if extra_heads is not None:
                    pooled = F.adaptive_avg_pool2d(F.relu(feat["f"]), 1).flatten(1)
                    t = extra[0].to(device)

                    def xloss(o, col):
                        o = o.float()
                        if mix:
                            return lam * F.cross_entropy(o, t[:, col].long()) + \
                                (1 - lam) * F.cross_entropy(o, t[idx, col].long())
                        return F.cross_entropy(o, t[:, col].long())
                    if thick_w:
                        loss = loss + thick_w * xloss(extra_heads["thick"](pooled), 0)
                    if adv_w:
                        loss = loss + adv_w * xloss(extra_heads["src"](GradRev.apply(pooled, 0.1)), 1)
                if x13 is not None:
                    ip_, z_ = model.last_ip.float(), model.last_z.float()
                    mf_ = model.last_mf.float() if use_meta else None

                    def mixce(o):
                        if mix:
                            return lam * F.cross_entropy(o, lab) + (1 - lam) * F.cross_entropy(o, lab[idx])
                        return F.cross_entropy(o, lab)
                    if ogm_a and use_meta:
                        # cabezas unimodales con características desacopladas: solo miden la contribución
                        oi, om = x13["uni_img"](ip_.detach()), x13["uni_meta"](mf_.detach())
                        loss = loss + mixce(oi) + mixce(om)
                        with torch.no_grad():
                            si = torch.softmax(oi, 1).gather(1, lab[:, None]).sum()
                            sm_ = torch.softmax(om, 1).gather(1, lab[:, None]).sum()
                        ogm_ratios.append((si / sm_.clamp_min(1e-6)).item())
                    if gb_w and use_meta:
                        loss = loss + gb_w * (mixce(x13["uni_img"](ip_)) + mixce(x13["uni_meta"](mf_)))
                    if ct_w and use_meta:
                        a_ = F.normalize(x13["p_img"](ip_), dim=1)
                        b_ = F.normalize(x13["p_meta"](mf_), dim=1)
                        s_ = a_ @ b_.t() / 0.1
                        tgt = torch.arange(len(a_), device=device)
                        loss = loss + ct_w * 0.5 * (F.cross_entropy(s_, tgt) + F.cross_entropy(s_.t(), tgt))
                    if bres_w:
                        tb = extra[0].to(device)[:, 2]
                        pb_ = x13["bres"](z_).squeeze(1)

                        def mse(t_):
                            mk = ~torch.isnan(t_)
                            return F.mse_loss(pb_[mk], t_[mk]) if mk.any() else pb_.sum() * 0
                        loss = loss + bres_w * ((lam * mse(tb) + (1 - lam) * mse(tb[idx])) if mix else mse(tb))
            scaler.scale(loss / args.accum).backward()
            if (bi + 1) % args.accum == 0:
                scaler.unscale_(opt)
                if ogm_a and ogm_ratios:
                    # OGM: se frena el codificador de la modalidad dominante (Peng et al., CVPR 2022)
                    r_ = float(np.mean(ogm_ratios))
                    ogm_ratios.clear()
                    ci, cm = ((1 - np.tanh(ogm_a * (r_ - 1)), 1.0) if r_ > 1
                              else (1.0, 1 - np.tanh(ogm_a * (1 / r_ - 1))))
                    for n_, p_ in model.named_parameters():
                        if p_.grad is not None:
                            if n_.startswith(("features", "img_proj")):
                                p_.grad.mul_(ci)
                            elif n_.startswith("meta_mlp"):
                                p_.grad.mul_(cm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
            run_loss += loss.item(); nb += 1
        sched.step()
        pv = predict(dl_val)
        auc = roc_auc_score(y["val"], pv) if len(np.unique(y["val"])) == 2 else float("nan")
        history.append({"epoch": epoch, "train_loss": run_loss / max(nb, 1), "val_auc": auc,
                        "lr": opt.param_groups[0]["lr"]})
        print(f"ep {epoch:3d} loss {run_loss / max(nb, 1):.4f} val_auc {auc:.4f}", flush=True)
        if swa_start:
            # x15: sin early stopping ni selección en validación; se promedian los pesos desde swa_start
            if epoch >= swa_start:
                swa_model.update_parameters(model)
            continue
        if auc > best["auc"]:
            best = {"auc": auc, "epoch": epoch, "state": copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})}
            bad = 0
        else:
            bad += 1
            if bad >= args.patience:
                print(f"Early stopping en época {epoch} (mejor {best['epoch']})")
                break

    if swa_start:
        # BatchNorm recalculado con el train (sin aumento) y el modelo promediado como modelo final
        model = swa_model.module
        for mod_ in model.modules():
            if isinstance(mod_, torch.nn.modules.batchnorm._BatchNorm):
                mod_.reset_running_stats()
                mod_.momentum = None
        model.train()
        with torch.no_grad():
            for img, m, *_ in eval_loader("train"):
                with torch.autocast("cuda", enabled=amp):
                    model(img.to(device, non_blocking=True), m.to(device) if use_meta else None)
        best = {"auc": roc_auc_score(y["val"], predict(dl_val)), "epoch": args.epochs, "state": None}
    else:
        model.load_state_dict(best["state"])
    rows = []
    p_val = predict(dl_val)
    p_test = predict(eval_loader("test"))
    p_perm = np.full_like(p_test, np.nan)
    gate_mean = None
    if use_meta:
        perm = np.random.default_rng(seed + 1).permutation(len(y["test"]))
        p_perm = predict(eval_loader("test", meta["test"][perm]))
        if cfg["model"] == "gated" and cfg["gate"]:
            gs = []
            with torch.no_grad():
                for img, m, *_ in eval_loader("test"):
                    with torch.autocast("cuda", enabled=amp):
                        model(img.to(device), m.to(device))
                    gs.append(model.last_gate.float().cpu().numpy())
            gate_mean = np.concatenate(gs).mean(0).tolist()
    for role, p, pp in [("val", p_val, np.full_like(p_val, np.nan)), ("test", p_test, p_perm)]:
        d = parts[role]
        rr = role
        if outer_role is not None:     # modo --inner: val interna = fuera de fold; test = val + test externos
            rr = "oof" if role == "val" else outer_role.loc[d.image_id.values].values
        rows.append(pd.DataFrame({"image_id": d.image_id.values, "role": rr, "label": y[role],
                                  "prob": p, "prob_meta_permuted": pp}))
        if cfg.get("save_tta"):
            rows[-1]["prob_tta"] = predict_tta(dl_val if role == "val" else eval_loader("test"))
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_smoke" if args.smoke else ""
    pd.concat(rows).to_csv(out_dir / f"preds{tag}.csv", index=False)
    write_json({"exp": args.exp, "split": args.split, "inner": args.inner, "config": cfg, "seed": seed, "encoder": enc.to_json(),
                "best_epoch": best["epoch"], "best_val_auc": best["auc"], "history": history,
                "gate_mean_test": gate_mean, "n": {r: len(d) for r, d in parts.items()},
                "minutes": (time.time() - t0) / 60, "args": vars(args),
                "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"},
               out_dir / f"run{tag}.json")
    if args.save_model:
        torch.save(best["state"], out_dir / "best_model.pth")
    # En los experimentos exploratorios (tier 3) el test no se mira hasta el paso final del protocolo
    if cfg["tier"] == 3 or args.inner is not None:
        test_txt = "test_auc=oculto"
    else:
        test_auc = roc_auc_score(y["test"], p_test) if len(np.unique(y["test"])) == 2 else float("nan")
        test_txt = f"test_auc={test_auc:.4f}"
    print(f"FIN {args.exp} {args.split}: best_epoch={best['epoch']} val_auc={best['auc']:.4f} "
          f"{test_txt} ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
