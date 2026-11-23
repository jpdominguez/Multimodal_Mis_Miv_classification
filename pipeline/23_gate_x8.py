#!/usr/bin/env python3
"""
Puerta en VALIDACIÓN de la batería x8: ajuste de lr × wd en los DOS brazos (protocolo fijado el 2026-10-20).
No lee el test.

  1) En cada brazo (image, concat) se elige la configuración con la mayor AUC media en validación (por imagen) sobre
     las 5 particiones de la repetición 0. lr 1e-4 / wd 0.01 es la receta actual (`image` y `concat_meta`).
  2) Principal: mejor concat frente a mejor image, ΔAUC por partición, por imagen y emparejada.
  3) Descriptiva: concat frente a image con la MISMA configuración, en las 6.
  Regla 3: media >= +0,005 y positiva en >= 2/3 de las particiones.
  Regla 5: el mismo criterio dentro de VRUH.

  python pipeline/23_gate_x8.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, load_unified, write_json  # noqa: E402
from experiments import X8_GRID  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
GATE_MEAN, GATE_POS_FRAC = 0.005, 2 / 3
SPLITS = [f"cv_r0_f{k}" for k in range(5)]
BASE_CFG = ("1e-4", "0.01")
CONFIGS = [BASE_CFG] + X8_GRID


def exp_name(arm, cfg):
    if cfg == BASE_CFG:
        return "image" if arm == "image" else "concat_meta"
    return f"x8_{arm}_lr{cfg[0]}_wd{cfg[1]}"


def val(exp, split):
    for base in (EXP_DIR, RUNS_DIR):
        f = base / exp / split / "preds.csv"
        if f.exists():
            d = pd.read_csv(f, dtype={"image_id": str})
            return d[d.role == "val"].set_index("image_id")
    return None


def auc(y, p):
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def passes(d):
    d = np.asarray(d, float)
    return bool(np.nanmean(d) >= GATE_MEAN and (d > 0).sum() >= np.ceil(GATE_POS_FRAC * len(d)))


def main():
    uni = load_unified().set_index("image_id")
    src, lesion = uni.source, uni.lesion_id

    def scores(v, idx):
        out = {"all": auc(v.label.loc[idx], v.prob.loc[idx])}
        for s in ["VRUH", "ISIC"]:
            k = idx[(src.loc[idx] == s).values]
            out[s] = auc(v.label.loc[k], v.prob.loc[k])
        q = pd.Series(v.prob.loc[idx].values, index=lesion.loc[idx].values).groupby(level=0).mean()
        yl = pd.Series(v.label.loc[idx].values, index=lesion.loc[idx].values).groupby(level=0).first()
        out["lesion"] = auc(yl.loc[q.index], q)
        return out

    rows, missing = [], []
    for s in SPLITS:
        ref = val("image", s)
        idx = ref.index
        for arm in ("image", "concat"):
            for cfg in CONFIGS:
                e = exp_name(arm, cfg)
                v = val(e, s)
                if v is None:
                    missing.append((e, s))
                    continue
                rows.append({"arm": arm, "lr": cfg[0], "wd": cfg[1], "exp": e, "split": s, **scores(v, idx)})
    if missing:
        sys.exit(f"Faltan {len(missing)} entrenamientos, p. ej. {missing[:3]}")
    g = pd.DataFrame(rows)
    g.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_x8_runs.csv", index=False)

    mean = g.groupby(["arm", "lr", "wd"])[["all", "VRUH", "ISIC", "lesion"]].mean()
    print("AUC media en validación (repetición 0, 5 particiones):")
    print(mean.round(4).to_string())
    best = {arm: mean.loc[arm]["all"].idxmax() for arm in ("image", "concat")}
    print(f"\nMejor configuración en validación: image lr {best['image'][0]} wd {best['image'][1]}; "
          f"concat lr {best['concat'][0]} wd {best['concat'][1]}")

    def paired(cfg_c, cfg_i):
        a = g[(g.arm == "concat") & (g.lr == cfg_c[0]) & (g.wd == cfg_c[1])].set_index("split")
        b = g[(g.arm == "image") & (g.lr == cfg_i[0]) & (g.wd == cfg_i[1])].set_index("split")
        return (a[["all", "VRUH", "ISIC", "lesion"]] - b[["all", "VRUH", "ISIC", "lesion"]]).loc[SPLITS]

    comps = {"x8_best_concat_vs_best_image": paired(best["concat"], best["image"])}
    comps.update({f"x8_matched_lr{c[0]}_wd{c[1]}": paired(c, c) for c in CONFIGS})
    log_rows = []
    print()
    for name, d in comps.items():
        r3, r5 = passes(d["all"]), passes(d["VRUH"])
        principal = name == "x8_best_concat_vs_best_image"
        print(f"{name:32s} ΔAUC val {d['all'].mean():+.4f} ({(d['all'] > 0).sum()}/5) | VRUH {d['VRUH'].mean():+.4f} "
              f"({(d['VRUH'] > 0).sum()}/5) | ISIC {d['ISIC'].mean():+.4f} | lesión {d['lesion'].mean():+.4f} "
              f"-> regla 3 {'sí' if r3 else 'no'}, regla 5 {'sí' if r5 else 'no'}"
              + ("  [PRINCIPAL: " + ("PASA" if r3 and r5 else "NO PASA") + "]" if principal else ""))
        log_rows.append({"date": str(date.today()), "step": "x8", "variant": name, "meta_model": "end-to-end",
                         "n_splits": len(d), "val_delta_mean": d["all"].mean(), "val_delta_median": d["all"].median(),
                         "val_delta_n_positive": int((d["all"] > 0).sum()), "passes_gate": bool(r3 and r5),
                         "role": "candidate" if principal else "descriptive",
                         "val_delta_VRUH_mean": d["VRUH"].mean(), "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()),
                         "val_delta_ISIC_mean": d["ISIC"].mean(), "val_delta_ISIC_n_positive": int((d["ISIC"] > 0).sum())})
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)
    # decisión para 24_x8_progress.py (qué configuraciones pasan a las 15 particiones)
    d = comps["x8_best_concat_vs_best_image"]
    write_json({"date": str(date.today()), "passes": passes(d["all"]) and passes(d["VRUH"]),
                "rule3": passes(d["all"]), "rule5": passes(d["VRUH"]),
                "best_image": list(best["image"]), "best_concat": list(best["concat"]),
                "exp_image": exp_name("image", best["image"]), "exp_concat": exp_name("concat", best["concat"]),
                "val_delta_mean": float(d["all"].mean()), "val_delta_n_positive": int((d["all"] > 0).sum()),
                "val_delta_VRUH_mean": float(d["VRUH"].mean()), "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum())},
               RESULTS_DIR / "tables" / "x8_decision.json")


if __name__ == "__main__":
    main()
