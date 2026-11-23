#!/usr/bin/env python3
"""
Puerta en VALIDACIÓN de la batería x11 (métodos de entrenamiento; protocolo fijado el 2026-10-21).
No lee el test.

Candidatos (6): para cada método (base, thick, adv), con y sin TTA, ensemble de las 3 semillas de concat frente al
de las 3 semillas de image con el MISMO método. ΔAUC por partición (repetición 0), por imagen y emparejada.
Regla 3: media >= +0,005 y positiva en >= 2/3 de las particiones. Regla 5: el mismo criterio dentro de VRUH.
Candidato final: entre los que pasan, el de mayor ΔAUC media (results/tables/x11_decision.json).

  python pipeline/27_gate_x11.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, load_unified, write_json  # noqa: E402
from experiments import X11_METHODS  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
GATE_MEAN, GATE_POS_FRAC = 0.005, 2 / 3
SPLITS = [f"cv_r0_f{k}" for k in range(5)]
ARMS = ("image", "concat")


def seeds(mt, arm):
    return [f"x11_{mt}_{arm}"] + [f"x11_{mt}_{arm}_s{k}" for k in (1, 2)]


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


def main():
    uni = load_unified().set_index("image_id")
    src, lesion = uni.source, uni.lesion_id

    def scores(lab, p):
        out = {"all": auc(lab, p)}
        for s in ["VRUH", "ISIC"]:
            k = lab.index[(src.loc[lab.index] == s).values]
            out[s] = auc(lab.loc[k], p.loc[k])
        q = pd.Series(p.values, index=lesion.loc[p.index].values).groupby(level=0).mean()
        yl = pd.Series(lab.values, index=lesion.loc[lab.index].values).groupby(level=0).first()
        out["lesion"] = auc(yl.loc[q.index], q)
        return out

    preds, missing = {}, []
    for mt in X11_METHODS:
        for a in ARMS:
            for e in seeds(mt, a):
                for s in SPLITS:
                    v = val(e, s)
                    if v is None:
                        missing.append((e, s))
                    preds[(e, s)] = v
    if missing:
        sys.exit(f"Faltan {len(missing)} entrenamientos, p. ej. {missing[:3]}")

    rows, log_rows, single = [], [], []
    for mt in X11_METHODS:
        for col in ("prob", "prob_tta"):
            name = f"x11_{mt}" + ("_tta" if col == "prob_tta" else "")
            d = []
            for s in SPLITS:
                idx = preds[(seeds(mt, "image")[0], s)].index
                lab = preds[(seeds(mt, "image")[0], s)].label.loc[idx]
                sc = {}
                for a in ARMS:
                    ps = [preds[(e, s)][col].loc[idx] for e in seeds(mt, a)]
                    sc[a] = scores(lab, pd.concat(ps, axis=1).mean(1))
                    single += [{"method": name, "arm": a, "split": s, "exp": e, "auc_val": auc(lab, p)}
                               for e, p in zip(seeds(mt, a), ps)]
                d.append({k: sc["concat"][k] - sc["image"][k] for k in ("all", "VRUH", "ISIC", "lesion")}
                         | {"auc_image": sc["image"]["all"], "auc_concat": sc["concat"]["all"]})
            d = pd.DataFrame(d, index=SPLITS)
            r3, r5 = passes(d["all"]), passes(d["VRUH"])
            rows.append({"candidate": name, "auc_val_image": d.auc_image.mean(), "auc_val_concat": d.auc_concat.mean(),
                         "val_delta_mean": d["all"].mean(), "val_delta_n_positive": int((d["all"] > 0).sum()),
                         "val_delta_VRUH_mean": d["VRUH"].mean(), "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()),
                         "val_delta_ISIC_mean": d["ISIC"].mean(), "val_delta_lesion_mean": d["lesion"].mean(),
                         "rule3": r3, "rule5": r5, "passes": r3 and r5})
            print(f"{name:15s} AUC val image {d.auc_image.mean():.4f} concat {d.auc_concat.mean():.4f} | "
                  f"ΔAUC {d['all'].mean():+.4f} ({(d['all'] > 0).sum()}/5) | VRUH {d['VRUH'].mean():+.4f} "
                  f"({(d['VRUH'] > 0).sum()}/5) | ISIC {d['ISIC'].mean():+.4f} | lesión {d['lesion'].mean():+.4f} "
                  f"-> {'PASA' if r3 and r5 else 'no pasa'}")
            log_rows.append({"date": str(date.today()), "step": "x11", "variant": name, "meta_model": "end-to-end",
                             "n_splits": len(d), "val_auc_img_mean": d.auc_image.mean(),
                             "val_auc_stack_mean": d.auc_concat.mean(), "val_delta_mean": d["all"].mean(),
                             "val_delta_median": d["all"].median(), "val_delta_n_positive": int((d["all"] > 0).sum()),
                             "passes_gate": bool(r3 and r5), "role": "candidate",
                             "val_delta_VRUH_mean": d["VRUH"].mean(),
                             "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()),
                             "val_delta_ISIC_mean": d["ISIC"].mean(),
                             "val_delta_ISIC_n_positive": int((d["ISIC"] > 0).sum())})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_x11.csv", index=False)
    pd.DataFrame(single).to_csv(RESULTS_DIR / "tables" / "exploratory_gate_x11_runs.csv", index=False)
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)

    ok = res[res.passes]
    final = ok.loc[ok.val_delta_mean.idxmax()] if len(ok) else None
    print("\n" + (f"Candidato final: {final.candidate} (ΔAUC val {final.val_delta_mean:+.4f})" if final is not None
                  else "Ningún candidato pasa la puerta: la batería termina aquí."))
    write_json({"date": str(date.today()), "passes": final is not None,
                "final_candidate": None if final is None else final.candidate,
                "val_delta_mean": None if final is None else float(final.val_delta_mean),
                "candidates": res.to_dict("records")},
               RESULTS_DIR / "tables" / "x11_decision.json")


if __name__ == "__main__":
    main()
