#!/usr/bin/env python3
"""
Puerta en VALIDACIÓN de la familia x8 + x9 (protocolo fijado el 2026-10-20). No lee el test.

Configuraciones de cada brazo (image, concat): la referencia (`image` / `concat_meta`), las 5 de la x8 (lr × wd) y
las 3 de la x9 (red base × resolución). Candidatos fijados antes de ver el brazo concat:
  1) x8_best:    mejor concat frente a mejor image entre la referencia y la x8 (6 por brazo);
  2) x9_best:    lo mismo entre la referencia y la x9 (4 por brazo);
  3) fam_best:   lo mismo entre las 9 configuraciones;
  4) x8_ens, 5) x9_ens, 6) fam_ens: media de probabilidades de las 6 / 4 / 9 configuraciones de cada brazo.
El "mejor" de cada brazo se elige por la AUC media en validación (por imagen). ΔAUC por partición, emparejada.
Regla 3: media >= +0,005 y positiva en >= 2/3 de las particiones. Regla 5: el mismo criterio dentro de VRUH.
Candidato final: entre los que pasan, el de mayor ΔAUC media. Sus modelos que falten en las repeticiones 1 y 2 se
listan en la decisión (phase2_tasks) para entrenarlos; el test se abre después, una sola vez.

  python pipeline/25_gate_family.py
"""
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, load_unified, write_json  # noqa: E402
from experiments import X8_GRID, X9_GRID  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
GATE_MEAN, GATE_POS_FRAC = 0.005, 2 / 3
SPLITS = [f"cv_r0_f{k}" for k in range(5)]
ARMS = ("image", "concat")
REF = {"image": "image", "concat": "concat_meta"}
CFG = {"ref": REF,
       **{f"lr{lr}_wd{wd}": {a: f"x8_{a}_lr{lr}_wd{wd}" for a in ARMS} for lr, wd in X8_GRID},
       **{f"{bb}_{size}": {a: f"x9_{a}_{bb}_{size}" for a in ARMS} for bb, size in X9_GRID}}
FAMILIES = {"x8": ["ref"] + [f"lr{lr}_wd{wd}" for lr, wd in X8_GRID],
            "x9": ["ref"] + [f"{bb}_{size}" for bb, size in X9_GRID]}
FAMILIES["fam"] = list(CFG)


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

    def scores(lab, p):
        out = {"all": auc(lab, p)}
        for s in ["VRUH", "ISIC"]:
            k = lab.index[(src.loc[lab.index] == s).values]
            out[s] = auc(lab.loc[k], p.loc[k])
        q = pd.Series(p.values, index=lesion.loc[p.index].values).groupby(level=0).mean()
        yl = pd.Series(lab.values, index=lesion.loc[lab.index].values).groupby(level=0).first()
        out["lesion"] = auc(yl.loc[q.index], q)
        return out

    probs, labels, missing = {}, {}, []
    for s in SPLITS:
        idx = val("image", s).index
        for c, exps in CFG.items():
            for a in ARMS:
                v = val(exps[a], s)
                if v is None:
                    missing.append((exps[a], s))
                    continue
                probs[(c, a, s)] = v.prob.loc[idx]
                labels[s] = v.label.loc[idx]
    if missing:
        sys.exit(f"Faltan {len(missing)} entrenamientos, p. ej. {missing[:3]}")

    single = pd.DataFrame([{"config": c, "arm": a, "split": s, **scores(labels[s], probs[(c, a, s)])}
                           for c in CFG for a in ARMS for s in SPLITS])
    single.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_family_x8x9_runs.csv", index=False)
    mean = single.groupby(["arm", "config"])["all"].mean()
    print("AUC media en validación por configuración (repetición 0, 5 particiones):")
    print(mean.unstack(0).round(4).to_string())

    cands = {}
    for fam, cfgs in FAMILIES.items():
        best = {a: mean.loc[a].loc[cfgs].idxmax() for a in ARMS}
        cands[f"{fam}_best"] = {a: [best[a]] for a in ARMS}
        cands[f"{fam}_ens"] = {a: cfgs for a in ARMS}
    rows, log_rows = [], []
    print()
    for name, mem in cands.items():
        d = []
        for s in SPLITS:
            sc = {a: scores(labels[s], pd.concat([probs[(c, a, s)] for c in mem[a]], axis=1).mean(1)) for a in ARMS}
            d.append({k: sc["concat"][k] - sc["image"][k] for k in ("all", "VRUH", "ISIC", "lesion")})
        d = pd.DataFrame(d, index=SPLITS)
        r3, r5 = passes(d["all"]), passes(d["VRUH"])
        rows.append({"candidate": name, "members_image": mem["image"], "members_concat": mem["concat"],
                     "val_delta_mean": d["all"].mean(), "val_delta_n_positive": int((d["all"] > 0).sum()),
                     "val_delta_VRUH_mean": d["VRUH"].mean(), "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()),
                     "val_delta_ISIC_mean": d["ISIC"].mean(), "val_delta_lesion_mean": d["lesion"].mean(),
                     "rule3": r3, "rule5": r5, "passes": r3 and r5})
        mi = "+".join(mem["image"]) if len(mem["image"]) < 3 else f"{len(mem['image'])} config."
        mc = "+".join(mem["concat"]) if len(mem["concat"]) < 3 else f"{len(mem['concat'])} config."
        print(f"{name:9s} image [{mi}] vs concat [{mc}]: ΔAUC val {d['all'].mean():+.4f} ({(d['all'] > 0).sum()}/5) | "
              f"VRUH {d['VRUH'].mean():+.4f} ({(d['VRUH'] > 0).sum()}/5) | ISIC {d['ISIC'].mean():+.4f} | "
              f"lesión {d['lesion'].mean():+.4f} -> {'PASA' if r3 and r5 else 'no pasa'}")
        log_rows.append({"date": str(date.today()), "step": "x8x9", "variant": f"family_{name}",
                         "meta_model": "end-to-end", "n_splits": len(d), "val_delta_mean": d["all"].mean(),
                         "val_delta_median": d["all"].median(), "val_delta_n_positive": int((d["all"] > 0).sum()),
                         "passes_gate": bool(r3 and r5), "role": "candidate",
                         "val_delta_VRUH_mean": d["VRUH"].mean(), "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()),
                         "val_delta_ISIC_mean": d["ISIC"].mean(), "val_delta_ISIC_n_positive": int((d["ISIC"] > 0).sum())})
    res = pd.DataFrame(rows)
    res.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_family_x8x9.csv", index=False)
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)

    ok = res[res.passes]
    final = ok.loc[ok.val_delta_mean.idxmax()] if len(ok) else None
    phase2 = []
    if final is not None:
        exps = [CFG[c][a] for a in ARMS for c in final[f"members_{a}"]]
        phase2 = [[e, f"cv_r{r}_f{k}"] for e in exps if e.startswith(("x8_", "x9_")) for r in (1, 2) for k in range(5)]
        print(f"\nCandidato final: {final.candidate} (ΔAUC val {final.val_delta_mean:+.4f}); "
              f"{len(phase2)} entrenamientos de las repeticiones 1 y 2 por hacer o comprobar")
    else:
        print("\nNingún candidato pasa la puerta: la familia termina aquí.")
    write_json({"date": str(date.today()), "passes": final is not None,
                "final_candidate": None if final is None else final.candidate,
                "members": None if final is None else {a: [CFG[c][a] for c in final[f"members_{a}"]] for a in ARMS},
                "val_delta_mean": None if final is None else float(final.val_delta_mean),
                "candidates": res.drop(columns=["members_image", "members_concat"]).to_dict("records"),
                "phase2_tasks": phase2},
               RESULTS_DIR / "tables" / "family_x8x9_decision.json")


if __name__ == "__main__":
    main()
