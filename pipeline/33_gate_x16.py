#!/usr/bin/env python3
"""
Puerta en VALIDACIÓN de la batería x16: LP-FT, muestreo por lesión y FiLM sobre el mapa de características
(protocolo fijado el 2026-10-21). No lee el test. Reutiliza las funciones de 30_gate_x13.py.

Candidatos (ensemble de 3 semillas, repetición 0): lpft y lesw (concat vs image con el mismo método) y fmfilm
(x16_fmfilm_concat vs x11_base_image). Si alguno pasa, se replica en las repeticiones 1 y 2 antes de plantear el test.
Regla 3: media >= +0,005 y positiva en >= 4/5; regla 5: lo mismo dentro de VRUH.

  python pipeline/33_gate_x16.py
"""
import importlib.util
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, load_unified, write_json  # noqa: E402

spec = importlib.util.spec_from_file_location("g13", REV / "30_gate_x13.py")
g = importlib.util.module_from_spec(spec)
spec.loader.exec_module(g)
SPLITS = [f"cv_r0_f{k}" for k in range(5)]
CANDS = {"lpft": ("x16_lpft_concat", "x16_lpft_image"), "lesw": ("x16_lesw_concat", "x16_lesw_image"),
         "fmfilm": ("x16_fmfilm_concat", "x11_base_image")}


def main():
    src = load_unified().set_index("image_id").source
    rows, log_rows = [], []
    for name, (cat, img) in CANDS.items():
        d = []
        for s in SPLITS:
            a, b = g.ens(g.seeds(cat), s), g.ens(g.seeds(img), s)
            if a is None or b is None:
                sys.exit(f"Faltan entrenamientos de {name} en {s}")
            (lab, pc), (_, pi) = a, b
            idx = lab.index
            r = {"split": s, "all": g.auc(lab, pc) - g.auc(lab, pi.loc[idx])}
            for so in ("VRUH", "ISIC"):
                k = idx[(src.loc[idx] == so).values]
                r[so] = g.auc(lab.loc[k], pc.loc[k]) - g.auc(lab.loc[k], pi.loc[k])
            d.append(r)
        d = pd.DataFrame(d)
        r3, r5 = g.passes(d["all"]), g.passes(d["VRUH"])
        rows.append({"candidate": name, "val_delta_mean": d["all"].mean(), "val_delta_n_positive": int((d["all"] > 0).sum()),
                     "val_delta_VRUH_mean": d["VRUH"].mean(), "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()),
                     "val_delta_ISIC_mean": d["ISIC"].mean(), "rule3": r3, "rule5": r5, "passes": r3 and r5})
        print(f"{name:8s} ΔAUC val {d['all'].mean():+.4f} ({(d['all'] > 0).sum()}/5) | VRUH {d['VRUH'].mean():+.4f} "
              f"({(d['VRUH'] > 0).sum()}/5) | ISIC {d['ISIC'].mean():+.4f} -> {'PASA' if r3 and r5 else 'no pasa'}")
        log_rows.append({"date": str(date.today()), "step": "x16", "variant": f"x16_{name}", "meta_model": name,
                         "n_splits": len(d), "val_delta_mean": d["all"].mean(), "val_delta_median": d["all"].median(),
                         "val_delta_n_positive": int((d["all"] > 0).sum()), "passes_gate": bool(r3 and r5),
                         "role": "candidate", "val_delta_VRUH_mean": d["VRUH"].mean(),
                         "val_delta_VRUH_n_positive": int((d["VRUH"] > 0).sum()), "val_delta_ISIC_mean": d["ISIC"].mean(),
                         "val_delta_ISIC_n_positive": int((d["ISIC"] > 0).sum())})
    R = pd.DataFrame(rows)
    R.to_csv(RESULTS_DIR / "tables" / "exploratory_gate_x16.csv", index=False)
    logf = RESULTS_DIR / "tables" / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        log = pd.concat([old[~old.variant.isin(log.variant)], log])
    log.to_csv(logf, index=False)
    ok = R[R.passes]
    final = ok.loc[ok.val_delta_mean.idxmax()] if len(ok) else None
    print("\n" + (f"Candidato final {final.candidate} (ΔAUC val {final.val_delta_mean:+.4f})"
                  if final is not None else "Ningún candidato pasa la puerta: la batería termina aquí."))
    write_json({"date": str(date.today()), "passes": final is not None,
                "final_candidate": None if final is None else final.candidate,
                "val_delta_mean": None if final is None else float(final.val_delta_mean),
                "candidates": R.to_dict("records")}, RESULTS_DIR / "tables" / "x16_decision.json")


if __name__ == "__main__":
    main()
