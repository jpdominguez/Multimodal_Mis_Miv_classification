#!/usr/bin/env python3
"""
Puerta en VALIDACIÓN de la batería x10 (antecedentes en la red, solo VRUH; protocolo fijado el 2026-10-21).
No lee el test. Ejecuta 15_gate_x6.py frente a image_vruh y deja la decisión en results/tables/x10_decision.json.
Candidato: ensemble de 5 semillas de x10_hx_vruh frente a 5 de image_vruh (regla 3). Descriptivo: lo mismo para
concat_meta_vruh (sin antecedentes).

  python pipeline/26_gate_x10.py
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, write_json  # noqa: E402

SEEDS = ["x10_hx_vruh"] + [f"x10_hx_vruh_s{k}" for k in range(1, 5)]


def main():
    r = subprocess.run([sys.executable, str(REV / "15_gate_x6.py"), "--ref", "image_vruh", "--tag", "x10",
                        "--exps", *SEEDS, "--family", "x10_hx_vruh"], capture_output=True, text=True)
    print(r.stdout, end="")
    if r.returncode != 0:
        sys.exit(r.stderr)
    log = pd.read_csv(RESULTS_DIR / "tables" / "exploratory_log.csv")
    log = log[(log.step == "x10") & log.variant.str.startswith("ens5_")]
    cands = [{"candidate": v.split("__")[0], "val_delta_mean": float(d), "val_delta_n_positive": int(n),
              "passes": bool(p) and "x10" in v}
             for v, d, n, p in zip(log.variant, log.val_delta_mean, log.val_delta_n_positive, log.passes_gate)]
    main_c = [c for c in cands if "x10" in c["candidate"]]
    if not main_c:
        sys.exit("No se encontró el ensemble de x10 en exploratory_log.csv")
    c = main_c[0]
    write_json({"date": str(date.today()), "passes": c["passes"], "final_candidate": c["candidate"],
                "val_delta_mean": c["val_delta_mean"], "candidates": cands},
               RESULTS_DIR / "tables" / "x10_decision.json")


if __name__ == "__main__":
    main()
