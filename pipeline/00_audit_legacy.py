#!/usr/bin/env python3
"""
Paso 0 - Auditoría reproducible del pipeline ORIGINAL (legacy/train_v3.py) usado en la versión original del estudio.

Comprueba y documenta (salida: pipeline/results/audit_legacy.md):
  A. El "test set" de 428 imágenes era la validación usada para early stopping/checkpoint.
  B. Pacientes ISIC repartidos por imagen entre train y validación (fuga).
  C. Los metadatos codificados identifican la fuente (ver 04_tabular_baselines.py).
  D. La cross-attention operaba sobre secuencias de longitud 1 (independiente de la consulta).
  E. "Correlation loss" no implementada; pérdida auxiliar de edad fijada a 0.
  F. Mixup aplicado solo a imagen/etiqueta, no a metadatos.
  G. Imágenes corruptas sustituidas por tensores de ceros.
  H. Recalcula Wilcoxon de la Tabla de la versión enviada (V y p exactos / aproximados).
  I. Métricas de las ejecuciones antiguas N_EXP_0-4 (versión original) y N_EXP_20-24 (réplica).
Requiere torch solo para el punto D (si no está, se omite).
"""
import itertools
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, ROOT, load_unified  # noqa: E402

CK = ROOT / "checkpoints_v3"
SRC = ROOT / "legacy" / "train_v3.py"


def old_metrics(path: Path):
    txt = path.read_text(errors="ignore")
    out = {}
    for key, pat in [("accuracy", r"^Accuracy:\s+([\d.]+)"), ("balanced_accuracy", r"Balanced Accuracy:\s+([\d.]+)"),
                     ("auc", r"AUC-ROC:\s+([\d.]+)"), ("kappa", r"Kappa[^:]*:\s+([\d.]+)"),
                     ("sensitivity", r"Sensitivity[^:]*:\s+([\d.]+)|Recall[^:]*:\s+([\d.]+)"),
                     ("specificity", r"Specificity[^:]*:\s+([\d.]+)"), ("f1", r"F1[^:]*:\s+([\d.]+)")]:
        m = re.search(pat, txt, re.M)
        if m:
            out[key] = float([g for g in m.groups() if g][0])
    return out


def exact_wilcoxon_p(V, n=5):
    """p exacto unilateral P(V >= v) sin empates."""
    sums = [sum(c) for r in range(n + 1) for c in itertools.combinations(range(1, n + 1), r)]
    return np.mean(np.array(sums) >= V)


def main():
    L = ["# Auditoría del pipeline original (legacy/train_v3.py)\n"]
    df = load_unified()
    m = df.set_index("image_path")

    # A + B
    L.append("## A-B. Conjunto evaluado y fuga de pacientes\n")
    rows = []
    for d in sorted(CK.glob("N_EXP_*/densenet121_*")):
        vp, tp = d / "val_partition.csv", d / "train_partition.csv"
        if not vp.exists():
            continue
        va, tr = pd.read_csv(vp), pd.read_csv(tp)
        va_ids = m.reindex(va.image_path); tr_ids = m.reindex(tr.image_path)
        shared = set(va_ids.patient_id.dropna()) & set(tr_ids.patient_id.dropna())
        rows.append({"run": f"{d.parent.name}/{d.name}", "val_n": len(va), "val_mis": int((va.label == 0).sum()),
                     "val_miv": int((va.label == 1).sum()), "patients_shared_train_val": len(shared),
                     "val_images_from_shared_patients": int(va_ids.patient_id.isin(shared).sum()),
                     "isic_val_images": int((va_ids.source == "ISIC").sum())})
    a = pd.DataFrame(rows)
    L.append("En train_v3.py el split train/validación usa `patient_level_split(..., seed=42)` con la columna de "
             "paciente detectada en la PRIMERA fila (VRUH -> `NHC`). Las filas ISIC no tienen `NHC`, por lo que cada "
             "imagen ISIC se trató como un paciente distinto. El mejor checkpoint se elige por accuracy en esa "
             "validación y TODAS las métricas, matrices de confusión, curvas ROC y umbrales informados originalmente se "
             "calcularon sobre esa misma validación (428 imágenes: 132 Mis / 296 Miv). No existía test independiente.\n")
    L.append(a.to_string(index=False) + "\n")

    # C
    sp = RESULTS_DIR / "tables" / "source_predictability.csv"
    if sp.exists():
        L.append("## C. Los metadatos identifican la fuente\n")
        L.append(pd.read_csv(sp).round(3).to_string(index=False) + "\n")
        L.append(f"Diámetro ausente: ISIC {df[df.source == 'ISIC'].diam_long.isna().mean():.1%} vs VRUH "
                 f"{df[df.source == 'VRUH'].diam_long.isna().mean():.1%}; segundo diámetro ausente en el "
                 f"{df[df.source == 'ISIC'].diam_short.isna().mean():.0%} de ISIC -> ratio=0.5 imputado. "
                 f"Prevalencia de Mis: ISIC {1 - df[df.source == 'ISIC'].label.mean():.1%} vs VRUH "
                 f"{1 - df[df.source == 'VRUH'].label.mean():.1%}.\n")

    # D
    L.append("## D. Cross-attention con secuencia de longitud 1\n")
    try:
        import torch
        torch.manual_seed(0)
        mha = torch.nn.MultiheadAttention(256, 8, batch_first=True).eval()
        kv = torch.randn(4, 1, 256)
        o1, w1 = mha(torch.randn(4, 1, 256), kv, kv)
        o2, w2 = mha(torch.randn(4, 1, 256), kv, kv)
        L.append(f"Con query/key/value de forma [B,1,256] (como en train_v3.py, líneas `unsqueeze(1)`), los pesos de "
                 f"atención son siempre 1 (min={w1.min().item():.3f}, max={w1.max().item():.3f}) y la salida NO depende "
                 f"de la consulta de metadatos: max |out(q1) - out(q2)| = {(o1 - o2).abs().max().item():.2e}. "
                 f"La 'cross-attention' equivalía a una proyección lineal de las características de imagen.\n")
    except ImportError:
        L.append("(torch no disponible: ejecútese en un nodo de cómputo)\n")

    # E, F, G
    src = SRC.read_text()
    L.append("## E-F-G. Componentes descritos originalmente frente al código\n")
    L.append(f"- 'Correlation loss': aparece en el docstring pero no en el cálculo de la pérdida "
             f"(ocurrencias de 'corr' fuera de comentarios: "
             f"{len([l for l in src.splitlines() if 'corr' in l.lower() and not l.strip().startswith(('#', '-', '*')) and 'correct' not in l.lower() and 'corrupt' not in l.lower()])}).")
    L.append(f"- Pérdida auxiliar de edad: `aux_age_loss = torch.tensor(0.0)` en ambas ramas "
             f"({src.count('aux_age_loss = torch.tensor(0.0)')} ocurrencias); solo el sexo tenía pérdida auxiliar.")
    L.append("- Mixup: `mixup_data` mezcla imágenes y etiquetas; los metadatos de la muestra i se conservan sin "
             "mezclar (comentario en el código: 'solo en imágenes, no en metadatos').")
    L.append("- Edad: se usaba como continua estandarizada Y como bin (no un código 0-20); diámetro: continuo + "
             "categoría (0 = faltante, <5, 5-10, >=10 mm), no '<1, <5, <10, >=10'.")
    L.append("- Estadísticos de normalización calculados sobre train+validación antes de partir.")
    ci = DATA_DIR / "corrupted_images.csv"
    if ci.exists():
        n = len(pd.read_csv(ci))
        L.append(f"- {n} imágenes JPEG/PNG truncadas lanzaban excepción y el Dataset original devolvía un tensor de "
                 f"ceros (imagen negra) sin avisar. En el pipeline nuevo se cargan en modo tolerante.\n")

    # H
    L.append("## H. Recalculo de Wilcoxon (valores por ejecución de la versión enviada)\n")
    img = {"accuracy": [.75, .75, .75, .75, .75], "balanced_accuracy": [.73, .71, .77, .75, .75],
           "sensitivity": [.77, .81, .71, .74, .75], "specificity": [.69, .62, .82, .77, .75],
           "precision": [.85, .83, .90, .88, .87], "kappa": [.43, .43, .47, .46, .46],
           "f1": [.81, .82, .79, .80, .81], "auc": [.79, .80, .82, .81, .81]}
    mm = {"accuracy": [.77, .80, .80, .77, .79], "balanced_accuracy": [.75, .81, .76, .78, .77],
          "sensitivity": [.79, .79, .84, .75, .82], "specificity": [.71, .84, .69, .82, .73],
          "precision": [.86, .92, .86, .90, .87], "kappa": [.48, .58, .52, .51, .53],
          "f1": [.82, .85, .85, .82, .84], "auc": [.81, .87, .83, .87, .81]}
    rows = []
    for k in img:
        d = np.round(np.array(mm[k]) - np.array(img[k]), 4)
        nz = d[d != 0]
        r = stats.rankdata(np.abs(nz))
        V = r[nz > 0].sum()
        try:
            p_exact = stats.wilcoxon(nz, alternative="greater", mode="exact").pvalue if len(nz) else np.nan
        except TypeError:
            p_exact = exact_wilcoxon_p(V, len(nz))
        p_approx = stats.wilcoxon(nz, alternative="greater", correction=True, mode="approx").pvalue if len(nz) else np.nan
        rows.append({"metric": k, "diffs": d.tolist(), "V": V, "n_nonzero": len(nz), "ties": len(nz) != len(np.unique(np.abs(nz))),
                     "p_exact_one_sided": round(p_exact, 3), "p_normal_approx_cc": round(p_approx, 3)})
    h = pd.DataFrame(rows)
    L.append(h.to_string(index=False) + "\n")
    L.append("Con n=5 el p unilateral mínimo posible es 1/32 = .031. Varias V publicadas no coinciden con las "
             "diferencias de la tabla redondeada (p. ej., especificidad). Con valores sin redondear (archivos "
             "metrics_report), la AUC de la ejecución 4 es 0.8142 vs 0.8062, de modo que V=15 para AUC es correcto.\n")

    # I
    L.append("## I. Métricas de las ejecuciones antiguas (umbral 0.5, sobre la validación)\n")
    rows = []
    for e in list(range(0, 5)) + list(range(20, 25)):
        for d in sorted((CK / f"N_EXP_{e}").glob("densenet121_*")):
            f = d / "metrics_report_threshold_0_50.txt"
            if f.exists():
                rows.append({"batch": "manuscript (N_EXP_0-4)" if e < 5 else "replication (N_EXP_20-24)",
                             "run": e, "model": d.name.replace("densenet121_", ""), **old_metrics(f)})
    i = pd.DataFrame(rows)
    i.to_csv(RESULTS_DIR / "tables" / "legacy_runs_metrics.csv", index=False)
    L.append(i.groupby(["batch", "model"])[["accuracy", "balanced_accuracy", "auc"]].agg(["mean", "std"]).round(3).to_string() + "\n")
    L.append("La réplica de abril de 2026 con el mismo código (N_EXP_20-24) no reproduce la ventaja del modelo "
             "multimodal.\n")
    out = RESULTS_DIR / "audit_legacy.md"
    out.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
