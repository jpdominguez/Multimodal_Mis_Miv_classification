#!/usr/bin/env python3
"""
Paso 2 - Búsqueda de duplicados / casi-duplicados (CF menor 1).

Hash perceptual (pHash, DCT 32x32 -> 64 bits) + dHash sobre la caché 320x320.
Informa de:
  - pares VRUH <-> ISIC con distancia de Hamming <= --thr-cross (posibles duplicados entre fuentes)
  - pares de PACIENTES DISTINTOS dentro de la misma fuente con distancia <= --thr-within
    (posible misma lesión con distinto identificador -> riesgo de fuga entre particiones)
Genera un CSV y una figura con los pares más cercanos para revisión visual.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA_DIR, RESULTS_DIR, REV, load_unified  # noqa: E402


def phash(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((32, 32), Image.LANCZOS), dtype=np.float64)
    d = dctn(g, norm="ortho")[:8, :8]
    return (d > np.median(d.ravel()[1:])).ravel()


def dhash(img: Image.Image) -> np.ndarray:
    g = np.asarray(img.convert("L").resize((9, 8), Image.LANCZOS), dtype=np.float64)
    return (g[:, 1:] > g[:, :-1]).ravel()


def hamming_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A.astype(np.uint8); B = B.astype(np.uint8)
    return (A[:, None, :] != B[None, :, :]).sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thr-cross", type=int, default=10)
    ap.add_argument("--thr-within", type=int, default=6)
    args = ap.parse_args()
    df = load_unified()
    ph, dh = [], []
    for p in df.cache_path:
        with Image.open(REV / p) as im:
            ph.append(phash(im)); dh.append(dhash(im))
    ph, dh = np.array(ph), np.array(dh)
    rows = []
    v = np.where(df.source == "VRUH")[0]; i = np.where(df.source == "ISIC")[0]
    D = hamming_matrix(ph[v], ph[i]); Dd = hamming_matrix(dh[v], dh[i])
    for a, b in zip(*np.where(D <= args.thr_cross)):
        rows.append(("cross_source", df.image_id[v[a]], df.image_id[i[b]], int(D[a, b]), int(Dd[a, b])))
    for src_idx in (v, i):
        Dw = hamming_matrix(ph[src_idx], ph[src_idx]); Ddw = hamming_matrix(dh[src_idx], dh[src_idx])
        pid = df.patient_id.values[src_idx]
        for a, b in zip(*np.where(np.triu(Dw <= args.thr_within, 1))):
            if pid[a] != pid[b]:
                rows.append(("within_source_diff_patient", df.image_id[src_idx[a]], df.image_id[src_idx[b]],
                             int(Dw[a, b]), int(Ddw[a, b])))
    out = pd.DataFrame(rows, columns=["type", "image_a", "image_b", "phash_dist", "dhash_dist"])
    meta = df.set_index("image_id")[["patient_id", "lesion_id", "class3", "image_name"]]
    out = out.join(meta.add_suffix("_a"), on="image_a").join(meta.add_suffix("_b"), on="image_b")
    out = out.sort_values(["type", "phash_dist", "dhash_dist"])
    out.to_csv(DATA_DIR / "duplicate_candidates.csv", index=False)
    print(f"Mínima distancia pHash VRUH-ISIC: {D.min()} (umbral {args.thr_cross})")
    print(out.groupby("type").size().to_string() if len(out) else "Sin candidatos")
    print("Candidatos con pHash<=4 y dHash<=8:",
          ((out.phash_dist <= 4) & (out.dhash_dist <= 8)).groupby(out.type).sum().to_dict() if len(out) else 0)

    # Figura de los 24 pares más cercanos
    top = out.head(24)
    if len(top):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cache = df.set_index("image_id").cache_path
        fig, axes = plt.subplots(len(top), 2, figsize=(4, 2 * len(top)))
        axes = np.atleast_2d(axes)
        for r, (_, row) in enumerate(top.iterrows()):
            for c, key in enumerate(["image_a", "image_b"]):
                axes[r, c].imshow(Image.open(REV / cache[row[key]]))
                axes[r, c].set_title(f"{row[key]} {row['patient_id_' + key[-1]]}\nd={row.phash_dist}/{row.dhash_dist}",
                                     fontsize=6)
                axes[r, c].axis("off")
        fig.tight_layout()
        (RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
        fig.savefig(RESULTS_DIR / "figures" / "duplicate_candidates_top.png", dpi=110)


if __name__ == "__main__":
    main()
