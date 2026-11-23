#!/usr/bin/env python3
"""Figuras de publicación.

  Figure 2  ROC media (nivel lesión) en los 15 folds de test; banda 2,5-97,5 % para la comparación primaria.
  Figure 3  Calibración (nivel lesión, deciles de probabilidad; predicción OOF promediada entre repeticiones).
  Figure 4  Forest plot de las comparaciones principales (ΔAUC con IC bootstrap por paciente).
  Figure S1 Forest plot de las ablaciones.

Paleta categórica validada (dataviz, modo claro, 3 slots, todas las comprobaciones superadas): azul #2a78d6,
naranja #eb6834, aqua #1baf7a; referencia (solo metadatos) en gris neutro. El aqua queda < 3:1 frente al fondo,
así que lleva codificación secundaria (estilo de línea / marcador) y leyenda.

  python pipeline/22_figures.py      ->  pipeline/results/figures/pub/*.png (300 ppp) y *.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, RUNS_DIR, load_unified  # noqa: E402

OUT = RESULTS_DIR / "figures" / "pub"
OUT.mkdir(parents=True, exist_ok=True)

INK, INK2, GRID, AXIS = "#0b0b0b", "#52514e", "#e6e5e1", "#b9b8b3"
STYLE = {
    "image": dict(label="Image only", color="#2a78d6", ls="-", marker="o"),
    "concat_meta": dict(label="Image + metadata (concatenation)", color="#eb6834", ls=(0, (5, 2)), marker="s"),
    "gated_meta": dict(label="Image + metadata (attention-based fusion)", color="#1baf7a", ls=(0, (6, 2, 1.5, 2)),
                       marker="^"),
    "logreg_meta": dict(label="Metadata only (logistic regression)", color="#8a8984", ls="-", marker="D"),
}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "xtick.major.width": 0.6,
    "ytick.major.width": 0.6, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
})

uni = load_unified().set_index("image_id")


def test_preds(exp):
    rows = []
    for f in sorted((RUNS_DIR / exp).glob("cv_r*_f*/preds.csv")):
        d = pd.read_csv(f, dtype={"image_id": str})
        d = d[d.role == "test"].assign(split=f.parent.name)
        rows.append(d)
    d = pd.concat(rows)
    d["lesion_id"] = uni.loc[d.image_id].lesion_id.values
    return d


def clean_axes(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


# ------------------------------------------------------------------------------------------------
# Figure 2: ROC media por fold
# ------------------------------------------------------------------------------------------------
grid = np.linspace(0, 1, 201)
preds = {e: test_preds(e) for e in STYLE}
fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, zorder=1)
for e, st in STYLE.items():
    tprs, aucs = [], []
    for s, g in preds[e].groupby("split"):
        les = g.groupby("lesion_id").agg(prob=("prob", "mean"), label=("label", "first"))
        fpr, tpr, _ = roc_curve(les.label, les.prob)
        t = np.interp(grid, fpr, tpr); t[0] = 0
        tprs.append(t); aucs.append(roc_auc_score(les.label, les.prob))
    tprs = np.array(tprs)
    if e in ("image", "concat_meta"):
        ax.fill_between(grid, np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), color=st["color"],
                        alpha=0.12, lw=0, zorder=2)
    ax.plot(grid, tprs.mean(0), color=st["color"], ls=st["ls"], lw=1.7 if e != "logreg_meta" else 1.3, zorder=3,
            label=f"{st['label']}\nAUC {np.mean(aucs):.3f} (SD {np.std(aucs, ddof=1):.3f})")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
ax.set_xlabel("1 − specificity"); ax.set_ylabel("Sensitivity (invasive melanoma)")
ax.legend(loc="lower right", fontsize=7, handlelength=2.6, labelspacing=0.7, borderaxespad=0.2)
clean_axes(ax)
save(fig, "Figure2_ROC")

# ------------------------------------------------------------------------------------------------
# Figure 3: calibración (OOF promediada entre repeticiones, nivel lesión, deciles)
# ------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(4.4, 4.4))
ax.plot([0, 1], [0, 1], color=AXIS, lw=0.8, zorder=1)
ax.text(0.17, 0.12, "Perfect calibration", rotation=45, color=INK2, fontsize=7, ha="center", va="center")
for e, st in STYLE.items():
    d = preds[e]
    img = d.groupby("image_id").agg(prob=("prob", "mean"), label=("label", "first"), lesion_id=("lesion_id", "first"))
    les = img.groupby("lesion_id").agg(prob=("prob", "mean"), label=("label", "first"))
    q = pd.qcut(les.prob, 10, duplicates="drop")
    cal = les.groupby(q, observed=True).agg(p=("prob", "mean"), o=("label", "mean"))
    ax.plot(cal.p, cal.o, color=st["color"], lw=1.3, marker=st["marker"], ms=5, mec="white", mew=0.8,
            label=st["label"], zorder=3)
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01); ax.set_aspect("equal")
ax.set_xlabel("Predicted probability of invasive melanoma"); ax.set_ylabel("Observed proportion of invasive melanoma")
ax.legend(loc="lower right", fontsize=7, handlelength=2.2, labelspacing=0.6, borderaxespad=0.2)
clean_axes(ax)
save(fig, "Figure3_calibration")

# ------------------------------------------------------------------------------------------------
# Forest plots
# ------------------------------------------------------------------------------------------------
cmp = pd.read_csv(RESULTS_DIR / "tables" / "paired_comparisons.csv")
cmp = cmp[cmp.level == "lesion"]


def row(A, B, sub="all"):
    r = cmp[(cmp.A == A) & (cmp.B == B) & (cmp.subgroup == sub)]
    assert len(r) == 1, (A, B, sub, len(r))
    return r.iloc[0]


def forest(groups, name, width=7.2, highlight=None, xlabel="Difference in AUC (95% CI)"):
    items = []
    for gname, rows in groups:
        items.append(("header", gname, None))
        for lab, A, B, sub in rows:
            items.append(("row", lab, row(A, B, sub)))
    n = len(items)
    fig, ax = plt.subplots(figsize=(width, 0.24 * n + 0.75))
    y = np.arange(n)[::-1]
    xs = [r.auc_diff_lo for k, _, r in items if k == "row"] + [r.auc_diff_hi for k, _, r in items if k == "row"]
    xmin, xmax = min(xs + [0]) - 0.01, max(xs + [0]) + 0.01
    ax.axvline(0, color=AXIS, lw=0.9, zorder=1)
    labels = []
    for yi, (k, lab, r) in zip(y, items):
        if k == "header":
            labels.append((yi, lab, True))
            continue
        hl = highlight is not None and (r.A, r.B, r.subgroup) == highlight
        col = "#2a78d6" if hl else INK2
        ax.plot([r.auc_diff_lo, r.auc_diff_hi], [yi, yi], color=col, lw=1.2, zorder=2, solid_capstyle="round")
        ax.plot(r.auc_diff, yi, marker="o", ms=5.5 if hl else 4.5, color=col, mec="white", mew=0.8, zorder=3)
        labels.append((yi, lab, False))
        txt = f"{r.auc_diff:+.3f} ({r.auc_diff_lo:+.3f} to {r.auc_diff_hi:+.3f})".replace("-", "−")
        ax.text(1.01, yi, txt, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=7.5,
                color=INK if hl else INK2)
    ax.set_yticks([])
    for yi, lab, header in labels:
        ax.text(-0.01, yi, lab, transform=ax.get_yaxis_transform(), va="center", ha="right",
                fontsize=8 if header else 7.5, fontweight="bold" if header else "normal",
                color=INK if header else INK2)
    ax.text(1.01, n - 0.2, "Difference (95% CI)", transform=ax.get_yaxis_transform(), va="bottom", ha="left",
            fontsize=7.5, fontweight="bold", color=INK)
    ax.set_ylim(-0.7, n - 0.3)
    ax.set_xlim(xmin, xmax)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=2.5)
    save(fig, name)


forest([
    ("Primary comparison: image + metadata (concatenation) vs image only", [
        ("All lesions", "concat_meta", "image", "all"),
        ("VRUH lesions", "concat_meta", "image", "VRUH"),
        ("ISIC lesions", "concat_meta", "image", "ISIC"),
        ("Mis vs thin Miv (<1.0 mm)", "concat_meta", "image", "Mis_vs_thin"),
        ("Mis vs thick Miv (≥1.0 mm)", "concat_meta", "image", "Mis_vs_thick")]),
    ("Fusion architecture", [
        ("Attention-based fusion vs image only", "gated_meta", "image", "all"),
        ("Attention-based fusion vs concatenation", "gated_meta", "concat_meta", "all")]),
    ("Source confounding", [
        ("Image + source vs image only", "image_source", "image", "all"),
        ("Image + metadata + source vs image + source", "concat_meta_source", "image_source", "all"),
        ("Image + metadata vs image + source", "concat_meta", "image_source", "all"),
        ("Image + metadata without diameter vs image only", "concat_drop_diam", "image", "all"),
        ("VRUH only: concatenation vs image only", "concat_meta_vruh", "image_vruh", "all"),
        ("VRUH only: attention-based fusion vs image only", "gated_meta_vruh", "image_vruh", "all")]),
    ("Control", [
        ("Image + metadata vs image + permuted metadata", "concat_meta", "concat_meta_permtrain", "all")]),
], "Figure4_forest", highlight=("concat_meta", "image", "all"))

forest([
    ("Single variables (concatenation)", [
        ("Age only", "concat_only_age", "image", "all"), ("Sex only", "concat_only_sex", "image", "all"),
        ("Anatomical site only", "concat_only_site", "image", "all"),
        ("Diameter only", "concat_only_diam", "image", "all")]),
    ("Leave one variable out (concatenation)", [
        ("Without age", "concat_drop_age", "image", "all"), ("Without sex", "concat_drop_sex", "image", "all"),
        ("Without anatomical site", "concat_drop_site", "image", "all"),
        ("Without diameter†", "concat_drop_diam", "image", "all")]),
    ("Metadata encoding (concatenation)", [
        ("Discretized age and diameter", "concat_meta_binned", "image", "all"),
        ("Imputation without missingness indicators", "concat_meta_impute", "image", "all"),
        ("Diameter ratio with indicator", "concat_meta_ratio", "image", "all"),
        ("Original coding†", "concat_meta_legacy", "image", "all")]),
    ("Attention-based fusion components", [
        ("Without gate", "gated_nogate", "image", "all"), ("Without cross-attention", "gated_noxattn", "image", "all"),
        ("Without auxiliary heads", "gated_noaux", "image", "all"), ("Without Mixup", "gated_nomixup", "image", "all"),
        ("Cross-entropy instead of focal loss", "gated_ce", "image", "all")]),
    ("Training recipe", [
        ("Image only without Mixup", "image_nomixup", "image", "all"),
        ("Image only with cross-entropy", "image_ce", "image", "all"),
        ("Concatenation without Mixup", "concat_nomixup", "image", "all"),
        ("Concatenation with cross-entropy", "concat_ce", "image", "all")]),
], "FigureS1_ablations", xlabel="Difference in AUC vs image only (95% CI)")
print("OK", OUT)
