"""Figuras 2 y 4 de la versión de cambios mínimos: matrices de confusión a nivel de lesión (predicciones out-of-fold)."""
import sys, importlib.util
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
sys.path.insert(0, "pipeline")
spec = importlib.util.spec_from_file_location("an", "pipeline/06_analyze.py"); an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
from common import load_unified
OUT = "pipeline/docs/figures_v2_minimo/"
import os; os.makedirs(OUT, exist_ok=True)
df = an.load_all(load_unified())
df = df[df.exp.isin(["image", "concat_meta"])]
U = {e: an.eval_units(df, e, "lesion", repeats=[0, 1, 2]) for e in ["image", "concat_meta"]}
LAB = ["Mis", "Miv"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})


def draw(ax, cm, title, norm):
    m = cm / cm.sum(1, keepdims=True) if norm else cm
    ax.imshow(m, cmap="Blues", vmin=0, vmax=1 if norm else cm.max())
    for i in range(2):
        for j in range(2):
            txt = "%.2f" % m[i, j] if norm else "%d" % cm[i, j]
            ax.text(j, i, txt, ha="center", va="center", fontsize=13,
                    color="white" if (m[i, j] > (0.5 if norm else cm.max() / 2)) else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(LAB); ax.set_yticks([0, 1]); ax.set_yticklabels(LAB)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title, fontsize=11)


fig, axs = plt.subplots(2, 2, figsize=(8.5, 8))
for k, (e, name) in enumerate([("image", "Image-only model"), ("concat_meta", "Multimodal model (concatenation)")]):
    u = U[e]; cm = confusion_matrix(u.label, u.yhat, labels=[0, 1])
    draw(axs[k, 0], cm, name + "\n(absolute)", False); draw(axs[k, 1], cm, name + "\n(normalized)", True)
    print(e, "youden", cm.ravel())
fig.tight_layout(); fig.savefig(OUT + "Figure2_confusion.png", dpi=300); fig.savefig(OUT + "Figure2_confusion.pdf")

fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
u = U["concat_meta"]
for k, t in enumerate([0.50, 0.40, 0.30]):
    cm = confusion_matrix(u.label, (u.prob >= t).astype(int), labels=[0, 1])
    draw(axs[k], cm, "Threshold %.2f" % t, False); print("thr", t, cm.ravel())
fig.tight_layout(); fig.savefig(OUT + "Figure4_thresholds.png", dpi=300); fig.savefig(OUT + "Figure4_thresholds.pdf")
