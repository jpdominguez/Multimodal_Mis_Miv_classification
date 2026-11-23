"""Tabla 3 (versión de cambios mínimos): métricas por lesión con los umbrales fijos 0,50 / 0,40 / 0,30 (predicciones out-of-fold, IC bootstrap por paciente)."""
import sys, importlib.util, numpy as np
sys.path.insert(0, "pipeline")
spec = importlib.util.spec_from_file_location("an", "pipeline/06_analyze.py"); an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
from common import load_unified
from sklearn.metrics import cohen_kappa_score, confusion_matrix
meta = load_unified()
df = an.load_all(meta)
df = df[df.exp.isin(["image", "concat_meta", "gated_meta"])]
B = 2000
for exp in ["image", "concat_meta", "gated_meta"]:
    u = an.eval_units(df, exp, "lesion", repeats=[0, 1, 2])
    y, p = u.label.values, u.prob.values
    idx = list(an.cluster_boot_idx(u.patient_id.values, B, seed=0))
    for t in [0.5, 0.4, 0.3]:
        def m(yy, pp):
            yh = (pp >= t).astype(int); tn, fp, fn, tp = confusion_matrix(yy, yh, labels=[0, 1]).ravel()
            se, sp = tp/(tp+fn), tn/(tn+fp)
            return np.array([(tp+tn)/len(yy), (se+sp)/2, se, sp, tp/(tp+fp) if tp+fp else np.nan, cohen_kappa_score(yy, yh), 2*tp/(2*tp+fp+fn), tp, fp, tn, fn])
        est = m(y, p); bs = np.array([m(y[i], p[i]) for i in idx])
        lo, hi = np.nanpercentile(bs, 2.5, axis=0), np.nanpercentile(bs, 97.5, axis=0)
        names = ["acc", "bacc", "sens", "spec", "ppv", "kappa", "f1"]
        print(exp, t, " ".join("%s=%.3f(%.3f-%.3f)" % (n, est[k], lo[k], hi[k]) for k, n in enumerate(names)), "TP/FP/TN/FN=%d/%d/%d/%d" % tuple(est[7:]))
