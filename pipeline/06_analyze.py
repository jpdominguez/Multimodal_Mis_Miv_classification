#!/usr/bin/env python3
"""
Paso 6 - Análisis estadístico principal.

Para cada experimento:
  * Umbral de Youden calculado en la VALIDACIÓN interna de cada partición y aplicado a su test.
  * Predicción out-of-fold por imagen = media de las repeticiones de CV (cada imagen se predice
    con modelos que nunca la vieron); clase = voto mayoritario de las predicciones umbralizadas.
  * Nivel lesión (análisis principal) = media de las imágenes de la lesión.
  * IC 95 % por bootstrap de PACIENTES (2000 réplicas), diferencias emparejadas con las mismas
    réplicas, p bootstrap, DeLong emparejado (AUC) y McNemar exacto; Holm en métricas secundarias.
  * Por fuente, subgrupos Mis vs Miv<1 mm y Mis vs Miv>=1 mm, calibración, control por permutación
    de metadatos en test, LOSO y resumen por fold.

Uso:  python pipeline/06_analyze.py [--boot 2000]
Salida: pipeline/results/{tables,figures}/ y pipeline/results/SUMMARY.md
"""
import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS_DIR, RUNS_DIR, binary_metrics, load_unified, youden_threshold  # noqa: E402
from experiments import EXPERIMENTS, TABULAR  # noqa: E402

warnings.filterwarnings("ignore")
TAB = RESULTS_DIR / "tables"
FIG = RESULTS_DIR / "figures"
PRIMARY = ("concat_meta", "image")
SENS_TARGET = 0.90                    # E1: especificidad con sensibilidad fijada en validación
DCA_THRESHOLDS = np.round(np.arange(0.10, 0.401, 0.05), 2)
SECONDARY_METRICS = ["balanced_accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "kappa", "brier"]
DISPLAY = {
    "image": "Image only", "concat_meta": "Image + metadata (concatenation)",
    "gated_meta": "Image + metadata (gated fusion + cross-attention)",
    "concat_drop_diam": "Image + metadata without diameter", "image_source": "Image + source indicator",
    "concat_meta_source": "Image + metadata + source", "concat_meta_permtrain": "Image + permuted metadata",
    "concat_meta_legacy": "Image + metadata, original coding", "logreg_meta": "Metadata only (logistic regression)",
    "logreg_source": "Source only (logistic regression)", "logreg_meta_source": "Metadata + source (logistic regression)",
    "logreg_meta_legacy": "Metadata only, original coding (LR)", "logreg_meta_nodiam": "Metadata without diameter (LR)",
}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_all(meta: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for f in glob.glob(str(RUNS_DIR / "*" / "*" / "preds.csv")):
        p = Path(f)
        d = pd.read_csv(f, dtype={"image_id": str})
        d["exp"], d["split"] = p.parent.parent.name, p.parent.name
        frames.append(d)
    if not frames:
        sys.exit("No hay predicciones en pipeline/runs/")
    df = pd.concat(frames, ignore_index=True)
    df["kind"] = df.split.str.split("_").str[0]
    df["repeat"] = np.where(df.kind == "cv", df.split.str.extract(r"_r(\d+)_")[0],
                            df.split.str.extract(r"_s(\d+)$")[0]).astype(float)
    df["repeat"] = df["repeat"].fillna(0).astype(int)
    df = df.merge(meta[["image_id", "patient_id", "lesion_id", "source", "class3"]], on="image_id", how="left")
    # umbral por partición (validación interna) -> clase en test
    thr = {}
    for (e, s), g in df[df.role == "val"].groupby(["exp", "split"]):
        thr[(e, s)] = youden_threshold(g.label.values, g.prob.values)
    df["thr"] = [thr.get((e, s), 0.5) for e, s in zip(df.exp, df.split)]
    df["yhat"] = (df.prob >= df.thr).astype(int)
    df["yhat05"] = (df.prob >= 0.5).astype(int)
    # Punto de trabajo clínico (prespecificado): mayor umbral con sensibilidad
    # >= 0,90 en la validación de cada partición, aplicado a su test.
    # DCA: probabilidades recalibradas (Platt) en la validación de cada partición, porque el muestreo
    # balanceado del entrenamiento descalibra la prevalencia.
    from sklearn.linear_model import LogisticRegression
    thr90, platt = {}, {}
    for (e, s), g in df[df.role == "val"].groupby(["exp", "split"]):
        y, p = g.label.values, g.prob.values
        pos = np.sort(p[y == 1])[::-1]
        thr90[(e, s)] = pos[int(np.ceil(SENS_TARGET * len(pos))) - 1] if len(pos) else 0.5
        z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))[:, None]
        platt[(e, s)] = LogisticRegression(C=1e6, max_iter=1000).fit(z, y) if len(np.unique(y)) == 2 else None
    df["thr90"] = [thr90.get((e, s), 0.5) for e, s in zip(df.exp, df.split)]
    df["yhat90"] = (df.prob >= df.thr90).astype(int)
    df["prob_cal"] = df.prob
    for (e, s), idx in df.groupby(["exp", "split"]).groups.items():
        m = platt.get((e, s))
        if m is not None:
            p = df.loc[idx, "prob"].values
            df.loc[idx, "prob_cal"] = m.predict_proba(np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1))[:, None])[:, 1]
    return df


# ---------------------------------------------------------------------------
# Agregación
# ---------------------------------------------------------------------------
def oof(df, exp, kind="cv", repeats=None, test_source=None):
    d = df[(df.exp == exp) & (df.kind == kind) & (df.role == "test")]
    if kind == "loso" and test_source:
        d = d[d.source == test_source]
    if repeats is not None:
        d = d[d.repeat.isin(repeats)]
    if d.empty:
        return None
    g = d.groupby("image_id")
    out = g.agg(prob=("prob", "mean"), prob_perm=("prob_meta_permuted", "mean"), vote=("yhat", "mean"),
                vote05=("yhat05", "mean"), thr=("thr", "mean"), vote90=("yhat90", "mean"), thr90=("thr90", "mean"),
                prob_cal=("prob_cal", "mean"), label=("label", "first"),
                patient_id=("patient_id", "first"), lesion_id=("lesion_id", "first"),
                source=("source", "first"), class3=("class3", "first"), n_rep=("repeat", "nunique")).reset_index()
    out["yhat"] = np.where(out.vote == 0.5, (out.prob >= out.thr), out.vote > 0.5).astype(int)
    out["yhat05"] = (out.vote05 > 0.5).astype(int)
    out["yhat90"] = np.where(out.vote90 == 0.5, (out.prob >= out.thr90), out.vote90 > 0.5).astype(int)
    return out


def to_lesion(img: pd.DataFrame) -> pd.DataFrame:
    g = img.groupby("lesion_id")
    extra = {k: (c, "mean") for k, c in [("vote90", "yhat90"), ("prob_cal", "prob_cal")] if c in img}
    out = g.agg(prob=("prob", "mean"), prob_perm=("prob_perm", "mean"), vote=("yhat", "mean"),
                vote05=("yhat05", "mean"), label=("label", "first"), patient_id=("patient_id", "first"),
                source=("source", "first"), class3=("class3", "max"), n_img=("image_id", "size"), **extra).reset_index()
    out["yhat"] = (out.vote >= 0.5).astype(int)
    out["yhat05"] = (out.vote05 >= 0.5).astype(int)
    if "vote90" in out:
        out["yhat90"] = (out.vote90 >= 0.5).astype(int)
    return out.rename(columns={"lesion_id": "unit"})


def eval_units(df, exp, level="lesion", repeats=None, kind="cv", test_source=None, subset=None):
    img = oof(df, exp, kind, repeats, test_source)
    if img is None:
        return None
    u = to_lesion(img) if level == "lesion" else img.rename(columns={"image_id": "unit"})
    if subset is not None:
        u = subset(u)
    return u.reset_index(drop=True)


def subgroup(name):
    return {"all": None,
            "VRUH": lambda u: u[u.source == "VRUH"], "ISIC": lambda u: u[u.source == "ISIC"],
            "Mis_vs_thin": lambda u: u[u.class3.isin([0, 1])], "Mis_vs_thick": lambda u: u[u.class3.isin([0, 2])]}[name]


# ---------------------------------------------------------------------------
# Estadística
# ---------------------------------------------------------------------------
def fast_auc(y, p):
    order = np.argsort(p, kind="mergesort")
    ranks = stats.rankdata(p)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def delong_paired(y, p1, p2):
    """DeLong (Sun & Xu 2014) para dos AUC correlacionadas. Devuelve (auc1, auc2, z, p)."""
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = pos.sum(), neg.sum()

    def comps(p):
        X, Y = p[pos], p[neg]
        tx = stats.rankdata(X); ty = stats.rankdata(Y); tz = stats.rankdata(np.r_[X, Y])
        auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
        v01 = (tz[:m] - tx) / n
        v10 = 1 - (tz[m:] - ty) / m
        return auc, v01, v10

    a1, v01a, v10a = comps(np.asarray(p1)); a2, v01b, v10b = comps(np.asarray(p2))
    sx = np.cov(np.vstack([v01a, v01b])); sy = np.cov(np.vstack([v10a, v10b]))
    s = sx / m + sy / n
    var = s[0, 0] + s[1, 1] - 2 * s[0, 1]
    if var <= 0:
        return a1, a2, 0.0, 1.0
    z = (a1 - a2) / np.sqrt(var)
    return a1, a2, z, 2 * stats.norm.sf(abs(z))


def mcnemar_exact(y, yh1, yh2):
    c1 = yh1 == y; c2 = yh2 == y
    b = int((c1 & ~c2).sum()); c = int((~c1 & c2).sum())
    if b + c == 0:
        return b, c, 1.0
    return b, c, stats.binomtest(min(b, c), b + c, 0.5).pvalue


def metric_vector(y, p, yh):
    tp = ((yh == 1) & (y == 1)).sum(); tn = ((yh == 0) & (y == 0)).sum()
    fp = ((yh == 1) & (y == 0)).sum(); fn = ((yh == 0) & (y == 1)).sum()
    n = len(y)
    sens = tp / (tp + fn) if tp + fn else np.nan; spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan; npv = tn / (tn + fn) if tn + fn else np.nan
    acc = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / n ** 2
    kappa = (acc - pe) / (1 - pe) if pe < 1 else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if (ppv + sens) else np.nan
    return {"auc": fast_auc(y, p), "balanced_accuracy": (sens + spec) / 2, "sensitivity": sens,
            "specificity": spec, "ppv": ppv, "npv": npv, "accuracy": acc, "f1": f1, "kappa": kappa,
            "brier": np.mean((p - y) ** 2)}


def cluster_boot_idx(patients: np.ndarray, B: int, seed=0):
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(patients, return_inverse=True)
    members = [np.where(inv == k)[0] for k in range(len(uniq))]
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[k] for k in pick])


def summarize(u: pd.DataFrame, B: int, use05=False):
    y, p = u.label.values, u.prob.values
    yh = (u.yhat05 if use05 else u.yhat).values
    point = binary_metrics(y, p, yh, calib=True)
    boots = [metric_vector(y[i], p[i], yh[i]) for i in cluster_boot_idx(u.patient_id.values, B)]
    bdf = pd.DataFrame(boots)
    out = {}
    for k in bdf.columns:
        out[k] = point[k]
        out[k + "_lo"], out[k + "_hi"] = np.nanpercentile(bdf[k], [2.5, 97.5])
    for k in ["cal_intercept", "cal_slope", "tp", "tn", "fp", "fn", "n"]:
        out[k] = point[k]
    out["n_patients"] = u.patient_id.nunique()
    return out


def compare(a: pd.DataFrame, b: pd.DataFrame, B: int):
    m = a.merge(b, on="unit", suffixes=("_a", "_b"))
    y = m.label_a.values
    pa, pb = m.prob_a.values, m.prob_b.values
    ya, yb = m.yhat_a.values, m.yhat_b.values
    res = {"n_units": len(m), "n_patients": m.patient_id_a.nunique()}
    pa_point = metric_vector(y, pa, ya); pb_point = metric_vector(y, pb, yb)
    diffs = []
    for i in cluster_boot_idx(m.patient_id_a.values, B, seed=1):
        va = metric_vector(y[i], pa[i], ya[i]); vb = metric_vector(y[i], pb[i], yb[i])
        diffs.append({k: va[k] - vb[k] for k in va})
    dd = pd.DataFrame(diffs)
    for k in dd.columns:
        res[f"{k}_A"], res[f"{k}_B"] = pa_point[k], pb_point[k]
        res[f"{k}_diff"] = pa_point[k] - pb_point[k]
        lo, hi = np.nanpercentile(dd[k], [2.5, 97.5])
        res[f"{k}_diff_lo"], res[f"{k}_diff_hi"] = lo, hi
        col = dd[k].dropna()
        res[f"{k}_p_boot"] = min(1.0, 2 * min((col <= 0).mean(), (col >= 0).mean())) if len(col) else np.nan
    if "yhat90_a" in m:
        a90, b90 = m.yhat90_a.values, m.yhat90_b.values
        sp = lambda yy, h: ((h == 0) & (yy == 0)).sum() / max((yy == 0).sum(), 1)  # noqa: E731
        se = lambda yy, h: ((h == 1) & (yy == 1)).sum() / max((yy == 1).sum(), 1)  # noqa: E731
        res["spec90_A"], res["spec90_B"] = sp(y, a90), sp(y, b90)
        res["sens90_A"], res["sens90_B"] = se(y, a90), se(y, b90)
        res["spec90_diff"] = res["spec90_A"] - res["spec90_B"]
        d90 = [sp(y[i], a90[i]) - sp(y[i], b90[i]) for i in cluster_boot_idx(m.patient_id_a.values, B, seed=3)]
        res["spec90_diff_lo"], res["spec90_diff_hi"] = np.percentile(d90, [2.5, 97.5])
        d90 = np.asarray(d90)
        res["spec90_p_boot"] = min(1.0, 2 * min((d90 <= 0).mean(), (d90 >= 0).mean()))
    _, _, z, p = delong_paired(y, pa, pb)
    res["auc_p_delong"] = p
    b_, c_, pm = mcnemar_exact(y, ya, yb)
    res["mcnemar_A_only_correct"], res["mcnemar_B_only_correct"], res["mcnemar_p"] = b_, c_, pm
    return res


def md(d, index=False):
    """DataFrame -> tabla Markdown (usa tabulate si está instalado)."""
    try:
        return d.to_markdown(index=index)
    except ImportError:
        d = d.reset_index() if index else d
        cols = list(map(str, d.columns))
        rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
        for r in d.itertuples(index=False):
            rows.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r) + " |")
        return "\n".join(rows)


def holm(pvals):
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0
    for rank, idx in enumerate(order):
        running = max(running, (len(p) - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    B = args.boot
    TAB.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    meta = load_unified()
    df = load_all(meta)
    all_exps = sorted(df.exp.unique())
    lines = ["# Resultados (generado por 06_analyze.py)\n"]

    # --- completitud ---
    comp = df[df.role == "test"].groupby(["exp", "kind"]).split.nunique().rename("n_splits").reset_index()
    comp.to_csv(TAB / "completeness.csv", index=False)
    lines.append("## Completitud de ejecuciones\n\n" + md(comp) + "\n")

    def reps_of(e):
        return sorted(df[(df.exp == e) & (df.kind == "cv")].repeat.unique())

    # --- tabla de todos los modelos (lesión e imagen, todas las fuentes) ---
    rows = []
    for e in all_exps:
        for level in ["lesion", "image"]:
            u = eval_units(df, e, level)
            if u is None:
                continue
            s = summarize(u, B)
            s05 = binary_metrics(u.label.values, u.prob.values, u.yhat05.values)
            # AUC de cada repetición por separado (sin el efecto ensemble de promediar repeticiones):
            # comparable entre experimentos de nivel 1 (3 repeticiones) y nivel 2 (1 repetición)
            per_rep = {r: eval_units(df, e, level, repeats=[r]) for r in reps_of(e)}
            per_rep = {r: fast_auc(v.label.values, v.prob.values) for r, v in per_rep.items() if v is not None}
            s["auc_r0"] = per_rep.get(0, np.nan)
            s["auc_mean_per_repeat"] = np.mean(list(per_rep.values())) if per_rep else np.nan
            rows.append({"exp": e, "name": DISPLAY.get(e, e), "level": level, "repeats": len(reps_of(e)),
                         **s, **{f"{k}@0.5": v for k, v in s05.items() if k in
                                 ["sensitivity", "specificity", "balanced_accuracy", "fn", "fp"]}})
    allm = pd.DataFrame(rows)
    allm.to_csv(TAB / "all_models_metrics.csv", index=False)

    def fmt(r, k):
        return f"{r[k]:.3f} ({r[k + '_lo']:.3f}-{r[k + '_hi']:.3f})"

    show = allm[allm.level == "lesion"].copy()
    show["AUC (95% CI)"] = [fmt(r, "auc") for _, r in show.iterrows()]
    show["Sens"] = [fmt(r, "sensitivity") for _, r in show.iterrows()]
    show["Spec"] = [fmt(r, "specificity") for _, r in show.iterrows()]
    show["BalAcc"] = [fmt(r, "balanced_accuracy") for _, r in show.iterrows()]
    show["Brier"] = show.brier.round(3)
    show["Cal. slope"] = show.cal_slope.round(2)
    show["AUC r0"] = show.auc_r0.round(3)
    show["AUC mean/rep"] = show.auc_mean_per_repeat.round(3)
    lines.append("## Todos los modelos, nivel lesión (umbral de Youden fijado en validación)\n\n"
                 "`AUC (95% CI)` usa la predicción OOF promediada entre repeticiones (ensemble de hasta 3 modelos "
                 "por imagen). Para comparar experimentos con distinto número de repeticiones usar `AUC r0` "
                 "(solo repetición 0) o `AUC mean/rep` (media de la AUC OOF de cada repetición).\n\n" +
                 md(show[["exp", "repeats", "n", "AUC (95% CI)", "AUC r0", "AUC mean/rep", "Sens", "Spec", "BalAcc",
                          "Brier", "Cal. slope"]]) + "\n")

    # --- comparaciones emparejadas ---
    pairs = [("concat_meta", "image", "PRIMARY"), ("gated_meta", "image", "secondary"),
             ("gated_meta", "concat_meta", "secondary"), ("concat_drop_diam", "image", "secondary"),
             ("image_source", "image", "confounding"), ("concat_meta_source", "image_source", "confounding"),
             # tras la corrección de metadatos (audit_metadatos.md) diam_missing codifica la fuente: aporte más allá de ella
             ("concat_meta", "image_source", "beyond-source"),
             ("concat_meta", "concat_meta_permtrain", "control"), ("concat_meta_legacy", "image", "confounding"),
             ("concat_meta_legacy", "concat_meta", "confounding"),
             ("logreg_meta", "image", "baseline"), ("logreg_source", "image", "baseline"),
             ("concat_meta_vruh", "image_vruh", "single-source"), ("gated_meta_vruh", "image_vruh", "single-source"),
             ("logreg_meta_vruh", "image_vruh", "single-source")]
    for e in all_exps:
        c = EXPERIMENTS.get(e)
        if c and c["tier"] == 2:
            ref = "gated_meta" if e.startswith("gated_") else "image" if e.startswith("image_") else "concat_meta"
            pairs.append((e, ref, "ablation"))
            if ref != "image":
                pairs.append((e, "image", "ablation"))
    comp_rows = []
    for a, b, role in pairs:
        if a not in all_exps or b not in all_exps:
            continue
        reps = sorted(set(reps_of(a)) & set(reps_of(b))) if a in EXPERIMENTS and b in EXPERIMENTS else None
        for level in ["lesion", "image"]:
            for sg in (["all", "VRUH", "ISIC", "Mis_vs_thin", "Mis_vs_thick"] if level == "lesion" else ["all"]):
                ua = eval_units(df, a, level, reps, subset=subgroup(sg))
                ub = eval_units(df, b, level, reps, subset=subgroup(sg))
                if ua is None or ub is None or ua.label.nunique() < 2:
                    continue
                r = compare(ua, ub, B)
                comp_rows.append({"A": a, "B": b, "role": role, "level": level, "subgroup": sg,
                                  "repeats": "all" if reps is None else ",".join(map(str, reps)), **r})
    cmp = pd.DataFrame(comp_rows)
    # Holm sobre métricas secundarias de la comparación primaria
    m = (cmp.A == PRIMARY[0]) & (cmp.B == PRIMARY[1]) & (cmp.level == "lesion") & (cmp.subgroup == "all")
    if m.any():
        idx = cmp[m].index[0]
        adj = holm([cmp.loc[idx, f"{k}_p_boot"] for k in SECONDARY_METRICS])
        for k, v in zip(SECONDARY_METRICS, adj):
            cmp.loc[idx, f"{k}_p_holm"] = v
    cmp.to_csv(TAB / "paired_comparisons.csv", index=False)

    def cmp_line(r):
        return (f"| {r.A} | {r.B} | {r.role} | {r.subgroup} | {r.n_units} | {r.auc_A:.3f} | {r.auc_B:.3f} | "
                f"{r.auc_diff:+.3f} ({r.auc_diff_lo:+.3f} to {r.auc_diff_hi:+.3f}) | {r.auc_p_boot:.3f} | "
                f"{r.auc_p_delong:.3f} | {r.balanced_accuracy_diff:+.3f} ({r.balanced_accuracy_diff_lo:+.3f} to "
                f"{r.balanced_accuracy_diff_hi:+.3f}) | {r.mcnemar_p:.3f} |")
    hdr = ("| A | B | role | subgroup | n lesions | AUC A | AUC B | ΔAUC (95% CI) | p boot | p DeLong | "
           "ΔBalAcc (95% CI) | p McNemar |\n|" + "---|" * 12)
    sub = cmp[(cmp.level == "lesion")]
    lines.append("## Comparaciones emparejadas (nivel lesión, bootstrap por paciente)\n\n" + hdr + "\n" +
                 "\n".join(cmp_line(r) for r in sub.itertuples()) + "\n")
    if m.any():
        r = cmp.loc[idx]
        lines.append("### Comparación primaria (prespecificada): concat_meta vs image, AUC a nivel lesión\n")
        lines.append(f"- ΔAUC = {r.auc_diff:+.3f} (IC95 {r.auc_diff_lo:+.3f} a {r.auc_diff_hi:+.3f}); "
                     f"p bootstrap = {r.auc_p_boot:.3f}; p DeLong = {r.auc_p_delong:.3f}")
        for k in SECONDARY_METRICS:
            lines.append(f"- {k}: Δ = {r[k + '_diff']:+.3f} ({r[k + '_diff_lo']:+.3f} a {r[k + '_diff_hi']:+.3f}); "
                         f"p Holm = {r[k + '_p_holm']:.3f}")
        lines.append("")

    # --- E1: punto de trabajo clínico (sensibilidad 0,90 fijada en validación) ---
    if "spec90_diff" in cmp:
        e1 = cmp[(cmp.level == "lesion") & cmp.role.isin(["PRIMARY", "secondary", "confounding", "beyond-source",
                                                          "single-source", "control"])]
        e1 = e1[["A", "B", "subgroup", "n_units", "sens90_A", "sens90_B", "spec90_A", "spec90_B", "spec90_diff",
                 "spec90_diff_lo", "spec90_diff_hi", "spec90_p_boot"]]
        e1.to_csv(TAB / "operating_point_sens90.csv", index=False)
        lines.append("## Punto de trabajo clínico: especificidad con sensibilidad 0,90 fijada en validación (nivel lesión)\n\n"
                     + md(e1[e1.subgroup.isin(["all", "VRUH"])].round(3)) + "\n")

    # --- E1: decision curve analysis (probabilidades recalibradas en validación) ---
    def net_benefit(y, p, pt):
        n = len(y); tp = ((p >= pt) & (y == 1)).sum(); fp = ((p >= pt) & (y == 0)).sum()
        return tp / n - fp / n * pt / (1 - pt)
    dca_rows = []
    dca_pairs = [("concat_meta", "image"), ("concat_meta", "image_source"), ("gated_meta", "image"),
                 ("concat_meta_vruh", "image_vruh")]
    for a, b in dca_pairs:
        if a not in all_exps or b not in all_exps:
            continue
        for sg in ["all", "VRUH"]:
            ua = eval_units(df, a, "lesion", subset=subgroup(sg)); ub = eval_units(df, b, "lesion", subset=subgroup(sg))
            mm = ua.merge(ub, on="unit", suffixes=("_a", "_b"))
            y = mm.label_a.values
            for pt in DCA_THRESHOLDS:
                nba, nbb = net_benefit(y, mm.prob_cal_a.values, pt), net_benefit(y, mm.prob_cal_b.values, pt)
                bd = [net_benefit(y[i], mm.prob_cal_a.values[i], pt) - net_benefit(y[i], mm.prob_cal_b.values[i], pt)
                      for i in cluster_boot_idx(mm.patient_id_a.values, min(B, 1000), seed=4)]
                dca_rows.append({"A": a, "B": b, "subgroup": sg, "threshold": pt, "nb_A": nba, "nb_B": nbb,
                                 "nb_treat_all": y.mean() - (1 - y.mean()) * pt / (1 - pt), "delta_nb": nba - nbb,
                                 "delta_lo": np.percentile(bd, 2.5), "delta_hi": np.percentile(bd, 97.5)})
    if dca_rows:
        dca = pd.DataFrame(dca_rows)
        dca.to_csv(TAB / "decision_curve.csv", index=False)
        lines.append("## Decision curve analysis (beneficio neto, nivel lesión, probabilidades recalibradas en validación)\n\n"
                     + md(dca[dca.subgroup == "all"].round(3)) + "\n")

    # --- nivel fold (descriptivo + Wilcoxon de sensibilidad) ---
    fold_rows = []
    for (e, s), g in df[(df.role == "test") & (df.kind == "cv")].groupby(["exp", "split"]):
        les = to_lesion(g.groupby("image_id").agg(prob=("prob", "mean"), prob_perm=("prob_meta_permuted", "mean"),
                                                  yhat=("yhat", "mean"), yhat05=("yhat05", "mean"),
                                                  label=("label", "first"), patient_id=("patient_id", "first"),
                                                  lesion_id=("lesion_id", "first"), source=("source", "first"),
                                                  class3=("class3", "first")).reset_index())
        mv = binary_metrics(les.label.values, les.prob.values, les.yhat.values)
        fold_rows.append({"exp": e, "split": s, **{k: mv[k] for k in ["auc", "balanced_accuracy", "sensitivity",
                                                                       "specificity", "brier"]}})
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(TAB / "fold_level_metrics.csv", index=False)
    fsum = folds.groupby("exp").agg(n_folds=("auc", "size"), auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                                    balacc_mean=("balanced_accuracy", "mean"), balacc_sd=("balanced_accuracy", "std"))
    wil = []
    for a, b, _ in pairs:
        fa = folds[folds.exp == a].set_index("split").auc; fb = folds[folds.exp == b].set_index("split").auc
        common = fa.index.intersection(fb.index)
        if len(common) >= 5:
            d = fa[common] - fb[common]
            w = stats.wilcoxon(d) if (d != 0).any() else None
            wil.append({"A": a, "B": b, "n_folds": len(common), "mean_diff": d.mean(), "sd_diff": d.std(),
                        "n_positive": int((d > 0).sum()), "wilcoxon_p_two_sided": w.pvalue if w else 1.0})
    pd.DataFrame(wil).to_csv(TAB / "fold_level_wilcoxon_sensitivity.csv", index=False)
    lines.append("## Resumen por fold (AUC a nivel lesión en cada fold de test)\n\n" + md(fsum.round(3), index=True) + "\n")
    if wil:
        lines.append("Wilcoxon por fold (análisis de sensibilidad, dos colas):\n\n" +
                     md(pd.DataFrame(wil).round(3)) + "\n")

    # --- control de permutación en test ---
    perm_rows = []
    for e in all_exps:
        u = eval_units(df, e, "lesion")
        if u is None or u.prob_perm.isna().all():
            continue
        y = u.label.values
        d = [fast_auc(y[i], u.prob.values[i]) - fast_auc(y[i], u.prob_perm.values[i])
             for i in cluster_boot_idx(u.patient_id.values, B, 2)]
        perm_rows.append({"exp": e, "auc_intact": fast_auc(y, u.prob.values), "auc_meta_permuted": fast_auc(y, u.prob_perm.values),
                          "drop": fast_auc(y, u.prob.values) - fast_auc(y, u.prob_perm.values),
                          "drop_lo": np.percentile(d, 2.5), "drop_hi": np.percentile(d, 97.5)})
    perm = pd.DataFrame(perm_rows)
    perm.to_csv(TAB / "test_time_metadata_permutation.csv", index=False)
    if len(perm):
        lines.append("## Dependencia de los metadatos: AUC con metadatos de test permutados entre pacientes\n\n" +
                     md(perm.round(3)) + "\n")

    # --- LOSO ---
    loso_rows = []
    for e in [x for x in all_exps if (df[(df.exp == x)].kind == "loso").any()]:
        for src in ["ISIC", "VRUH"]:
            u = eval_units(df, e, "lesion", kind="loso", test_source=src)
            if u is None:
                continue
            s = summarize(u, B)
            loso_rows.append({"exp": e, "train_on": "VRUH" if src == "ISIC" else "ISIC", "test_on": src, **s})
    for a, b in [("concat_meta_loso", "image_loso"), ("gated_meta_loso", "image_loso"), ("logreg_meta", "image_loso")]:
        for src in ["ISIC", "VRUH"]:
            ua = eval_units(df, a, "lesion", kind="loso", test_source=src)
            ub = eval_units(df, b, "lesion", kind="loso", test_source=src)
            if ua is not None and ub is not None:
                r = compare(ua, ub, B)
                loso_rows.append({"exp": f"{a} - {b}", "test_on": src, "auc": r["auc_diff"],
                                  "auc_lo": r["auc_diff_lo"], "auc_hi": r["auc_diff_hi"], "p_delong": r["auc_p_delong"]})
    loso = pd.DataFrame(loso_rows)
    loso.to_csv(TAB / "loso.csv", index=False)
    if len(loso):
        lines.append("## Leave-one-source-out (nivel lesión)\n\n" +
                     md(loso[["exp", "test_on", "auc", "auc_lo", "auc_hi"] +
                             [c for c in ["sensitivity", "specificity", "p_delong"] if c in loso]].round(3)) + "\n")

    # --- partición original (diagnóstico; test = validación usada para early stopping) ---
    leg_rows = []
    for e in [x for x in all_exps if (df[df.exp == x].kind == "legacy").any()]:
        d = df[(df.exp == e) & (df.kind == "legacy") & (df.role == "test")]
        leg_rows.append({"exp": e, "n_images": len(d), "auc_image_level": fast_auc(d.label.values, d.prob.values),
                         **{k: v for k, v in binary_metrics(d.label.values, d.prob.values, d.yhat.values).items()
                            if k in ["sensitivity", "specificity", "balanced_accuracy"]}})
    if leg_rows:
        leg = pd.DataFrame(leg_rows)
        leg.to_csv(TAB / "legacy_split_reproduction.csv", index=False)
        lines.append("## Partición original (N_EXP_20; diagnóstico, nivel imagen, sin IC)\n\n"
                     "Test = validación usada para early stopping, con fuga de pacientes ISIC. Solo para "
                     "comprobar que el pipeline nuevo reproduce las cifras antiguas en esas condiciones.\n\n" +
                     md(leg.round(3)) + "\n")

    # --- gating ---
    gates = []
    for f in glob.glob(str(RUNS_DIR / "gated*" / "*" / "run.json")):
        j = json.load(open(f))
        if j.get("gate_mean_test"):
            gates.append({"exp": j["exp"], "split": j["split"], "w_image": j["gate_mean_test"][0],
                          "w_meta": j["gate_mean_test"][1], "best_epoch": j["best_epoch"]})
    if gates:
        g = pd.DataFrame(gates)
        g.to_csv(TAB / "gate_weights.csv", index=False)
        lines.append("## Pesos medios del gate en test\n\n" + md(g.groupby("exp")[["w_image", "w_meta", "best_epoch"]]
                                                        .mean().round(3), index=True) + "\n")

    figures(df, all_exps)
    (RESULTS_DIR / "SUMMARY.md").write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\nTablas en {TAB}\nFiguras en {FIG}\nResumen en {RESULTS_DIR / 'SUMMARY.md'}")


def figures(df, all_exps):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [e for e in ["image", "concat_meta", "gated_meta", "concat_drop_diam", "logreg_meta"] if e in all_exps]
    colors = dict(zip(main, ["#4C72B0", "#DD8452", "#55A868", "#8172B3", "#937860"]))
    grid = np.linspace(0, 1, 101)

    # ROC media +- banda (percentiles 2.5-97.5 entre los 15 folds), nivel lesión
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for e in main:
        tprs, aucs = [], []
        for s, g in df[(df.exp == e) & (df.kind == "cv") & (df.role == "test")].groupby("split"):
            les = g.groupby("lesion_id").agg(prob=("prob", "mean"), label=("label", "first"))
            fpr, tpr, _ = roc_curve(les.label, les.prob)
            tprs.append(np.interp(grid, fpr, tpr)); tprs[-1][0] = 0; aucs.append(roc_auc_score(les.label, les.prob))
        tprs = np.array(tprs)
        ax.plot(grid, tprs.mean(0), color=colors[e], label=f"{DISPLAY.get(e, e)}\nAUC {np.mean(aucs):.3f} (SD {np.std(aucs, ddof=1):.3f})")
        ax.fill_between(grid, np.percentile(tprs, 2.5, 0), np.percentile(tprs, 97.5, 0), color=colors[e], alpha=0.15)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("1 - Specificity"); ax.set_ylabel("Sensitivity")
    ax.set_title("Mean ROC across 15 test folds (lesion level)\nshaded: 2.5-97.5 percentile band", fontsize=9)
    ax.legend(fontsize=6.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG / "roc_mean_band.png", dpi=200); plt.close(fig)

    # Calibración (nivel lesión, OOF)
    fig, ax = plt.subplots(figsize=(5, 5))
    for e in main:
        u = eval_units(df, e, "lesion")
        q = pd.qcut(u.prob, 10, duplicates="drop")
        cal = u.groupby(q).agg(p=("prob", "mean"), o=("label", "mean"))
        ax.plot(cal.p, cal.o, "o-", ms=3, color=colors[e], label=DISPLAY.get(e, e))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("Predicted probability of invasive melanoma"); ax.set_ylabel("Observed proportion")
    ax.legend(fontsize=6.5); ax.set_title("Calibration (lesion level, deciles)", fontsize=9)
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=200); plt.close(fig)

    # Forest plot de ΔAUC frente a image-only
    cmp = pd.read_csv(TAB / "paired_comparisons.csv")
    f = cmp[(cmp.level == "lesion") & (cmp.subgroup == "all") & (cmp.B.isin(["image", "image_vruh", "image_source"]))]
    if len(f):
        f = f.sort_values("auc_diff")
        fig, ax = plt.subplots(figsize=(6.5, 0.32 * len(f) + 1))
        yy = np.arange(len(f))
        ax.errorbar(f.auc_diff, yy, xerr=[f.auc_diff - f.auc_diff_lo, f.auc_diff_hi - f.auc_diff], fmt="o",
                    color="#333", ms=4, capsize=2)
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks(yy); ax.set_yticklabels([f"{a} vs {b}" for a, b in zip(f.A, f.B)], fontsize=7)
        ax.set_xlabel("ΔAUC (lesion level) with 95% patient-bootstrap CI")
        fig.tight_layout(); fig.savefig(FIG / "forest_delta_auc.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
