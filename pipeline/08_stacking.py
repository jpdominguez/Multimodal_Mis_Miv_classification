#!/usr/bin/env python3
"""
Paso exploratorio 1 (y 3) - Fusión tardía (stacking) de la predicción de imagen con metadatos.
Las reglas de la puerta se fijaron antes de ver resultados.

Para cada partición de CV (15):
  * Entradas: runs/image/<split>/preds.csv y runs/<modelo_meta>/<split>/preds.csv (val y test).
  * El combinador se ajusta SOLO con la validación interna de la partición.
  * Puerta (solo validación): predicción cross-fitted dentro de la validación (5 folds agrupados por
    paciente) -> AUC del stacker vs AUC de S0 (mismo cross-fitting con solo logit p_img, mismos folds).
  * Test: combinador ajustado con toda la validación, aplicado una vez. Se GUARDA pero NO se evalúa
    ni se imprime aquí (se abre solo en el paso 5).

Variantes:
  S1         LR sobre [logit p_img, logit p_meta]
  S2         LR (L2, C por CV interna) sobre [logit p_img] + metadatos codificados
  S3         S2 + interacciones logit p_img x {age_z, diam_log_z, diam_missing}
  S1_src     S1 + indicador de fuente                       (control de confusión)
  S1_nodiam  S1 con logreg_meta_nodiam                      (control de confusión)
  S1_perm    S1 con p_meta permutado entre imágenes          (control negativo)
  --meta <exp> cambia el modelo de metadatos de S1/S1_src/S1_perm (paso 3: p. ej. hgb_meta).

Uso:   python pipeline/08_stacking.py [--meta logreg_meta] [--variants S1,S2,...]
Salida: pipeline/runs_exploratory/stack_<variante>[_<meta>]/<split>/preds.csv
        pipeline/results/tables/exploratory_gate_<step>.csv (por partición)
        pipeline/results/tables/exploratory_log.csv (una fila por variante; se acumula)
"""
import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REV, RESULTS_DIR, RUNS_DIR, SPLITS_CSV, MetaEncoder, load_split, load_unified  # noqa: E402

EXP_DIR = REV / "runs_exploratory"
TAB = RESULTS_DIR / "tables"
META4 = ["age", "sex", "site", "diam"]
GATE_MEAN, GATE_POS = 0.005, 10        # regla 3 (puerta en validación)


def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, p) if len(np.unique(y)) == 2 else np.nan


def read_preds(exp, split):
    f = RUNS_DIR / exp / split / "preds.csv"
    if not f.exists():                       # modelos de metadatos del paso 3 (09_meta_models.py)
        f = EXP_DIR / exp / split / "preds.csv"
    d = pd.read_csv(f, dtype={"image_id": str})
    return {r: g.set_index("image_id").prob for r, g in d.groupby("role")}


def make_model(kind):
    if kind == "small":   # 2-4 entradas: casi sin regularización
        return LogisticRegression(C=1.0, max_iter=5000)
    return LogisticRegressionCV(Cs=[0.01, 0.03, 0.1, 0.3, 1, 3], cv=5, scoring="roc_auc", max_iter=5000)


def features(variant, d, p_img, p_meta, enc):
    """d: filas del dataset unificado; p_img, p_meta: arrays alineados con d."""
    li = logit(p_img)[:, None]
    if variant.startswith("S1"):
        X = np.hstack([li, logit(p_meta)[:, None]])
        if variant == "S1_src":
            X = np.hstack([X, (d.source.values == "ISIC").astype(float)[:, None]])
        return X, "small"
    M = enc.transform(d)
    X = np.hstack([li, M])
    if variant == "S3":
        idx = [enc.names.index(n) for n in ["age_z", "diam_log_z", "diam_missing"] if n in enc.names]
        X = np.hstack([X, li * M[:, idx]])
    return X, "cv"


def run_split(split, variant, meta_exp, df, seed, image_exp="image"):
    sp = load_split(split)
    pi, pm = read_preds(image_exp, split), read_preds(meta_exp, split)
    val_ids = [i for i in sp["val"] if i in pi["val"].index]
    test_ids = [i for i in sp["test"] if i in pi["test"].index]
    dv, dt = df.loc[val_ids], df.loc[test_ids]
    yv = dv.label.values
    piv, pit = pi["val"].loc[val_ids].values, pi["test"].loc[test_ids].values
    pmv, pmt = pm["val"].loc[val_ids].values, pm["test"].loc[test_ids].values
    if variant == "S1_perm":
        rng = np.random.default_rng(seed)
        pmv, pmt = pmv[rng.permutation(len(pmv))], pmt[rng.permutation(len(pmt))]
    enc = MetaEncoder(META4, missing="indicator").fit(df.loc[sp["train"]])

    Xv, kind = features(variant, dv, piv, pmv, enc)
    Xt, _ = features(variant, dt, pit, pmt, enc)

    # --- puerta: cross-fitting dentro de la validación, agrupado por paciente ---
    # Referencia S0 = mismo procedimiento con solo logit p_img (mismos folds): al juntar predicciones de 5
    # modelos con escalas distintas la AUC agregada baja aunque el ranking de cada fold no cambie, así que
    # comparar con p_img crudo penaliza a cualquier stacker (lo delató el control S1_perm: -0,025).
    cf, cf0, cf0s = (np.full(len(yv), np.nan) for _ in range(3))
    X0 = logit(piv)[:, None]
    X0s = np.hstack([X0, (dv.source.values == "ISIC").astype(float)[:, None]])   # imagen + fuente
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed % (2 ** 31))
    for tr, te in skf.split(Xv, yv, groups=dv.patient_id.values):
        cf[te] = make_model(kind).fit(Xv[tr], yv[tr]).predict_proba(Xv[te])[:, 1]
        cf0[te] = make_model("small").fit(X0[tr], yv[tr]).predict_proba(X0[te])[:, 1]
        cf0s[te] = make_model("small").fit(X0s[tr], yv[tr]).predict_proba(X0s[te])[:, 1]
    res = {"split": split, "n_val": len(yv), "auc_val_img": auc(yv, piv), "auc_val_S0": auc(yv, cf0),
           "auc_val_stack": auc(yv, cf)}
    res["delta_vs_raw"] = res["auc_val_stack"] - res["auc_val_img"]
    res["delta"] = res["auc_val_stack"] - res["auc_val_S0"]      # puerta
    # Regla 5 (confusión por fuente), también solo en validación:
    res["delta_vs_S0src"] = res["auc_val_stack"] - auc(yv, cf0s)   # ¿aporta más allá de imagen + fuente?
    for src in ["VRUH", "ISIC"]:                                    # ¿aporta dentro de cada fuente?
        k = dv.source.values == src
        res[f"delta_{src}"] = auc(yv[k], cf[k]) - auc(yv[k], cf0[k])

    # --- test: ajuste con toda la validación; se guarda sin evaluar ---
    m = make_model(kind).fit(Xv, yv)
    ptest = m.predict_proba(Xt)[:, 1]
    return res, pd.concat([
        pd.DataFrame({"image_id": val_ids, "role": "val", "label": yv, "prob": cf, "prob_meta_permuted": np.nan}),
        pd.DataFrame({"image_id": test_ids, "role": "test", "label": dt.label.values, "prob": ptest,
                      "prob_meta_permuted": np.nan})])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="logreg_meta")
    ap.add_argument("--variants", default="S1,S2,S3,S1_src,S1_nodiam,S1_perm")
    ap.add_argument("--step", default="1")
    ap.add_argument("--image", default="image", help="modelo de imagen de referencia (p. ej. image_vruh)")
    args = ap.parse_args()
    df = load_unified().set_index("image_id", drop=False)
    splits = sorted(s for s in pd.read_csv(SPLITS_CSV, usecols=["split_id"]).split_id.unique() if s.startswith("cv_"))
    TAB.mkdir(parents=True, exist_ok=True)

    gate_rows, log_rows = [], []
    for v in args.variants.split(","):
        meta_exp = "logreg_meta_nodiam" if v == "S1_nodiam" else args.meta
        name = f"stack_{v}" + ("" if meta_exp in ("logreg_meta", "logreg_meta_nodiam") else f"_{meta_exp}") \
            + ("" if args.image == "image" else f"_on_{args.image}")
        rows = []
        for k, s in enumerate(splits):
            res, preds = run_split(s, v, meta_exp, df, seed=1000 + k, image_exp=args.image)
            out = EXP_DIR / name / s
            out.mkdir(parents=True, exist_ok=True)
            preds.to_csv(out / "preds.csv", index=False)
            rows.append({"variant": name, "meta_model": meta_exp, **res})
        g = pd.DataFrame(rows)
        gate_rows.append(g)
        passed = (g.delta.mean() >= GATE_MEAN) and ((g.delta > 0).sum() >= GATE_POS)
        log_rows.append({"date": str(date.today()), "step": args.step, "variant": name, "meta_model": meta_exp,
                         "n_splits": len(g), "val_auc_img_mean": g.auc_val_img.mean(),
                         "val_auc_S0_mean": g.auc_val_S0.mean(), "val_auc_stack_mean": g.auc_val_stack.mean(),
                         "val_delta_vs_raw_mean": g.delta_vs_raw.mean(), "val_delta_mean": g.delta.mean(),
                         "val_delta_median": g.delta.median(), "val_delta_n_positive": int((g.delta > 0).sum()),
                         "passes_gate": bool(passed),
                         "val_delta_vs_S0src_mean": g.delta_vs_S0src.mean(),
                         "val_delta_vs_S0src_n_positive": int((g.delta_vs_S0src > 0).sum()),
                         "val_delta_VRUH_mean": g.delta_VRUH.mean(), "val_delta_VRUH_n_positive": int((g.delta_VRUH > 0).sum()),
                         "val_delta_ISIC_mean": g.delta_ISIC.mean(), "val_delta_ISIC_n_positive": int((g.delta_ISIC > 0).sum()),
                         "role": "control" if v in ("S1_src", "S1_nodiam", "S1_perm") else "candidate"})
        print(f"{name:28s} ΔAUC val media {g.delta.mean():+.4f} (mediana {g.delta.median():+.4f}; "
              f"{(g.delta > 0).sum()}/{len(g)} positivas) -> {'PASA' if passed else 'no pasa'} | "
              f"vs img+fuente {g.delta_vs_S0src.mean():+.4f} ({(g.delta_vs_S0src > 0).sum()}/15) | "
              f"VRUH {g.delta_VRUH.mean():+.4f} ({(g.delta_VRUH > 0).sum()}/15) | "
              f"ISIC {g.delta_ISIC.mean():+.4f} ({(g.delta_ISIC > 0).sum()}/15)", flush=True)

    pd.concat(gate_rows).to_csv(TAB / f"exploratory_gate_step{args.step}_{args.meta}_{args.image}.csv", index=False)
    logf = TAB / "exploratory_log.csv"
    log = pd.DataFrame(log_rows)
    if logf.exists():
        old = pd.read_csv(logf)
        old = old[~old.variant.isin(log.variant)]   # re-ejecutar una variante sustituye su fila
        log = pd.concat([old, log])
    log.to_csv(logf, index=False)
    print(f"\nRegistro: {logf}\nPredicciones (test sin evaluar): {EXP_DIR}")


if __name__ == "__main__":
    main()
