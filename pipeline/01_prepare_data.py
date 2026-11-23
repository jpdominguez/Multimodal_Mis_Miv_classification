#!/usr/bin/env python3
"""
Paso 1 - Dataset unificado y auditado.

Genera pipeline/data/dataset_unified.csv con una fila por imagen y:
  image_id, source (VRUH/ISIC), patient_id (prefijado por fuente), lesion_id,
  label (0=Mis, 1=Miv), class3 (0=Mis, 1=Miv<1mm, 2=Miv>=1mm), breslow_mm,
  age (años), sex, site, diam_long, diam_short, image_path, cache_path.

Además:
  - Caché de imágenes redimensionadas a 320x320 (pipeline/data/cache_320/).
  - Tabla 1 descriptiva por fuente y datos faltantes (pipeline/results/tables/).
  - Informe de incoherencias (Breslow del nombre de archivo vs clase, etc.).

Uso:  python pipeline/01_prepare_data.py [--no-cache]
"""
import argparse
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, DATA_DIR, RESULTS_DIR, REV, ROOT, SITES, UNIFIED_CSV  # noqa: E402

VRUH_CSV = ROOT / "reports" / "dataset_procesado_VRUH.csv"
ISIC_CSV = ROOT / "reports" / "dataset_procesado_ISIC.csv"
VRUH_FULL_CSV = ROOT / "reports" / "lista_imagenes_VRUH.csv"


def parse_breslow_from_name(name: str):
    """'1234567-6,5-2.jpg' -> 6.5 ; '7654321-0,0.jpg' -> 0.0 ; None si no se puede."""
    stem = name.rsplit(".", 1)[0]
    # El token de Breslow debe llevar separador decimal ("6,5", "0.0"); un entero suelto es el índice de imagen
    m = re.match(r"^\s*[A-Za-z]*\s*\d+(?:\s*\(\d+\))?\s*-\s*(\d+[.,]+\d+)", stem)
    if not m:
        return None
    tok = re.sub(r"[.,]+", ".", m.group(1))
    try:
        return float(tok)
    except ValueError:
        return None


def clean_id(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if s.endswith(".0"):
        s = s[:-2]
    return s


def build_vruh() -> pd.DataFrame:
    d = pd.read_csv(VRUH_CSV)
    d = d[d.CLASS.notna()].copy()
    out = pd.DataFrame()
    out["image_path"] = d.image_path
    out["source"] = "VRUH"
    out["patient_id"] = "V_" + d.NHC.map(clean_id)
    out["class3"] = d.CLASS.astype(float).astype(int)
    out["breslow_mm"] = d.image_name.map(parse_breslow_from_name)
    out["age"] = pd.to_numeric(d.AGE, errors="coerce")
    out["sex"] = d.GENDER_STD.where(d.GENDER_STD.isin(["MALE", "FEMALE"]))
    out["site"] = d.LOCATION_STD.where(d.LOCATION_STD.isin(SITES))
    # LOCATION_STD sale de LOCATION I (región amplia): planta/palma/uña quedaban como extremidad, mientras
    # que en ISIC 'palms/soles' es ACRAL -> ACRAL era de facto un indicador de ISIC (audit_metadatos.md, punto 2)
    loc2 = d["LOCATION II"].astype(str).str.strip().str.lower()
    out.loc[loc2.isin(["sole", "palm", "nail"]).values, "site"] = "ACRAL"
    d1 = pd.to_numeric(d["DIAMETER 1"], errors="coerce")
    d2 = pd.to_numeric(d["DIAMETER 2"], errors="coerce")
    d1 = d1.where(d1 > 0); d2 = d2.where(d2 > 0)
    out["diam_long"] = np.fmax(d1, d2)
    out["diam_short"] = np.where(d1.notna() & d2.notna(), np.fmin(d1, d2), np.nan)
    # Lesión: carpeta del paciente + token de Breslow + localización + diámetros
    folder = d.image_path.map(lambda p: str(Path(p).parent))
    brs = out.breslow_mm.map(lambda b: "NA" if pd.isna(b) else f"{b:g}")
    out["lesion_id"] = ("VL_" + out.patient_id + "_" + brs + "_" + d["LOCATION II"].fillna("NA").astype(str)
                        + "_" + d1.fillna(-1).astype(str))
    out["_folder"] = folder.values
    out["image_name"] = d.image_name.values
    # Antecedentes (disponibles antes de la biopsia P2). Vienen de la base clínica
    # completa (lista_imagenes_VRUH.csv), emparejada por ruta de imagen.
    full = pd.read_csv(VRUH_FULL_CSV, usecols=["image_path", "Personal history", "Family history"]).drop_duplicates("image_path")
    hx = d[["image_path"]].merge(full, on="image_path", how="left")
    for col, name in [("Personal history", "personal_hx"), ("Family history", "family_hx")]:
        x = pd.to_numeric(hx[col], errors="coerce")
        out[name] = x.where(x.isin([0, 1])).values
    return out.reset_index(drop=True)


def build_isic() -> pd.DataFrame:
    d = pd.read_csv(ISIC_CSV)
    out = pd.DataFrame()
    out["image_path"] = d.image_path
    out["source"] = "ISIC"
    pid = d.patient_id.map(clean_id)
    out["patient_id"] = np.where(pid.notna(), "I_" + pid.fillna(""), "I_img_" + d.isic_id)
    out["class3"] = d.CLASS.astype(float).astype(int)
    out["breslow_mm"] = pd.to_numeric(d.mel_thick_mm, errors="coerce")
    out["age"] = pd.to_numeric(d.AGE, errors="coerce")
    out["sex"] = d.GENDER_STD.where(d.GENDER_STD.isin(["MALE", "FEMALE"]))
    out["site"] = d.LOCATION_STD.where(d.LOCATION_STD.isin(SITES))
    dl = pd.to_numeric(d.clin_size_long_diam_mm, errors="coerce")
    out["diam_long"] = dl.where(dl > 0)
    out["diam_short"] = np.nan
    out["personal_hx"] = np.nan          # no existe en ISIC
    out["family_hx"] = np.nan
    out["lesion_id"] = ("IL_" + out.patient_id + "_" + d.anatom_site_general.fillna("NA").astype(str) + "_"
                        + out.breslow_mm.fillna(-1).astype(str) + "_" + d.diagnosis.fillna("NA").astype(str)
                        + "_" + dl.fillna(-1).astype(str))
    out["_folder"] = ""
    out["image_name"] = d.image_name.values
    return out.reset_index(drop=True)


def _cache_one(args):
    src, dst, size = args
    from PIL import Image, ImageFile, ImageOps
    dst = Path(dst)
    if dst.exists():
        return "ok"
    status = "ok"
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(src) as im:
            im.load()
    except Exception:
        status = "truncated_loaded"
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    try:
        bicubic = getattr(Image, "Resampling", Image).BICUBIC
        with Image.open(src) as im:
            im = ImageOps.exif_transpose(im).convert("RGB").resize((size, size), bicubic)
            dst.parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, quality=95)
        return status
    except Exception as e:  # noqa
        print("ERROR caché", src, e)
        return "failed"


def describe(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(var, level, f):
        r = {"variable": var, "level": level}
        for name, g in [("VRUH", df[df.source == "VRUH"]), ("ISIC", df[df.source == "ISIC"]), ("Total", df)]:
            r[name] = f(g)
        rows.append(r)

    n = lambda g: len(g)  # noqa: E731
    add("Images", "n", lambda g: f"{len(g)}")
    add("Patients", "n", lambda g: f"{g.patient_id.nunique()}")
    add("Lesions (reconstructed)", "n", lambda g: f"{g.lesion_id.nunique()}")
    add("Images per lesion", "median (IQR)", lambda g: "{:.0f} ({:.0f}-{:.0f})".format(
        *np.percentile(g.groupby('lesion_id').size(), [50, 25, 75])))
    for k, lab in [(0, "Melanoma in situ"), (1, "Invasive, Breslow <1.0 mm"), (2, "Invasive, Breslow >=1.0 mm")]:
        add("Class (images)", lab, lambda g, k=k: f"{(g.class3 == k).sum()} ({100 * (g.class3 == k).mean():.1f}%)")
    for k, lab in [(0, "Melanoma in situ"), (1, "Invasive, Breslow <1.0 mm"), (2, "Invasive, Breslow >=1.0 mm")]:
        add("Class (lesions)", lab, lambda g, k=k: f"{(g.groupby('lesion_id').class3.first() == k).sum()}")
    add("Breslow, invasive (mm)", "median (IQR)", lambda g: "{:.2f} ({:.2f}-{:.2f})".format(
        *np.nanpercentile(g[g.class3 > 0].breslow_mm, [50, 25, 75])) if g[g.class3 > 0].breslow_mm.notna().any() else "NA")
    add("Age (years)", "mean (SD)", lambda g: f"{g.age.mean():.1f} ({g.age.std():.1f})")
    add("Age (years)", "missing", lambda g: f"{g.age.isna().sum()} ({100 * g.age.isna().mean():.1f}%)")
    for s in ["FEMALE", "MALE"]:
        add("Sex", s.title(), lambda g, s=s: f"{(g.sex == s).sum()} ({100 * (g.sex == s).mean():.1f}%)")
    add("Sex", "missing", lambda g: f"{g.sex.isna().sum()} ({100 * g.sex.isna().mean():.1f}%)")
    for s in SITES:
        add("Anatomical site", s, lambda g, s=s: f"{(g.site == s).sum()} ({100 * (g.site == s).mean():.1f}%)")
    add("Anatomical site", "missing", lambda g: f"{g.site.isna().sum()} ({100 * g.site.isna().mean():.1f}%)")
    add("Longest diameter (mm)", "median (IQR)", lambda g: "{:.1f} ({:.1f}-{:.1f})".format(
        *np.nanpercentile(g.diam_long, [50, 25, 75])) if g.diam_long.notna().any() else "NA")
    add("Longest diameter (mm)", "missing", lambda g: f"{g.diam_long.isna().sum()} ({100 * g.diam_long.isna().mean():.1f}%)")
    add("Shortest diameter (mm)", "missing", lambda g: f"{g.diam_short.isna().sum()} ({100 * g.diam_short.isna().mean():.1f}%)")
    add("Complete metadata (age, sex, site, longest diameter)", "n",
        lambda g: f"{g[['age', 'sex', 'site', 'diam_long']].notna().all(1).sum()} "
                  f"({100 * g[['age', 'sex', 'site', 'diam_long']].notna().all(1).mean():.1f}%)")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    v, i = build_vruh(), build_isic()
    df = pd.concat([v, i], ignore_index=True)
    df["label"] = (df.class3 > 0).astype(int)
    df.insert(0, "image_id", [f"{'V' if s == 'VRUH' else 'I'}{k:05d}" for k, s in enumerate(df.source)])
    df["cache_path"] = [str(Path("data/cache_320") / s / f"{iid}.jpg") for iid, s in zip(df.image_id, df.source)]

    # ---- incoherencias ----
    issues = []
    vr = df[df.source == "VRUH"]
    bad_parse = vr[vr.breslow_mm.isna()]
    for _, r in bad_parse.iterrows():
        issues.append(("breslow_unparsed", r.image_id, r.image_name, ""))
    mism = vr[vr.breslow_mm.notna() & (
        ((vr.class3 == 0) & (vr.breslow_mm > 0)) |
        ((vr.class3 == 1) & ((vr.breslow_mm <= 0) | (vr.breslow_mm >= 1))) |
        ((vr.class3 == 2) & (vr.breslow_mm < 1)))]
    for _, r in mism.iterrows():
        issues.append(("breslow_vs_class", r.image_id, r.image_name, f"class3={r.class3}, breslow={r.breslow_mm}"))
    for _, r in df[df.image_name.str.contains("Captura|pantalla", case=False, na=False)].iterrows():
        issues.append(("screenshot_like_name", r.image_id, r.image_name, ""))
    # pacientes con lesiones de distinta clase
    pc = df.groupby("patient_id").class3.nunique()
    for p in pc[pc > 1].index:
        issues.append(("patient_multiple_classes", p, "", str(sorted(df[df.patient_id == p].class3.unique()))))
    lc = df.groupby("lesion_id").class3.nunique()
    for l in lc[lc > 1].index:
        issues.append(("lesion_multiple_classes", l, "", ""))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(issues, columns=["issue", "id", "image_name", "detail"]).to_csv(DATA_DIR / "data_issues.csv", index=False)

    df.drop(columns=["_folder"]).to_csv(UNIFIED_CSV, index=False)
    tdir = RESULTS_DIR / "tables"; tdir.mkdir(parents=True, exist_ok=True)
    describe(df).to_csv(tdir / "table1_descriptives_by_source.csv", index=False)

    # Datos faltantes por fuente y clase
    miss = (df.assign(age_missing=df.age.isna(), sex_missing=df.sex.isna(), site_missing=df.site.isna(),
                      diam_long_missing=df.diam_long.isna(), diam_short_missing=df.diam_short.isna())
            .groupby(["source", "label"])[["age_missing", "sex_missing", "site_missing",
                                            "diam_long_missing", "diam_short_missing"]].agg(["sum", "mean"]))
    miss.to_csv(tdir / "missingness_by_source_and_class.csv")

    print(f"Imágenes: {len(df)} | pacientes: {df.patient_id.nunique()} | lesiones: {df.lesion_id.nunique()}")
    print(df.groupby("source").agg(images=("image_id", "size"), patients=("patient_id", "nunique"),
                                   lesions=("lesion_id", "nunique"), mis=("label", lambda s: (s == 0).sum())))
    print(f"Incidencias: {len(issues)} -> {DATA_DIR / 'data_issues.csv'}")
    print(pd.Series([x[0] for x in issues]).value_counts().to_string() if issues else "")

    if not args.no_cache:
        jobs = [(p, REV / c, args.size) for p, c in zip(df.image_path, df.cache_path)]
        with ProcessPoolExecutor(args.workers) as ex:
            ok = list(ex.map(_cache_one, jobs, chunksize=16))
        st = pd.Series(ok, index=df.image_id)
        print(f"Caché: {st.value_counts().to_dict()} en {CACHE_DIR}")
        bad = df.set_index("image_id").loc[st[st != "ok"].index, ["image_name", "image_path"]].assign(status=st[st != "ok"])
        bad.to_csv(DATA_DIR / "corrupted_images.csv")


if __name__ == "__main__":
    main()
