"""
Utilidades compartidas del pipeline (Mis frente a Miv).

Contiene:
  - Rutas del proyecto.
  - Codificador de metadatos ajustado SOLO con el fold de entrenamiento
    (variables continuas, one-hot, indicadores de dato faltante).
  - Dataset con imágenes precargadas en memoria (caché redimensionada).
  - Modelos: image-only, concatenación simple (late fusion) y gated fusion
    con cross-attention ESPACIAL real (metadatos -> 49 tokens de DenseNet).
  - Métricas y utilidades de umbral.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "pipeline"
DATA_DIR = REV / "data"
CACHE_DIR = DATA_DIR / "cache_320"
UNIFIED_CSV = DATA_DIR / "dataset_unified.csv"
SPLITS_CSV = DATA_DIR / "splits.csv"
RUNS_DIR = REV / "runs"
RESULTS_DIR = REV / "results"

META_MODELS = ("concat", "gated", "film", "add_lin", "add_mlp")   # modelos de red que usan metadatos
SITES = ["HEAD_NECK", "TORSO", "UPPER_LIMB", "LOWER_LIMB", "ACRAL"]
SEXES = ["FEMALE", "MALE"]
AGE_BINS = [30, 50, 70]          # <30, 30-49, 50-69, >=70
DIAM_BINS = [5, 10]              # <5, 5-9.9, >=10 mm
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def stable_seed(*parts) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()
    return int(h[:8], 16)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_unified() -> pd.DataFrame:
    return pd.read_csv(UNIFIED_CSV, dtype={"patient_id": str, "lesion_id": str, "image_id": str})


def load_split(split_id: str) -> Dict[str, List[str]]:
    sp = pd.read_csv(SPLITS_CSV, dtype=str)
    sp = sp[sp.split_id == split_id]
    if sp.empty:
        raise ValueError(f"split {split_id} no existe en {SPLITS_CSV}")
    return {role: sp[sp.role == role].image_id.tolist() for role in ["train", "val", "test"]}


# ---------------------------------------------------------------------------
# Codificación de metadatos
# ---------------------------------------------------------------------------
class MetaEncoder:
    """Codifica metadatos con estadísticos ajustados en el fold de entrenamiento.

    meta_vars:  subconjunto de {age, sex, site, diam, ratio, source, personal_hx, family_hx}
    age_enc:    'cont' (continua estandarizada) | 'bin' (one-hot 4 bins) | 'both'
    diam_enc:   'cont' (log1p estandarizado)   | 'bin' (one-hot 3 bins)  | 'both'
    missing:    'indicator' -> imputación (mediana/ceros) + indicador binario
                'impute'    -> imputación sin indicador
                'legacy'    -> réplica de la codificación del pipeline original
                               (ratio=0.5, 0 en valores normalizados, categorías UNKNOWN)
    """

    def __init__(self, meta_vars: Sequence[str], age_enc="cont", diam_enc="cont", missing="indicator"):
        self.meta_vars = list(meta_vars)
        self.age_enc = age_enc
        self.diam_enc = diam_enc
        self.missing = missing
        self.stats: Dict[str, float] = {}
        self.names: List[str] = []
        self.cont_mask: Optional[np.ndarray] = None

    # -- ajuste -------------------------------------------------------------
    @staticmethod
    def _isic_policy(df: pd.DataFrame) -> pd.DataFrame:
        """En ISIC la disponibilidad del diámetro depende de la clase (falta en el 100 % de los Mis y solo
        existe en parte de los invasivos, por colección del ISIC Archive): se trata como NO disponible en
        toda la fuente, de modo que su ausencia solo identifica la fuente y no la clase
        (results/audit_metadatos.md, punto 1)."""
        if "source" not in df or not (df.source == "ISIC").any():
            return df
        df = df.copy()
        isic = (df.source == "ISIC").values
        df.loc[isic, "diam_long"] = np.nan
        df.loc[isic, "diam_short"] = np.nan
        return df

    def _ind_mask(self, df: pd.DataFrame) -> np.ndarray:
        """Filas en las que el indicador de faltante de edad/sexo/localización es informativo. En ISIC la
        ausencia de estos campos también depende de la clase (p. ej., edad ausente => invasivo), así que en
        ISIC se imputan SIN indicador."""
        return (df.source != "ISIC").values if "source" in df else np.ones(len(df), bool)

    def fit(self, df: pd.DataFrame) -> "MetaEncoder":
        if self.missing != "legacy":
            df = self._isic_policy(df)
        age = df.age.astype(float)
        dl = np.log1p(df.diam_long.astype(float))
        ratio = df.diam_short.astype(float) / df.diam_long.astype(float)
        self.stats = {
            "age_mean": float(age.mean()), "age_sd": float(age.std() or 1.0),
            "age_median": float(age.median()),
            # sin diámetros observados (p. ej. LOSO entrenado solo con ISIC) -> variable constante 0
            "dl_mean": float(dl.mean()) if dl.notna().any() else 0.0,
            "dl_sd": float(dl.std()) if dl.notna().sum() > 1 and dl.std() > 0 else 1.0,
            "dl_median": float(dl.median()) if dl.notna().any() else 0.0,
            "ratio_mean": float(ratio.mean()) if ratio.notna().any() else 0.7,
            "ratio_sd": float(ratio.std()) if ratio.notna().sum() > 1 else 0.15,
            "ratio_median": float(ratio.median()) if ratio.notna().any() else 0.7,
            # para el modo legacy
            "d1_mean": float(df.diam_long.astype(float).mean()),
            "d1_sd": float(df.diam_long.astype(float).std() or 1.0),
            "d2_mean": float(df.diam_short.astype(float).mean()) if df.diam_short.notna().any() else 0.0,
            "d2_sd": float(df.diam_short.astype(float).std()) if df.diam_short.notna().sum() > 1 else 1.0,
            # imputación en ISIC sin indicador (ver _ind_mask): moda de sexo/localización
            "sex_mode": str(df.sex.dropna().mode().iloc[0]) if df.sex.notna().any() else SEXES[0],
            "site_mode": str(df.site.dropna().mode().iloc[0]) if df.site.notna().any() else SITES[0],
        }
        X, names, cont = self._encode(df.iloc[:1])
        self.names, self.cont_mask = names, np.array(cont, dtype=bool)
        return self

    @property
    def dim(self) -> int:
        return len(self.names)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X, _, _ = self._encode(df)
        return X

    def to_json(self) -> dict:
        return {"meta_vars": self.meta_vars, "age_enc": self.age_enc, "diam_enc": self.diam_enc,
                "missing": self.missing, "stats": self.stats, "names": self.names}

    # -- codificación -------------------------------------------------------
    def _encode(self, df: pd.DataFrame):
        cols, names, cont = [], [], []
        ind = self.missing == "indicator"
        s = self.stats

        def add(vals, name, is_cont):
            cols.append(np.asarray(vals, dtype=np.float32))
            names.append(name)
            cont.append(is_cont)

        n = len(df)
        if self.missing == "legacy":
            return self._encode_legacy(df)
        df = self._isic_policy(df)
        im = self._ind_mask(df)
        if (~im).any():
            # Imputación indistinguible de un valor observado en ISIC: moda para sexo/localización (todo-ceros
            # en el one-hot delataría el faltante) y mediana redondeada a 5 años para la edad (ISIC da la edad
            # en múltiplos de 5; un valor imputado no múltiplo de 5 también lo delataría)
            df = df.copy()
            no = ~im
            df.loc[no & df.sex.isna().values, "sex"] = s.get("sex_mode", SEXES[0])
            df.loc[no & df.site.isna().values, "site"] = s.get("site_mode", SITES[0])
            df.loc[no & df.age.isna().values, "age"] = 5 * round(s["age_median"] / 5)

        if "age" in self.meta_vars:
            age = df.age.astype(float).values
            miss = np.isnan(age)
            if self.age_enc in ("cont", "both"):
                a = np.where(miss, s["age_median"], age)
                add((a - s["age_mean"]) / s["age_sd"], "age_z", True)
            if self.age_enc in ("bin", "both"):
                b = np.digitize(np.nan_to_num(age, nan=-1), AGE_BINS)
                for k in range(len(AGE_BINS) + 1):
                    add(((b == k) & ~miss).astype(float), f"age_bin{k}", False)
            if ind:
                add((miss & im).astype(float), "age_missing", False)

        if "sex" in self.meta_vars:
            sex = df.sex.fillna("").values
            for c in SEXES:
                add((sex == c).astype(float), f"sex_{c}", False)
            if ind:
                add(~np.isin(sex, SEXES) & im, "sex_missing", False)

        if "site" in self.meta_vars:
            site = df.site.fillna("").values
            for c in SITES:
                add((site == c).astype(float), f"site_{c}", False)
            if ind:
                add(~np.isin(site, SITES) & im, "site_missing", False)

        if "diam" in self.meta_vars:
            d = df.diam_long.astype(float).values
            miss = np.isnan(d)
            if self.diam_enc in ("cont", "both"):
                dl = np.where(miss, s["dl_median"], np.log1p(np.nan_to_num(d, nan=0)))
                add((dl - s["dl_mean"]) / s["dl_sd"], "diam_log_z", True)
            if self.diam_enc in ("bin", "both"):
                b = np.digitize(np.nan_to_num(d, nan=-1), DIAM_BINS)
                for k in range(len(DIAM_BINS) + 1):
                    add(((b == k) & ~miss).astype(float), f"diam_bin{k}", False)
            if ind:
                add(miss.astype(float), "diam_missing", False)

        if "ratio" in self.meta_vars:
            r = (df.diam_short.astype(float) / df.diam_long.astype(float)).values
            miss = np.isnan(r)
            r = np.where(miss, s["ratio_median"], r)
            add((r - s["ratio_mean"]) / (s["ratio_sd"] + 1e-8), "ratio_z", True)
            if ind:
                add(miss.astype(float), "ratio_missing", False)

        for hx in ["personal_hx", "family_hx"]:     # antecedentes (solo VRUH; en ISIC no existen)
            if hx in self.meta_vars:
                # Sin indicador de faltante: en VRUH la documentación de los antecedentes depende de la clase
                # (family_hx falta en el 79 % de los Mis y el 64 % de los Miv), así que el indicador sería una fuga.
                # Se imputa 0 (moda, "sin antecedentes").
                add(np.nan_to_num(df[hx].astype(float).values, nan=0.0), hx, False)

        if "source" in self.meta_vars:
            add((df.source.values == "ISIC").astype(float), "source_ISIC", False)

        if not cols:
            return np.zeros((n, 0), np.float32), [], []
        return np.stack(cols, 1).astype(np.float32), names, cont

    def _encode_legacy(self, df: pd.DataFrame):
        """Réplica de train_v3.py (_encode_metadata_advanced) para el control 'legacy'."""
        s = self.stats
        n = len(df)
        cols, names, cont = [], [], []

        def add(v, name, c):
            cols.append(np.asarray(v, np.float32)); names.append(name); cont.append(c)

        age = df.age.astype(float).values
        a = np.where(np.isnan(age), 0.0, np.clip((age - s["age_mean"]) / s["age_sd"], -3, 3))
        add(a, "age_z_legacy", True)
        d1 = df.diam_long.astype(float).values
        d2 = df.diam_short.astype(float).values
        add(np.where(np.isnan(d1), 0.0, np.clip((d1 - s["d1_mean"]) / s["d1_sd"], -3, 3)), "d1_legacy", True)
        add(np.where(np.isnan(d2), 0.0, np.clip((d2 - s["d2_mean"]) / (s["d2_sd"] + 1e-8), -3, 3)), "d2_legacy", True)
        sex = df.sex.fillna("UNKNOWN").values
        for c in SEXES + ["UNKNOWN"]:
            add((sex == c).astype(float), f"sex_{c}", False)
        site = df.site.fillna("UNKNOWN").values
        for c in SITES + ["UNKNOWN"]:
            add((site == c).astype(float), f"site_{c}", False)
        ab = np.where(np.isnan(age), -1, np.digitize(np.nan_to_num(age, nan=0), AGE_BINS))
        add(ab / 3.0, "age_bin_legacy", False)
        ratio = d2 / d1
        add(np.where(np.isnan(ratio), 0.5, ratio), "ratio_legacy_0.5", False)
        mx = np.fmax(d1, d2)
        cat = np.where(np.isnan(mx) | (mx <= 0), 0, 1 + np.digitize(np.nan_to_num(mx, nan=0), DIAM_BINS))
        add(cat / 3.0, "diam_cat_legacy", False)
        return np.stack(cols, 1).astype(np.float32), names, cont


def aux_targets(df: pd.DataFrame, enc: MetaEncoder) -> Dict[str, np.ndarray]:
    """Objetivos para las cabezas auxiliares (edad z, sexo, localización, log-diámetro z), -1/NaN si falta.
    El diámetro sigue la política de ISIC (no disponible en toda la fuente)."""
    s = enc.stats
    age = df.age.astype(float).values
    age_z = np.where(np.isnan(age), np.nan, (age - s["age_mean"]) / s["age_sd"])
    sex = np.array([SEXES.index(x) if x in SEXES else -1 for x in df.sex.fillna("")])
    site = np.array([SITES.index(x) if x in SITES else -1 for x in df.site.fillna("")])
    d = MetaEncoder._isic_policy(df).diam_long.astype(float).values
    diam_z = np.where(np.isnan(d), np.nan, (np.log1p(np.nan_to_num(d, nan=0)) - s["dl_mean"]) / s["dl_sd"])
    return {"age": age_z.astype(np.float32), "sex": sex.astype(np.int64), "site": site.astype(np.int64),
            "diam": diam_z.astype(np.float32)}


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    from sklearn.metrics import roc_curve
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    k = int(np.argmax(j))
    t = float(thr[k])
    return min(max(t, 1e-6), 1 - 1e-6)


def calibration_intercept_slope(y, p):
    """Intercepto (calibration-in-the-large) y pendiente de calibración (logístico)."""
    import statsmodels.api as sm
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    try:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit().params[1]
        inter = sm.GLM(y, np.ones_like(lp)[:, None], family=sm.families.Binomial(), offset=lp).fit().params[0]
        return float(inter), float(slope)
    except Exception:
        return float("nan"), float("nan")


def binary_metrics(y: np.ndarray, p: np.ndarray, yhat: np.ndarray, calib: bool = False) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, cohen_kappa_score
    y = np.asarray(y).astype(int)
    yhat = np.asarray(yhat).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum()); tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum()); fn = int(((yhat == 0) & (y == 1)).sum())
    sens = tp / (tp + fn) if tp + fn else float("nan")
    spec = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    out = {
        "auc": roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan"),
        "balanced_accuracy": (sens + spec) / 2,
        "sensitivity": sens, "specificity": spec, "ppv": ppv, "npv": npv,
        "accuracy": (tp + tn) / len(y),
        "f1": 2 * ppv * sens / (ppv + sens) if (ppv + sens) else float("nan"),
        "kappa": cohen_kappa_score(y, yhat) if len(np.unique(np.r_[y, yhat])) > 1 else 0.0,
        "brier": float(np.mean((p - y) ** 2)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": len(y),
    }
    if calib:
        out["cal_intercept"], out["cal_slope"] = calibration_intercept_slope(y, p)
    return out


# ---------------------------------------------------------------------------
# Torch: dataset y modelos (import perezoso para que el resto funcione sin torch)
# ---------------------------------------------------------------------------
def build_torch_components():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset
    from torchvision import models, transforms
    from PIL import Image

    class CachedImageDataset(Dataset):
        def __init__(self, images: List[np.ndarray], meta: np.ndarray, labels: np.ndarray,
                     aux: Dict[str, np.ndarray], train: bool, img_size=224,
                     meta_noise: float = 0.0, cont_mask: Optional[np.ndarray] = None):
            self.images, self.meta, self.labels, self.aux = images, meta, labels, aux
            self.train, self.meta_noise = train, meta_noise
            self.cont_mask = cont_mask if cont_mask is not None else np.zeros(meta.shape[1], bool)
            norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
            if train:
                self.tf = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomVerticalFlip(0.2),
                    transforms.RandomRotation(5),
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                    transforms.Resize((img_size, img_size)),
                    transforms.ToTensor(), norm])
            else:
                self.tf = transforms.Compose([
                    transforms.Resize((img_size, img_size)), transforms.ToTensor(), norm])

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, i):
            img = self.tf(Image.fromarray(self.images[i]))
            m = self.meta[i].copy()
            if self.train and self.meta_noise > 0 and self.cont_mask.any():
                m[self.cont_mask] += np.random.normal(0, self.meta_noise, self.cont_mask.sum()).astype(np.float32)
            out = (img, torch.from_numpy(m), torch.tensor(self.labels[i], dtype=torch.long),
                   torch.tensor(self.aux["age"][i]), torch.tensor(self.aux["sex"][i]),
                   torch.tensor(self.aux["site"][i]), torch.tensor(self.aux["diam"][i]))
            if "extra" in self.aux:     # batería x11: [clase de grosor, fuente] (solo entrenamiento)
                out = out + (torch.from_numpy(self.aux["extra"][i]),)
            return out

    class FocalLoss(nn.Module):
        """Focal loss (gamma=2) con alpha por clase; acepta reducción 'none'."""
        def __init__(self, alpha=None, gamma=2.0):
            super().__init__()
            self.alpha, self.gamma = alpha, gamma

        def forward(self, logits, target):
            ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
            pt = torch.exp(-F.cross_entropy(logits, target, reduction="none"))
            return ((1 - pt) ** self.gamma * ce).mean()

    class MultimodalNet(nn.Module):
        """
        model='image'  : DenseNet121 -> proyección 256 -> clasificador (baseline).
        model='concat' : + MLP de metadatos (64) concatenado con la imagen (late fusion).
        model='gated'  : gated fusion (opcional) + cross-attention ESPACIAL (opcional):
                         la consulta (metadatos) atiende a los 7x7=49 tokens del mapa
                         de características de DenseNet, + cabezas auxiliares (opcional).
        Exploratorios (todos empiezan EXACTAMENTE como el modelo de imagen):
        model='film'    : FiLM, los metadatos modulan la proyección de imagen: z = ip*(1+γ) + β (γ=β=0 al inicio).
        model='add_lin' : logit = clasificador(imagen) + lineal(metadatos)  (aditivo en log-odds, init 0).
        model='add_mlp' : logit = clasificador(imagen) + MLP(metadatos)     (init 0 en la última capa).
        meta_dropout: dropout de la rama de metadatos (por defecto = dropout).
        backbone: 'densenet121' (todos los modelos principales) o 'resnet50' (batería x9; sin model='gated', que
                  supone 7x7 tokens de 1024 canales).
        """
        def __init__(self, model: str, meta_dim: int, gate=True, xattn=True, aux=False,
                     dropout=0.5, pretrained=True, mod_drop_img=0.0, mod_drop_meta=0.0, meta_dropout=None,
                     aux_diam=False, backbone="densenet121", fmap_film=False):
            super().__init__()
            self.model, self.gate_on, self.xattn_on, self.aux_on = model, gate, xattn, aux
            # modality dropout (solo en entrenamiento, modelo concat): anula por muestra la rama de imagen
            # y/o la de metadatos para que el clasificador no pueda ignorar los metadatos (paso exploratorio 4)
            self.mod_drop_img, self.mod_drop_meta = mod_drop_img, mod_drop_meta
            if backbone == "resnet50":
                rn = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
                self.features = nn.Sequential(*list(rn.children())[:-2])     # sin avgpool ni fc
                fd = 2048
            else:
                try:
                    dn = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None)
                except AttributeError:
                    dn = models.densenet121(pretrained=pretrained)
                self.features = dn.features
                fd = 1024
            self.img_proj = nn.Sequential(nn.Linear(fd, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout))
            use_meta = model in META_MODELS and meta_dim > 0
            self.use_meta = use_meta
            md = dropout if meta_dropout is None else meta_dropout
            if use_meta and model != "add_lin":
                self.meta_mlp = nn.Sequential(
                    nn.Linear(meta_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(md),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU())
            if use_meta and model == "film":
                self.film = nn.Linear(64, 2 * 256)
                nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
            if use_meta and model in ("add_lin", "add_mlp"):
                self.meta_out = nn.Linear(meta_dim if model == "add_lin" else 64, 2)
                nn.init.zeros_(self.meta_out.weight); nn.init.zeros_(self.meta_out.bias)
            cls_in = 256
            if model == "concat" and use_meta:
                cls_in = 256 + 64
            if model == "gated" and use_meta:
                self.meta_proj = nn.Sequential(nn.Linear(64, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(dropout))
                if gate:
                    self.gate = nn.Sequential(nn.Linear(fd + 64, 256), nn.ReLU(), nn.Linear(256, 2), nn.Softmax(dim=1))
                if xattn:
                    self.tok_proj = nn.Conv2d(fd, 256, 1)
                    self.pos = nn.Parameter(torch.zeros(1, 49, 256))
                    nn.init.trunc_normal_(self.pos, std=0.02)
                    self.xattn = nn.MultiheadAttention(256, 8, dropout=dropout * 0.5, batch_first=True)
                    self.xattn_norm = nn.LayerNorm(256)
            # cabezas auxiliares: con metadatos de entrada (gated/concat) o, en model='image' con aux=True,
            # como información privilegiada (LUPI: solo en entrenamiento; la inferencia usa solo la imagen)
            self.aux_on = aux and (use_meta or model == "image")
            if self.aux_on:
                self.aux_age = nn.Sequential(nn.Linear(fd, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
                self.aux_sex = nn.Sequential(nn.Linear(fd, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, len(SEXES)))
                self.aux_site = nn.Sequential(nn.Linear(fd, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, len(SITES)))
            self.classifier = nn.Sequential(nn.Linear(cls_in, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                            nn.Dropout(dropout), nn.Linear(128, 2))
            # cabeza de diámetro (LUPI): se crea al final para no alterar la inicialización de los modelos previos
            self.aux_diam = (nn.Sequential(nn.Linear(fd, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))
                             if (self.aux_on and aux_diam) else None)
            # x16: FiLM sobre el mapa de características (γ, β por canal desde la rama de metadatos; init 0)
            self.fmap_film = None
            if fmap_film and use_meta and model == "concat":
                self.fmap_film = nn.Linear(64, 2 * fd)
                nn.init.zeros_(self.fmap_film.weight); nn.init.zeros_(self.fmap_film.bias)
            self.last_gate = None
            self.last_attn = None

        def forward(self, x, meta=None, return_aux=False):
            fmap = F.relu(self.features(x))                       # [B,1024,7,7]
            if self.fmap_film is not None:
                g_, b_ = self.fmap_film(self.meta_mlp(meta)).chunk(2, 1)
                fmap = fmap * (1 + g_[:, :, None, None]) + b_[:, :, None, None]
            pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)     # [B,1024]
            ip = self.img_proj(pooled)
            if not self.use_meta:
                z = ip
            elif self.model == "film":
                g, b = self.film(self.meta_mlp(meta)).chunk(2, 1)
                z = ip * (1 + g) + b
            elif self.model in ("add_lin", "add_mlp"):
                z = ip
            elif self.model == "concat":
                mf = self.meta_mlp(meta)
                if self.training and (self.mod_drop_img > 0 or self.mod_drop_meta > 0):
                    u = torch.rand(ip.size(0), 1, device=ip.device)
                    drop_i = u < self.mod_drop_img                                   # nunca las dos a la vez
                    drop_m = (u >= self.mod_drop_img) & (u < self.mod_drop_img + self.mod_drop_meta)
                    ip = ip * (~drop_i).to(ip.dtype)
                    mf = mf * (~drop_m).to(mf.dtype)
                self.last_mf = mf               # para las pérdidas de la batería x13 (no cambia el cálculo)
                z = torch.cat([ip, mf], 1)
            else:
                mf = self.meta_mlp(meta)
                mp = self.meta_proj(mf)
                if self.gate_on:
                    g = self.gate(torch.cat([pooled, mf], 1))
                    self.last_gate = g.detach()
                    z = g[:, :1] * ip + g[:, 1:] * mp
                else:
                    z = ip + mp
                if self.xattn_on:
                    tok = self.tok_proj(fmap).flatten(2).transpose(1, 2)    # [B,49,256]
                    tok = tok + self.pos[:, :tok.shape[1]]
                    att, w = self.xattn(mp.unsqueeze(1), tok, tok)          # query = metadatos
                    self.last_attn = w.detach()
                    z = z + self.xattn_norm(att.squeeze(1))                 # conexión residual
            self.last_ip, self.last_z = ip, z
            logits = self.classifier(z)
            if self.use_meta and self.model == "add_lin":
                logits = logits + self.meta_out(meta)
            elif self.use_meta and self.model == "add_mlp":
                logits = logits + self.meta_out(self.meta_mlp(meta))
            if return_aux and self.aux_on:
                return logits, (self.aux_age(pooled).squeeze(1), self.aux_sex(pooled), self.aux_site(pooled),
                                self.aux_diam(pooled).squeeze(1) if self.aux_diam is not None else None)
            return logits

    return {"CachedImageDataset": CachedImageDataset, "FocalLoss": FocalLoss, "MultimodalNet": MultimodalNet}


def load_cached_images(df: pd.DataFrame) -> List[np.ndarray]:
    from PIL import Image
    out = []
    for p in df.cache_path:
        with Image.open(REV / p) as im:
            out.append(np.asarray(im.convert("RGB")).copy())
    return out


def write_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
