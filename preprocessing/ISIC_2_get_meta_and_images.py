"""
Construye un único CSV a partir del CSV intermedio de ISIC con:
- Rutas de imagen (image_path, image_name, extension, size_bytes)
- Metadatos seleccionados y normalizados (sexo, localización, edad, diámetro)
- Columnas normalizadas adicionales: GENDER_STD, LOCATION_STD, AGE
- Columna CLASS y una etiqueta clínica derivada (IN_SITU/INVASIVO/DESCONOCIDO)

Ejemplo de uso CLI:
    python -m src.ISIC_2_get_meta_and_images \
        --csv "reports/lista_imagenes_ISIC.csv" \
        --meta-cols "sex,anatom_site_general,age_approx,clin_size_long_diam_mm" \
        --out "reports/dataset_procesado_ISIC.csv"

Por defecto, meta-cols busca columnas típicas de ISIC: sex, anatom_site_general, age_approx, clin_size_long_diam_mm.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, TYPE_CHECKING
import argparse
import sys

try:
	import pandas as pd  # type: ignore
except Exception:
	pd = None  # type: ignore

if TYPE_CHECKING:  # solo para tipado estático
	from pandas import DataFrame


DEFAULT_INPUT_CSV = Path("reports") / "lista_imagenes_ISIC.csv"
# Columnas típicas de ISIC que queremos preservar
DEFAULT_META_CANDIDATES = [
	"isic_id", "sex", "anatom_site_general", "age_approx", 
	"clin_size_long_diam_mm", "mel_thick_mm", "mel_class",
	"diagnosis", "benign_malignant", "patient_id"
]
ALWAYS_INCLUDE = ["image_path", "image_name", "extension", "size_bytes"]


def load_csv(path: Path) -> "DataFrame":
	if pd is None:
		raise RuntimeError("Este script requiere pandas. Instálalo e inténtalo de nuevo.")
	if not path.exists():
		raise FileNotFoundError(f"No se encontró el CSV de entrada: {path}")
	return pd.read_csv(path)


def _resolve_columns(df: "DataFrame", requested: Optional[Sequence[str]]) -> List[str]:
	cols = [str(c) for c in df.columns]
	if requested is None:
		# Elegir por defecto las que existan en candidatos
		out = [c for c in DEFAULT_META_CANDIDATES if c in cols]
		return out
	# Normalizar espacios
	req = [c.strip() for c in requested if c and c.strip()]
	# Mantener solo las que existan
	return [c for c in req if c in cols]


def build_metadata_df(df: "DataFrame", meta_columns: Optional[Sequence[str]] = None) -> "DataFrame":
	selected_meta = _resolve_columns(df, meta_columns)
	include_cols = [c for c in ALWAYS_INCLUDE if c in df.columns]
	use_cols = include_cols + selected_meta
	# Eliminar duplicados en el orden
	seen = set()
	ordered = []
	for c in use_cols:
		if c not in seen:
			ordered.append(c)
			seen.add(c)
	return df[ordered].copy()


def _class_to_label(c: Optional[int]) -> str:
	if c is None or (isinstance(c, float) and pd.isna(c)):
		return "DESCONOCIDO"
	try:
		ci = int(c)
	except Exception:
		return "DESCONOCIDO"
	if ci == 0:
		return "IN_SITU"
	if ci in (1, 2):
		return "INVASIVO"
	return "DESCONOCIDO"


def _norm_str(x) -> str:
	s = str(x) if x is not None else ""
	s = s.strip().lower()
	# quitar acentos
	import unicodedata
	s = unicodedata.normalize("NFKD", s)
	s = "".join(c for c in s if not unicodedata.combining(c))
	# normalizar separadores
	s = s.replace("-", " ").replace("_", " ").replace("/", " ")
	return s


def _normalize_gender_val(x) -> str:
	"""Normaliza valores de sexo del dataset ISIC."""
	s = _norm_str(x)
	if s in ("m", "male", "masculino", "h", "hombre", "varon", "varón", "1", "1.0"):
		return "MALE"
	if s in ("f", "female", "femenino", "mujer", "2", "2.0"):
		return "FEMALE"
	return "UNKNOWN"


def _normalize_location_val(x) -> str:
	"""Normaliza valores de localización anatómica del dataset ISIC."""
	s = _norm_str(x)
	# reglas específicas de ISIC
	if "head" in s or "neck" in s:
		return "HEAD_NECK"
	if "torso" in s or "trunk" in s or "anterior torso" in s or "posterior torso" in s:
		return "TORSO"
	if "upper extremity" in s or "upper limb" in s:
		return "UPPER_LIMB"
	if "lower extremity" in s or "lower limb" in s:
		return "LOWER_LIMB"
	if "palms" in s or "soles" in s or "acral" in s:
		return "ACRAL"
	if "genital" in s:
		return "GENITAL"
	if s == "" or s == "nan":
		return "UNKNOWN"
	return "OTHER"


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Extraer DataFrame procesado desde el CSV intermedio de ISIC")
	parser.add_argument("--csv", type=str, default=str(DEFAULT_INPUT_CSV), help="Ruta al CSV de entrada (generado por ISIC_1_generate_image_data.py)")
	parser.add_argument("--meta-cols", type=str, default=None, help="Columnas de metadatos separadas por comas. Por defecto: sex,anatom_site_general,age_approx,clin_size_long_diam_mm")
	parser.add_argument("--out", type=str, default="reports/dataset_procesado_ISIC.csv", help="Ruta de salida para el CSV unificado")
	args = parser.parse_args(argv)

	if pd is None:
		print("Este script requiere pandas. Instala pandas y vuelve a intentar.", file=sys.stderr)
		return 1

	in_path = Path(args.csv)
	df = load_csv(in_path)

	# Procesar columnas pedidas
	meta_cols = None
	if args.meta_cols:
		meta_cols = [c.strip() for c in args.meta_cols.split(",") if c.strip()]

	# Unificar: imagen + metadatos + normalizados + CLASS + etiqueta clínica
	df_meta = build_metadata_df(df, meta_cols)
	
	# Añadir CLASS si no está incluido ya
	cols = list(df_meta.columns)
	if "CLASS" in df.columns and "CLASS" not in cols:
		df_meta = df_meta.join(df["CLASS"])  # index alineado, el df proviene del mismo CSV
	
	# Etiqueta clínica
	if "CLASS" in df_meta.columns:
		df_meta["CLASS_LABEL"] = df_meta["CLASS"].apply(_class_to_label)
	else:
		df_meta["CLASS_LABEL"] = "DESCONOCIDO"

	# Normalización de género -> GENDER_STD
	if "sex" in df_meta.columns:
		df_meta["GENDER_STD"] = df_meta["sex"].apply(_normalize_gender_val)

	# Normalización de localización -> LOCATION_STD
	if "anatom_site_general" in df_meta.columns:
		df_meta["LOCATION_STD"] = df_meta["anatom_site_general"].apply(_normalize_location_val)

	# Renombrar age_approx a AGE para consistencia con VRUH
	if "age_approx" in df_meta.columns:
		df_meta["AGE"] = df_meta["age_approx"]
	
	# Renombrar clin_size_long_diam_mm a DIAMETER 1 para consistencia con VRUH
	if "clin_size_long_diam_mm" in df_meta.columns:
		df_meta["DIAMETER 1"] = df_meta["clin_size_long_diam_mm"]
	
	# Crear DIAMETER 2 vacío para mantener compatibilidad con train.py
	df_meta["DIAMETER 2"] = None

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df_meta.to_csv(out_path, index=False)

	print(f"CSV unificado escrito en: {out_path} (filas: {len(df_meta)})")
	
	# Mostrar estadísticas básicas
	if "CLASS" in df_meta.columns:
		class_counts = df_meta["CLASS"].value_counts().to_dict()
		print(f"\nDistribución de clases:")
		print(f"  - CLASS 0 (in situ): {class_counts.get(0, 0)}")
		print(f"  - CLASS 1 (< 1.0mm): {class_counts.get(1, 0)}")
		print(f"  - CLASS 2 (>= 1.0mm): {class_counts.get(2, 0)}")
	
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
