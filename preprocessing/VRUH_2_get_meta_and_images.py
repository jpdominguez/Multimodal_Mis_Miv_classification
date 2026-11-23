"""
Construye un único CSV a partir del CSV maestro por imagen (`src/lista_imagenes.csv` por defecto) con:
- Rutas de imagen (image_path, image_name, extension, size_bytes si existen)
- Metadatos seleccionados (por defecto sexo/género, localización y edad)
- Columnas normalizadas adicionales: GENDER_STD, LOCATION_I_STD, LOCATION_II_STD, LOCATION_STD
- Columna CLASS y una etiqueta clínica derivada (IN_SITU/INVASIVO/DESCONOCIDO)

Ejemplo de uso CLI:
		python -m src.VRUH_get_meta_and_images \\
		--csv "src/lista_imagenes_VRUH.csv" \
			--meta-cols "GENDER,LOCATION I,LOCATION II,AGE" \\
		--out "reports/dataset_procesado_VRUH.csv"

Por defecto, meta-cols busca columnas típicas: 'GENDER', 'LOCATION I', 'LOCATION II'.
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


DEFAULT_INPUT_CSV = Path("reports") / "lista_imagenes_VRUH.csv"
# Candidatas por defecto para "sexo, location, género" (variantes EN/ES)
DEFAULT_META_CANDIDATES = [
	"SEXO", "GENERO", "GÉNERO", "GENDER",
	"LOCATION", "LOCATION I", "LOCATION II",
	"AGE", "EDAD", "DIAMETER 1", "DIAMETER 2", "NHC"
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
	s = s.replace("-", " ")
	return s


def _normalize_gender_val(x) -> str:
	s = _norm_str(x)
	if s in ("m", "male", "masculino", "h", "hombre", "varon", "varón", "1", "1.0"):
		return "MALE"
	if s in ("f", "female", "femenino", "mujer", "2", "2.0"):
		return "FEMALE"
	return "UNKNOWN"


def _normalize_location_val(x) -> str:
	s = _norm_str(x)
	# reglas por keywords
	# cabeza/cuello
	if any(k in s for k in ["head", "neck", "face", "scalp", "cabeza", "cuello", "cara"]):
		return "HEAD_NECK"
	# torso
	if any(k in s for k in ["torso", "trunk", "abdomen", "chest", "back", "espalda", "pecho", "abdominal", "anterior torso", "posterior torso"]):
		return "TORSO"
	# extremidad superior
	if any(k in s for k in ["arm", "hand", "forearm", "shoulder", "upper limb", "brazo", "mano", "hombro", "antebrazo", "upper"]):
		return "UPPER_LIMB"
	# extremidad inferior
	if any(k in s for k in ["leg", "thigh", "knee", "foot", "ankle", "lower limb", "pierna", "muslo", "rodilla", "pie", "tobillo", "lower"]):
		return "LOWER_LIMB"
	# acral
	if any(k in s for k in ["acral", "palmar", "plantar"]):
		return "ACRAL"
	# genital/inguinal
	if any(k in s for k in ["genital", "groin", "inguinal"]):
		return "GENITAL"
	if s == "" or s == "nan":
		return "UNKNOWN"
	return "OTHER"


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Extraer DataFrames de metadatos e imágenes y de clases desde el CSV de lista de imágenes")
	parser.add_argument("--csv", type=str, default=str(DEFAULT_INPUT_CSV), help="Ruta al CSV de entrada (generado por src.data_VRUH)")
	parser.add_argument("--meta-cols", type=str, default=None, help="Columnas de metadatos separadas por comas. Por defecto: GENDER,LOCATION I,LOCATION II")
	parser.add_argument("--out", type=str, default="reports/dataset_procesado_VRUH.csv", help="Ruta de salida para el CSV unificado")
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
	gender_candidates = ["SEXO", "GENERO", "GÉNERO", "GENDER"]
	gender_src = next((c for c in gender_candidates if c in df_meta.columns), None)
	if gender_src is not None:
		df_meta["GENDER_STD"] = df_meta[gender_src].apply(_normalize_gender_val)

	# Normalización de localización -> LOCATION_I_STD, LOCATION_II_STD, LOCATION_STD
	if "LOCATION I" in df_meta.columns:
		df_meta["LOCATION_I_STD"] = df_meta["LOCATION I"].apply(_normalize_location_val)
	if "LOCATION II" in df_meta.columns:
		df_meta["LOCATION_II_STD"] = df_meta["LOCATION II"].apply(_normalize_location_val)
	# Columna combinada priorizando LOCATION I
	if "LOCATION_I_STD" in df_meta.columns:
		df_meta["LOCATION_STD"] = df_meta["LOCATION_I_STD"]
	elif "LOCATION_II_STD" in df_meta.columns:
		df_meta["LOCATION_STD"] = df_meta["LOCATION_II_STD"]

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)
	df_meta.to_csv(out_path, index=False)

	print(f"CSV unificado escrito en: {out_path} (filas: {len(df_meta)})")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

