"""
Genera un CSV con una fila por imagen uniendo toda la información de la base de datos por paciente
y añadiendo una columna CLASS basada en Tis/Breslow.

Reglas CLASS:
- 0 si la imagen corresponde a un paciente Tis o con Breslow 0
- 1 si la imagen es de un paciente con Breslow < 1.0 pero no es Tis ni tiene Breslow 0
- 2 si la imagen es de un paciente con Breslow >= 1.0

Uso rápido (ejemplos):
  python -m src.data_VRUH \
	--db "data/Base de datos. Prueba 4.0.xlsx" \
	--db-col "NHC" \
	--image-root "data/VRUH" \
	--out "scripts/lista_imagenes_VRUH.csv"

Puedes especificar las columnas de Breslow y T mediante --breslow-col y --tstage-col.
Si no se indican, se intentarán detectar automáticamente por nombre.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from typing import Dict, List, Any, Optional, Iterable, Tuple

try:
	import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - entorno sin pandas
	pd = None  # type: ignore


DEFAULT_IMAGE_ROOT = Path("data") / "VRUH/IMÁGENES MELANOMA"  # Busca en toda la carpeta VRUH (incluye Melanoma/Nevi)
DEFAULT_DB = Path("data") / "VRUH" / "Base de datos. Prueba 4.0.xlsx"


def _normalize_token(s: Any) -> str:
	import unicodedata
	s = str(s).lower()
	s = unicodedata.normalize("NFKD", s)
	s = "".join(c for c in s if not unicodedata.combining(c))
	s = s.replace(",", ".").replace(" ", "")
	return s


def _id_variants(raw_id: Any) -> List[str]:
	# Genera variantes del ID para maching flexible en rutas/nombres
	rid = str(raw_id).strip()
	if rid == "" or rid.lower() == "nan":
		return []
	v = _normalize_token(rid)
	variants = {v}
	import re
	v_simple = re.sub(r"[^0-9a-z]", "", v)
	if v_simple:
		variants.add(v_simple)
		if v_simple.startswith("t"):
			variants.add(v_simple[1:])
		else:
			variants.add("t" + v_simple)
	return list(variants)


def _read_db(path: Path):
	if not path.exists():
		raise FileNotFoundError(f"Base de datos no encontrada: {path}")
	suf = path.suffix.lower()
	if suf in (".xlsx", ".xls"):
		if pd is None:
			raise RuntimeError("pandas no está instalado. Instálalo para leer Excel.")
		return pd.read_excel(path)
	elif suf == ".csv":
		if pd is None:
			# Requerimos pandas para manejar CSV de forma coherente con el resto del flujo
			raise RuntimeError("pandas no disponible para procesar CSV. Instálalo y vuelve a intentar.")
		else:
			return pd.read_csv(path)
	else:
		raise RuntimeError(f"Formato no soportado: {suf}")


def _find_column(df, candidates: Iterable[str]) -> Optional[str]:
	"""Busca una columna en df que coincida case-insensitive por cualquiera de los candidatos."""
	cols = [str(c) for c in df.columns]
	lower_map = {str(c).lower(): c for c in cols}
	# Búsqueda directa exacta (case-insensitive)
	for cand in candidates:
		lc = cand.lower()
		if lc in lower_map:
			return lower_map[lc]
	# Búsqueda por contains
	for cand in candidates:
		lc = cand.lower()
		for c in cols:
			if lc in str(c).lower():
				return c
	return None


def _parse_breslow(val: Any) -> Optional[float]:
	if val is None:
		return None
	s = str(val).strip()
	if s == "" or s.lower() == "nan":
		return None
	# Sustituir coma decimal por punto y quitar textos como "mm"
	s = s.replace(",", ".")
	# Extraer número flotante
	import re
	m = re.search(r"-?\d+(?:\.\d+)?", s)
	if not m:
		return None
	try:
		return float(m.group(0))
	except Exception:
		return None


def _is_tis(val: Any) -> bool:
	if val is None:
		return False
	s = _normalize_token(val)
	# Buscar 'tis' como token
	return "tis" in s


def infer_class(row: Dict[str, Any], breslow_col: Optional[str], tstage_col: Optional[str]) -> Optional[int]:
	# Prioridad Tis => clase 0
	if tstage_col is not None and tstage_col in row and _is_tis(row.get(tstage_col)):
		return 0
	# Breslow
	b = None
	if breslow_col is not None and breslow_col in row:
		b = _parse_breslow(row.get(breslow_col))
	if b is None:
		return None  # desconocido, se deja vacío
	if b == 0:
		return 0
	if 0 < b < 1.0:
		return 1
	return 2


def build_variant_to_rows(df, id_col: str) -> Dict[str, List[Dict[str, Any]]]:
	mapping: Dict[str, List[Dict[str, Any]]] = {}
	for _, r in df.iterrows():
		raw_id = r[id_col]
		row_dict = {str(k): r[k] for k in df.columns}
		vars_ = _id_variants(raw_id)
		for v in vars_:
			mapping.setdefault(v, []).append(row_dict)
	return mapping


def walk_images(image_root: Path) -> List[Dict[str, Any]]:
	"""Recorre image_root y retorna lista de registros de imagen."""
	out: List[Dict[str, Any]] = []
	if not image_root.exists():
		raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {image_root}")
	for p in image_root.rglob("*"):
		if not p.is_file():
			continue
		try:
			size = p.stat().st_size
		except Exception:
			size = None
		out.append(
			{
				"image_path": str(p.resolve()),
				"image_name": p.name,
				"extension": p.suffix.lower(),
				"size_bytes": size,
				"parts_norm": [_normalize_token(pp) for pp in p.parts],
				"name_norm": _normalize_token(p.name),
			}
		)
	return out


def match_images_to_rows(images: List[Dict[str, Any]], variant_to_rows: Dict[str, List[Dict[str, Any]]]) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
	"""Devuelve lista de (row_db, rec_imagen) por cada match encontrado."""
	matches: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
	for rec in images:
		# probar contra todas las variantes
		matched = False
		for var, rows in variant_to_rows.items():
			if var and (var in rec["name_norm"] or any(var in part for part in rec["parts_norm"])):
				for row in rows:
					matches.append((row, rec))
				matched = True
				break
		# Si no hay match, lo ignoramos (podría añadirse lógica para analizar no encontrados)
		_ = matched
	return matches


def main(argv: Optional[List[str]] = None) -> int:
	parser = argparse.ArgumentParser(description="Generar CSV por imagen con metadata completa y columna CLASS")
	parser.add_argument("--db", type=str, default=str(DEFAULT_DB), help="Ruta al archivo de base de datos (XLSX o CSV)")
	parser.add_argument("--db-col", type=str, default="NHC", help="Nombre de la columna que contiene el identificador del paciente")
	parser.add_argument("--image-root", type=str, default=str(DEFAULT_IMAGE_ROOT), help="Carpeta raíz de imágenes a recorrer")
	parser.add_argument("--out", type=str, default="reports/lista_imagenes_VRUH.csv", help="CSV de salida")
	parser.add_argument(
		"--report", type=str, default=None,
		help="Ruta del reporte TXT con conteo por clases. Por defecto, mismo nombre que --out pero con extensión .txt"
	)
	parser.add_argument("--breslow-col", type=str, default=None, help="Nombre de la columna de Breslow (si se omite, se intentará detectar)")
	parser.add_argument("--tstage-col", type=str, default=None, help="Nombre de la columna con estadio T/TNM (para detectar Tis)")
	args = parser.parse_args(argv)

	db_path = Path(args.db)
	try:
		df = _read_db(db_path)
	except Exception as e:
		print(f"ERROR leyendo la base de datos: {e}", file=sys.stderr)
		return 1

	if pd is None:
		print("Este script requiere pandas para operar de forma fiable. Instálalo e inténtalo de nuevo.", file=sys.stderr)
		return 1

	if args.db_col not in df.columns:
		print(
			"Columna de ID no encontrada. Columnas disponibles:\n  " + "\n  ".join(map(str, df.columns)),
			file=sys.stderr,
		)
		return 1

	# Detectar columnas Breslow/T si no se pasan
	breslow_col = args.breslow_col or _find_column(df, ["breslow", "espesor", "grosor", "thickness"])
	tstage_col = args.tstage_col or _find_column(df, ["t", "tnm", "estadio", "stage"])

	if breslow_col is None:
		print("Aviso: No se detectó columna de Breslow. La CLASS puede quedar vacía.", file=sys.stderr)
	if tstage_col is None:
		print("Aviso: No se detectó columna de estadio T/TNM. Se usará solo Breslow para CLASS.", file=sys.stderr)

	# Construir índice por variantes de ID
	variant_to_rows = build_variant_to_rows(df, args.db_col)

	# Recolectar imágenes
	images = walk_images(Path(args.image_root))

	# Emparejar
	pairs = match_images_to_rows(images, variant_to_rows)

	# Preparar salida: columnas DB + info imagen + CLASS
	db_cols = [str(c) for c in df.columns]
	extra_cols = ["image_path", "image_name", "extension", "size_bytes", "CLASS"]
	header = db_cols + extra_cols

	out_path = Path(args.out)
	out_path.parent.mkdir(parents=True, exist_ok=True)

	written = 0
	class_counts = {0: 0, 1: 0, 2: 0, None: 0}
	with out_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		for row, rec in pairs:
			# Calcular CLASS
			c = infer_class(row, breslow_col, tstage_col)
			# Construir fila
			values = [row.get(cn) for cn in db_cols] + [
				rec.get("image_path"),
				rec.get("image_name"),
				rec.get("extension"),
				rec.get("size_bytes"),
				c if c is not None else "",
			]
			writer.writerow(values)
			written += 1
			if c in class_counts:
				class_counts[c] += 1
			else:
				class_counts[None] += 1

	print(f"CSV escrito en: {out_path} (filas: {written})")
	if written == 0:
		print("Advertencia: no se generaron filas. Revisa que los IDs del DB coincidan con nombres/rutas de imágenes.", file=sys.stderr)
	# Generar reporte TXT con conteos
	report_path = Path(args.report) if args.report else out_path.with_suffix('.txt')
	try:
		# Mapeo semántico adicional
		# Clase 0 -> in situ; Clases 1 y 2 -> invasivo
		in_situ = class_counts.get(0, 0)
		invasivo = class_counts.get(1, 0) + class_counts.get(2, 0)
		desconocido = class_counts.get(None, 0)
		total = sum(v for v in class_counts.values())
		with report_path.open('w', encoding='utf-8') as rf:
			rf.write("Reporte de imágenes por clase\n")
			rf.write("===============================\n\n")
			rf.write(f"Total de imágenes: {total}\n\n")
			rf.write("Por CLASS:\n")
			rf.write(f"  - CLASS 0 (in situ): {in_situ}\n")
			rf.write(f"  - CLASS 1: {class_counts.get(1, 0)}\n")
			rf.write(f"  - CLASS 2: {class_counts.get(2, 0)}\n")
			rf.write(f"  - Desconocido: {desconocido}\n\n")
			rf.write("Agrupación clínica:\n")
			rf.write(f"  - In situ (CLASS 0): {in_situ}\n")
			rf.write(f"  - Invasivo (CLASS 1 + 2): {invasivo}\n")
		print(f"Reporte escrito en: {report_path}")
	except Exception as e:
		print(f"No se pudo escribir el reporte: {e}", file=sys.stderr)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

