"""
Genera un CSV con una fila por imagen del dataset ISIC uniendo metadata desde metadata.csv
y añadiendo una columna CLASS basada en mel_thick_mm y mel_class.

Reglas CLASS:
- 0 si la imagen corresponde a melanoma in situ (mel_thick_mm == 0.00 o mel_class == "melanoma in situ")
- 1 si mel_thick_mm < 1.0 pero no es in situ (mel_thick_mm entre 0.01 y 0.99)
- 2 si mel_thick_mm >= 1.0

Uso rápido (ejemplos):
  python -m src.ISIC_1_generate_image_data \
    --metadata "data/ISIC-images-BT/metadata.csv" \
    --image-root "data/ISIC-images-BT" \
    --out "reports/lista_imagenes_ISIC.csv"
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import sys
from typing import Dict, List, Any, Optional

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover - entorno sin pandas
    pd = None  # type: ignore


DEFAULT_IMAGE_ROOT = Path("data") / "ISIC-images-BT"
DEFAULT_METADATA = Path("data") / "ISIC-images-BT" / "metadata.csv"


def _parse_thickness(val: Any) -> Optional[float]:
    """Parsea el grosor del melanoma (mel_thick_mm)."""
    if val is None or pd.isna(val):
        return None
    try:
        return float(val)
    except Exception:
        return None


def _is_in_situ_by_class(val: Any) -> bool:
    """Detecta si mel_class indica melanoma in situ."""
    if val is None or pd.isna(val):
        return False
    s = str(val).lower().strip()
    return "in situ" in s or s == "melanoma in situ"


def infer_class(row: Dict[str, Any]) -> Optional[int]:
    """
    Inferir CLASS según grosor del melanoma.
    - 0: melanoma in situ (grosor 0.00 o mel_class="melanoma in situ")
    - 1: grosor < 1.0 (pero no in situ)
    - 2: grosor >= 1.0
    """
    mel_class = row.get("mel_class")
    mel_thick = _parse_thickness(row.get("mel_thick_mm"))
    
    # Prioridad a mel_class si indica in situ
    if _is_in_situ_by_class(mel_class):
        return 0
    
    # Si no hay grosor disponible, no podemos clasificar
    if mel_thick is None:
        return None
    
    # Clasificar según grosor
    if mel_thick == 0.0:
        return 0
    elif 0 < mel_thick < 1.0:
        return 1
    else:  # >= 1.0
        return 2


def walk_images(image_root: Path) -> List[Dict[str, Any]]:
    """Recorre image_root y retorna lista de registros de imagen."""
    out: List[Dict[str, Any]] = []
    if not image_root.exists():
        raise FileNotFoundError(f"No se encontró la carpeta de imágenes: {image_root}")
    
    for p in image_root.rglob("*"):
        if not p.is_file():
            continue
        # Ignorar archivos que no son imágenes
        if p.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue
        
        try:
            size = p.stat().st_size
        except Exception:
            size = None
        
        # Extraer ISIC_ID del nombre del archivo (sin extensión)
        isic_id = p.stem  # e.g., "ISIC_0020963"
        
        out.append(
            {
                "isic_id": isic_id,
                "image_path": str(p.resolve()),
                "image_name": p.name,
                "extension": p.suffix.lower(),
                "size_bytes": size,
            }
        )
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Generar CSV por imagen del dataset ISIC con metadata completa y columna CLASS")
    parser.add_argument("--metadata", type=str, default=str(DEFAULT_METADATA), help="Ruta al archivo metadata.csv")
    parser.add_argument("--image-root", type=str, default=str(DEFAULT_IMAGE_ROOT), help="Carpeta raíz de imágenes ISIC")
    parser.add_argument("--out", type=str, default="reports/lista_imagenes_ISIC.csv", help="CSV de salida")
    parser.add_argument(
        "--report", type=str, default=None,
        help="Ruta del reporte TXT con conteo por clases. Por defecto, mismo nombre que --out pero con extensión .txt"
    )
    args = parser.parse_args(argv)

    metadata_path = Path(args.metadata)
    
    if not metadata_path.exists():
        print(f"ERROR: Archivo de metadata no encontrado: {metadata_path}", file=sys.stderr)
        return 1

    if pd is None:
        print("Este script requiere pandas. Instálalo e inténtalo de nuevo.", file=sys.stderr)
        return 1

    # Cargar metadata
    try:
        df_metadata = pd.read_csv(metadata_path)
    except Exception as e:
        print(f"ERROR leyendo metadata: {e}", file=sys.stderr)
        return 1

    if 'isic_id' not in df_metadata.columns:
        print("ERROR: La columna 'isic_id' no se encontró en metadata.csv", file=sys.stderr)
        return 1

    # Crear diccionario de metadata por isic_id
    metadata_dict = {}
    for _, row in df_metadata.iterrows():
        isic_id = row['isic_id']
        metadata_dict[isic_id] = row.to_dict()

    # Recolectar imágenes
    images = walk_images(Path(args.image_root))
    print(f"Imágenes encontradas: {len(images)}")

    # Preparar salida: columnas metadata + info imagen + CLASS
    metadata_cols = list(df_metadata.columns)
    extra_cols = ["image_path", "image_name", "extension", "size_bytes", "CLASS"]
    header = metadata_cols + extra_cols

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    matched = 0
    class_counts = {0: 0, 1: 0, 2: 0, None: 0}
    
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        for img_rec in images:
            isic_id = img_rec["isic_id"]
            
            # Buscar metadata correspondiente
            if isic_id not in metadata_dict:
                # No hay metadata para esta imagen, omitir
                continue
            
            matched += 1
            metadata_row = metadata_dict[isic_id]
            
            # Calcular CLASS
            c = infer_class(metadata_row)
            
            # Construir fila
            values = [metadata_row.get(col) for col in metadata_cols] + [
                img_rec.get("image_path"),
                img_rec.get("image_name"),
                img_rec.get("extension"),
                img_rec.get("size_bytes"),
                c if c is not None else "",
            ]
            writer.writerow(values)
            written += 1
            
            if c in class_counts:
                class_counts[c] += 1
            else:
                class_counts[None] += 1

    print(f"CSV escrito en: {out_path}")
    print(f"Filas escritas: {written} (imágenes con metadata: {matched}/{len(images)})")
    
    if written == 0:
        print("Advertencia: no se generaron filas. Revisa que los nombres de archivo coincidan con los isic_id en metadata.", file=sys.stderr)

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
            rf.write("Reporte de imágenes ISIC por clase\n")
            rf.write("====================================\n\n")
            rf.write(f"Total de imágenes procesadas: {total}\n\n")
            rf.write("Por CLASS:\n")
            rf.write(f"  - CLASS 0 (in situ): {in_situ}\n")
            rf.write(f"  - CLASS 1 (< 1.0mm): {class_counts.get(1, 0)}\n")
            rf.write(f"  - CLASS 2 (>= 1.0mm): {class_counts.get(2, 0)}\n")
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
