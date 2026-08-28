# =============================================================================
# CELDA 11A — Diagnóstico seguro del montaje D1D
# Sólo enumera directorios y marcadores de release bajo /kaggle/input. No abre
# shards, no lee holdout, no carga pesos, no usa GPU y no inicia entrenamiento.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path


INPUT_ROOT = Path("/kaggle/input")
EXPECTED_RELEASE = "d1d-v1-router-entropy-train-only"

if not INPUT_ROOT.exists():
    raise RuntimeError(f"No existe el directorio esperado: {INPUT_ROOT}")

print("CELDA 11A — diagnóstico seguro del montaje D1D")
print(f"INPUT_ROOT: {INPUT_ROOT}")
print("TOP_LEVEL:")
for path in sorted(INPUT_ROOT.iterdir()):
    print(f"- {path}")

markers = []
for marker in sorted(INPUT_ROOT.rglob("aethel_d1d_v5_source_release.json")):
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"MARCADOR_NO_LEIBLE: {marker} ({type(exc).__name__})")
        continue
    markers.append((marker, payload.get("release")))

print("D1D_MARKERS:")
if not markers:
    print("- ninguno")
else:
    for marker, release in markers:
        print(f"- {marker} :: release={release!r}")

zip_candidates = sorted(INPUT_ROOT.rglob("*.zip"))
print("ZIP_CANDIDATES:")
if not zip_candidates:
    print("- ninguno")
else:
    for path in zip_candidates:
        print(f"- {path}")

d1d_matches = [marker for marker, release in markers if release == EXPECTED_RELEASE]
if len(d1d_matches) == 1:
    print(f"D1D_RELEASE_FOUND: {d1d_matches[0].parent}")
    print("D1D_MOUNT_DIAGNOSTIC_READY")
elif len(d1d_matches) == 0:
    print("D1D_RELEASE_NOT_FOUND")
    print("Causa probable: el dataset D1D no está adjunto/actualizado, o se subió el ZIP sin extraer su contenido.")
    print("No se editó notebook, no se activó GPU y no se inició entrenamiento.")
else:
    raise RuntimeError("Se encontraron varios marcadores D1D; no se puede seleccionar uno automáticamente.")
