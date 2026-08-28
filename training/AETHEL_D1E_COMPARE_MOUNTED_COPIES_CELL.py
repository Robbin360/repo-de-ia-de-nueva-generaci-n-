# =============================================================================
# CELDA D1E-COMPARE — comparar copias montadas sin leer corpus, holdout ni pesos
# Sólo inspecciona nombres, tamaños y hashes de archivos de código/contrato.
# =============================================================================

from __future__ import annotations

import hashlib
import json
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
EXPECTED_RELEASE = "d1e-v1-router-entropy-strength-train-only"
REQUIRED = (
    "training/aethel_d1e_source_release.json",
    "training/run_kaggle_d1e_router_entropy_strength_diagnostic.sh",
    "training/AETHEL_D1E_ROUTER_ENTROPY_EXECUTION_CELL.py",
    "training/summarize_d1a_router_metrics.py",
    "engine/train_aethel_gpu.py",
)
FORBIDDEN_SUFFIXES = {".pt", ".pth", ".ckpt", ".safetensors", ".jsonl", ".pyc"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


markers = []
for marker in sorted(INPUT_ROOT.rglob("aethel_d1e_source_release.json")):
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        continue
    if payload.get("release") != EXPECTED_RELEASE:
        continue
    root = marker.parent.parent
    required = {item: (root / item).is_file() for item in REQUIRED}
    forbidden = sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in FORBIDDEN_SUFFIXES
    )
    marker_record = {
        "root": str(root),
        "marker_sha256": sha256(marker),
        "required_files": required,
        "required_count": sum(required.values()),
        "forbidden_artifact_count": len(forbidden),
        "forbidden_artifacts": forbidden[:10],
    }
    markers.append(marker_record)

print("CELDA D1E-COMPARE — comparación segura de copias montadas")
print(json.dumps(markers, ensure_ascii=False, indent=2))
if len(markers) != 2:
    raise RuntimeError(f"Se esperaban exactamente 2 copias D1E; se encontraron {len(markers)}.")
if any(item["forbidden_artifact_count"] for item in markers):
    raise RuntimeError("Se detectaron artefactos protegidos; no se seleccionó ninguna copia.")
print("D1E_MOUNT_COMPARISON_READY")
print("No se leyó corpus/holdout, no se cargaron pesos y no se ejecutó entrenamiento.")
