# =============================================================================
# CELDA 11 — D1D: verificar el release de entropía en modo bloqueado
# Propósito: resolver únicamente el bundle D1D V5. No edita inputs, no elige
# GPU, no copia código, no lee Dataset, no carga pesos y no entrena.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path


D1D_EXECUTION_ENABLED = False
D1D_NOTEBOOK_EDIT_CONFIRMATION = "PENDING_D1D_NOTEBOOK_EDIT"
D1D_RUN_CONFIRMATION = "PENDING_D1D_TRAIN_ONLY_RUN"
D1D_GPU_CONFIRMATION = "PENDING_D1D_GPU"
D1D_FINAL_TOKEN = "PENDING_FINAL_D1D"
D1D_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1D_PYTORCH_FALLBACK"

SOURCE_DATASET = Path("/kaggle/input")
EXPECTED_RELEASE = "d1d-v1-router-entropy-train-only"


def resolve_d1d_source(source_dataset: Path) -> Path:
    matches: list[Path] = []
    for marker in sorted(source_dataset.rglob("aethel_d1d_v5_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        # El marcador vive en <repo>/training/; la raíz del bundle es dos niveles arriba.
        source_root = marker.parent.parent
        if (
            payload.get("release") == EXPECTED_RELEASE
            and payload.get("d1d_execution_authorized") is False
            and payload.get("training_authorized") is False
            and payload.get("notebook_edit_authorized") is False
            and payload.get("gpu_authorized") is False
            and (source_root / "engine" / "train_aethel_gpu.py").is_file()
            and (source_root / "engine" / "router_auxiliary.py").is_file()
            and (source_root / "training" / "AETHEL_D1D_ROUTER_ENTROPY_PROTOCOL_2026-08-25.md").is_file()
        ):
            matches.append(source_root)
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente el release D1D '{EXPECTED_RELEASE}'. Candidatos:\n"
            f"{rendered}\nNo se editó notebook, no se activó GPU y no se inició entrenamiento."
        )
    return matches[0]


source_input = resolve_d1d_source(SOURCE_DATASET)
print("CELDA 11 — D1D: verificar el release de entropía en modo bloqueado")
print(f"SOURCE_INPUT_D1D: {source_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")

pending_values = {
    "D1D_EXECUTION_ENABLED": D1D_EXECUTION_ENABLED,
    "D1D_NOTEBOOK_EDIT_CONFIRMATION": D1D_NOTEBOOK_EDIT_CONFIRMATION,
    "D1D_RUN_CONFIRMATION": D1D_RUN_CONFIRMATION,
    "D1D_GPU_CONFIRMATION": D1D_GPU_CONFIRMATION,
    "D1D_FINAL_TOKEN": D1D_FINAL_TOKEN,
    "D1D_PYTORCH_FALLBACK_CONFIRMATION": D1D_PYTORCH_FALLBACK_CONFIRMATION,
}
approved_values = {
    "D1D_EXECUTION_ENABLED": True,
    "D1D_NOTEBOOK_EDIT_CONFIRMATION": "APPROVED_D1D_NOTEBOOK_EDIT",
    "D1D_RUN_CONFIRMATION": "APPROVED_D1D_TRAIN_ONLY_RUN",
    "D1D_GPU_CONFIRMATION": "APPROVED_D1D_GPU",
    "D1D_FINAL_TOKEN": "APPROVED_FINAL_D1D",
    "D1D_PYTORCH_FALLBACK_CONFIRMATION": "APPROVED_D1D_PYTORCH_FALLBACK",
}

if pending_values != approved_values:
    print("D1D_CELL_PREPARED_NOT_EXECUTED")
    print("D1D_PENDING_NOTEBOOK_EDIT_AND_RUN_AUTHORIZATION")
    print(
        "D1D permanece bloqueada: no se editó notebook, no se seleccionó GPU, "
        "no se leyó Dataset train/holdout, no se cargaron pesos y no se entrenó."
    )
else:
    raise RuntimeError(
        "Esta CELDA 11 es sólo de verificación bloqueada. La ejecución requiere "
        "una autorización inmediata separada y una celda ejecutable nueva."
    )
