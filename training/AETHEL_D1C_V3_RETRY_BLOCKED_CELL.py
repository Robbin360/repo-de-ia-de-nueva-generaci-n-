# =============================================================================
# CELDA 9 — D1C V3: verificar el release de retry en modo bloqueado
# Propósito: confirmar únicamente el release V3 que contiene la plantilla de
# retry. Las cinco puertas quedan cerradas; esta celda no ejecuta un retry.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path


# Cinco puertas finales, deliberadamente cerradas.
D1C_V3_RETRY_ENABLED = False
D1C_V3_RETRY_RUN_CONFIRMATION = "PENDING_D1C_V3_RETRY_RUN"
D1C_V3_RETRY_GPU_CONFIRMATION = "PENDING_D1C_V3_RETRY_GPU"
D1C_V3_RETRY_FINAL_TOKEN = "PENDING_FINAL_D1C_V3_RETRY"
D1C_V3_RETRY_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1C_V3_RETRY_PYTORCH_FALLBACK"

SOURCE_DATASET = Path("/kaggle/input/datasets/felixtremigual/aethel-nextgen-source-e0-v1")
EXPECTED_RELEASE = "d1c-v3-retry-cell-train-only"


def resolve_d1c_v3_source(source_dataset: Path) -> Path:
    matches: list[Path] = []
    for marker in sorted(source_dataset.rglob("aethel_kaggle_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_root = marker.parent
        if (
            payload.get("release") == EXPECTED_RELEASE
            and payload.get("d1c_v3_execution_authorized") is False
            and payload.get("training_authorized") is False
            and (source_root / "run_kaggle_d1c_router_aux_loss_diagnostic.sh").is_file()
            and (source_root / "AETHEL_D1C_V2_RETRY_EXECUTION_CELL.py").is_file()
        ):
            matches.append(source_root)
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente un release D1C V3 '{EXPECTED_RELEASE}'. Candidatos:\n"
            f"{rendered}\nNo se activó GPU, no se leyó Dataset y no se ejecutó retry."
        )
    return matches[0]


source_input = resolve_d1c_v3_source(SOURCE_DATASET)
print("CELDA 9 — D1C V3: verificar el release de retry en modo bloqueado")
print(f"SOURCE_INPUT_D1C_V3: {source_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")

pending_values = {
    "D1C_V3_RETRY_ENABLED": D1C_V3_RETRY_ENABLED,
    "D1C_V3_RETRY_RUN_CONFIRMATION": D1C_V3_RETRY_RUN_CONFIRMATION,
    "D1C_V3_RETRY_GPU_CONFIRMATION": D1C_V3_RETRY_GPU_CONFIRMATION,
    "D1C_V3_RETRY_FINAL_TOKEN": D1C_V3_RETRY_FINAL_TOKEN,
    "D1C_V3_RETRY_PYTORCH_FALLBACK_CONFIRMATION": D1C_V3_RETRY_PYTORCH_FALLBACK_CONFIRMATION,
}
approved_values = {
    "D1C_V3_RETRY_ENABLED": True,
    "D1C_V3_RETRY_RUN_CONFIRMATION": "APPROVED_D1C_V3_RETRY_RUN",
    "D1C_V3_RETRY_GPU_CONFIRMATION": "APPROVED_D1C_V3_RETRY_GPU",
    "D1C_V3_RETRY_FINAL_TOKEN": "APPROVED_FINAL_D1C_V3_RETRY",
    "D1C_V3_RETRY_PYTORCH_FALLBACK_CONFIRMATION": "APPROVED_D1C_V3_RETRY_PYTORCH_FALLBACK",
}

if pending_values != approved_values:
    print("D1C_V3_CELL_PREPARED_NOT_EXECUTED")
    print("D1C_V3_RETRY_PENDING_FINAL_AUTHORIZATION")
    print(
        "D1C V3 permanece bloqueada: no se seleccionó GPU, no se leyó Dataset train/holdout, "
        "no se copiaron archivos, no se cargaron pesos y no se entrenó."
    )
else:
    raise RuntimeError(
        "Esta CELDA 9 es sólo de verificación bloqueada. Una ejecución requeriría un protocolo "
        "nuevo, un release ejecutable distinto y autorizaciones inmediatas separadas."
    )
