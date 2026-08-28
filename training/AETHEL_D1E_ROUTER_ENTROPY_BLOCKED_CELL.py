# =============================================================================
# CELDA D1E — verificar el release de entropía reforzada en modo bloqueado
# Propósito: resolver sólo el bundle D1E local. No edita inputs, no elige GPU,
# no copia código, no lee Dataset, no carga pesos y no entrena.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path


D1E_EXECUTION_ENABLED = False
D1E_NOTEBOOK_EDIT_CONFIRMATION = "PENDING_D1E_NOTEBOOK_EDIT"
D1E_RUN_CONFIRMATION = "PENDING_D1E_TRAIN_ONLY_RUN"
D1E_GPU_CONFIRMATION = "PENDING_D1E_GPU"
D1E_FINAL_TOKEN = "PENDING_FINAL_D1E"
D1E_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1E_PYTORCH_FALLBACK"

SOURCE_DATASET = Path("/kaggle/input")
EXPECTED_RELEASE = "d1e-v1-router-entropy-strength-train-only"
EXPECTED_ENTROPY_WEIGHT = 0.03


def resolve_d1e_source(source_dataset: Path) -> Path:
    matches: list[Path] = []
    for marker in sorted(source_dataset.rglob("aethel_d1e_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_root = marker.parent.parent
        if (
            payload.get("release") == EXPECTED_RELEASE
            and payload.get("router_entropy_loss_weight") == EXPECTED_ENTROPY_WEIGHT
            and payload.get("d1e_execution_authorized") is False
            and payload.get("training_authorized") is False
            and payload.get("notebook_edit_authorized") is False
            and payload.get("gpu_authorized") is False
            and (source_root / "engine" / "train_aethel_gpu.py").is_file()
            and (source_root / "engine" / "router_auxiliary.py").is_file()
            and (source_root / "training" / "AETHEL_D1E_ROUTER_ENTROPY_STRENGTH_PROTOCOL_2026-08-25.md").is_file()
        ):
            matches.append(source_root)
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente el release D1E '{EXPECTED_RELEASE}'. Candidatos:\n"
            f"{rendered}\nNo se editó notebook, no se activó GPU y no se inició entrenamiento."
        )
    return matches[0]


source_input = resolve_d1e_source(SOURCE_DATASET)
print("CELDA D1E — verificar el release de entropía reforzada en modo bloqueado")
print(f"SOURCE_INPUT_D1E: {source_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")
print(f"ROUTER_ENTROPY_LOSS_WEIGHT: {EXPECTED_ENTROPY_WEIGHT}")

pending_values = {
    "D1E_EXECUTION_ENABLED": D1E_EXECUTION_ENABLED,
    "D1E_NOTEBOOK_EDIT_CONFIRMATION": D1E_NOTEBOOK_EDIT_CONFIRMATION,
    "D1E_RUN_CONFIRMATION": D1E_RUN_CONFIRMATION,
    "D1E_GPU_CONFIRMATION": D1E_GPU_CONFIRMATION,
    "D1E_FINAL_TOKEN": D1E_FINAL_TOKEN,
    "D1E_PYTORCH_FALLBACK_CONFIRMATION": D1E_PYTORCH_FALLBACK_CONFIRMATION,
}
approved_values = {
    "D1E_EXECUTION_ENABLED": True,
    "D1E_NOTEBOOK_EDIT_CONFIRMATION": "APPROVED_D1E_NOTEBOOK_EDIT",
    "D1E_RUN_CONFIRMATION": "APPROVED_D1E_TRAIN_ONLY_RUN",
    "D1E_GPU_CONFIRMATION": "APPROVED_D1E_GPU",
    "D1E_FINAL_TOKEN": "APPROVED_FINAL_D1E",
    "D1E_PYTORCH_FALLBACK_CONFIRMATION": "APPROVED_D1E_PYTORCH_FALLBACK",
}

if pending_values != approved_values:
    print("D1E_CELL_PREPARED_NOT_EXECUTED")
    print("D1E_PENDING_NOTEBOOK_EDIT_AND_RUN_AUTHORIZATION")
    print(
        "D1E permanece bloqueada: no se editó notebook, no se seleccionó GPU, "
        "no se leyó Dataset train/holdout, no se cargaron pesos y no se entrenó."
    )
else:
    raise RuntimeError(
        "Esta celda sólo es de verificación bloqueada. La ejecución requiere "
        "una autorización inmediata separada y una celda ejecutable nueva."
    )
