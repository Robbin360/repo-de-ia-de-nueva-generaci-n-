# =============================================================================
# CELDA 7 — D1C: diagnóstico de peso auxiliar del router, variante habilitable
# Propósito: preparar una única corrida D1C desde inicialización nueva y sólo train.
# Estado de esta plantilla: cinco puertas cerradas; no debe ejecutarse como diagnóstico.
# Convención: toda celda futura preparada para este notebook debe comenzar con
# “CELDA <número> — <propósito>” y conservar un estado explícito.
# =============================================================================

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


# Cinco puertas finales, deliberadamente cerradas en la variante entregable.
D1C_EXECUTION_ENABLED = False
D1C_RUN_CONFIRMATION = "PENDING_D1C_RUN"
D1C_GPU_CONFIRMATION = "PENDING_D1C_GPU"
D1C_FINAL_EXECUTION_TOKEN = "PENDING_FINAL_D1C_EXECUTION"
D1C_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1C_PYTORCH_FALLBACK"

EXPECTED_RELEASE = "d1c-v1-router-aux-loss-005-train-only"
SOURCE_DATASET = "aethel-nextgen-source-e0-v1"
DATA_DATASET = "aethel-nextgen-data-v1"
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/aethel-d1c-router-aux-loss-005-run-v1")
SOURCE_WORK = WORK_ROOT / "aethel-nextgen-source"
OUTPUT_DIR = WORK_ROOT / "output"


def resolve_d1c_source() -> Path:
    matches: list[Path] = []
    for marker in sorted(INPUT_ROOT.rglob("aethel_kaggle_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_root = marker.parent.parent
        if (
            SOURCE_DATASET in str(marker)
            and payload.get("release") == EXPECTED_RELEASE
            and payload.get("d1c_execution_authorized") is False
            and (source_root / "training" / "run_kaggle_d1c_router_aux_loss_diagnostic.sh").is_file()
            and (source_root / "training" / "validate_aethel_train_only_mount.py").is_file()
            and (source_root / "training" / "summarize_d1a_router_metrics.py").is_file()
        ):
            matches.append(source_root)
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente un release D1C '{EXPECTED_RELEASE}'. Candidatos:\n"
            f"{rendered}\nNo se activó GPU ni entrenamiento."
        )
    return matches[0]


def resolve_data_root() -> Path:
    matches = sorted(
        manifest.parent
        for manifest in INPUT_ROOT.rglob("package_manifest.json")
        if DATA_DATASET in str(manifest) and (manifest.parent / "tokenizer.json").is_file()
    )
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente un Dataset de datos '{DATA_DATASET}'. Candidatos:\n"
            f"{rendered}\nNo se activó GPU ni entrenamiento."
        )
    return matches[0]


source_input = resolve_d1c_source()
print(f"SOURCE_INPUT_D1C: {source_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")

pending_values = {
    "D1C_EXECUTION_ENABLED": D1C_EXECUTION_ENABLED,
    "D1C_RUN_CONFIRMATION": D1C_RUN_CONFIRMATION,
    "D1C_GPU_CONFIRMATION": D1C_GPU_CONFIRMATION,
    "D1C_FINAL_EXECUTION_TOKEN": D1C_FINAL_EXECUTION_TOKEN,
    "D1C_PYTORCH_FALLBACK_CONFIRMATION": D1C_PYTORCH_FALLBACK_CONFIRMATION,
}
approved_values = {
    "D1C_EXECUTION_ENABLED": True,
    "D1C_RUN_CONFIRMATION": "APPROVED_D1C_RUN",
    "D1C_GPU_CONFIRMATION": "APPROVED_D1C_GPU",
    "D1C_FINAL_EXECUTION_TOKEN": "APPROVED_FINAL_D1C_EXECUTION",
    "D1C_PYTORCH_FALLBACK_CONFIRMATION": "APPROVED_D1C_PYTORCH_FALLBACK",
}

if pending_values != approved_values:
    print("D1C_EXECUTION_PENDING_FINAL_AUTHORIZATION")
    print(
        "D1C permanece bloqueada: no se seleccionó GPU, no se copió código, "
        "no se leyó Dataset train/holdout, no se cargaron pesos y no se entrenó."
    )
elif WORK_ROOT.exists():
    raise RuntimeError(
        f"D1C bloqueada: el directorio de trabajo ya existe: {WORK_ROOT}. "
        "No se borra ni se reutiliza una salida previa."
    )
else:
    # Esta rama sigue requiriendo confirmaciones inmediatas distintas de esta plantilla.
    # D1C inicia desde cero, con sólo train y fallback PyTorch experimental autorizado.
    data_input = resolve_data_root()
    shutil.copytree(source_input, SOURCE_WORK)
    os.environ["AETHEL_SOURCE_DIR"] = str(SOURCE_WORK)
    os.environ["AETHEL_DATA_DIR"] = str(data_input)
    os.environ["AETHEL_D1C_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["AETHEL_D1C_RUN_AUTHORIZED"] = "YES"
    os.environ["AETHEL_D1C_GPU_AUTHORIZED"] = "YES"
    os.environ["AETHEL_D1C_ALLOW_PYTORCH_FALLBACK"] = "YES"
    os.environ.pop("AETHEL_RESUME_CHECKPOINT", None)

    launcher = SOURCE_WORK / "training" / "run_kaggle_d1c_router_aux_loss_diagnostic.sh"
    completed = subprocess.run(["bash", str(launcher)], text=True, capture_output=True)
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    if completed.returncode != 0 or "D1C_DIAGNOSTIC_COMPLETE" not in completed.stdout:
        raise RuntimeError(
            "D1C no completó. No evalúes holdout, no reanudes pesos y no inicies D2; "
            "comparte el output completo para revisión."
        )
    print("D1C_DIAGNOSTIC_COMPLETE")
