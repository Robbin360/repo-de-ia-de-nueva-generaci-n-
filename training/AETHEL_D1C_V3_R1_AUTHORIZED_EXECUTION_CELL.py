# =============================================================================
# CELDA 10 — D1C V3-R1: retry aislado con perfil de release, ejecución autorizada
# Propósito: ejecutar una única corrida nueva desde inicialización nueva.
# Alcance autorizado: 768 pasos, train-only, GPU T4, fallback PyTorch, sin holdout.
# =============================================================================

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


# Seis puertas autorizadas de forma explícita y separada para esta única corrida.
D1C_V3_R1_RETRY_ENABLED = True
D1C_V3_R1_RETRY_RUN_CONFIRMATION = "APPROVED_D1C_V3_R1_RETRY_RUN"
D1C_V3_R1_RETRY_GPU_CONFIRMATION = "APPROVED_D1C_V3_R1_RETRY_GPU"
D1C_V3_R1_RETRY_FINAL_TOKEN = "APPROVED_FINAL_D1C_V3_R1_RETRY"
D1C_V3_R1_RETRY_PYTORCH_FALLBACK_CONFIRMATION = "APPROVED_D1C_V3_R1_RETRY_PYTORCH_FALLBACK"
D1C_V3_R1_RELEASE_PROFILE_CONFIRMATION = "APPROVED_D1C_V3_R1_RELEASE_PROFILE"

EXPECTED_RELEASE = "d1c-v4-v3-r1-launcher-profile-train-only"
SOURCE_DATASET = Path("/kaggle/input/datasets/felixtremigual/aethel-nextgen-source-e0-v1")
DATA_DATASET = "aethel-nextgen-data-v1"
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/aethel-d1c-v3-r1-retry-run")
SOURCE_WORK = WORK_ROOT / "aethel-nextgen-source"
OUTPUT_DIR = WORK_ROOT / "output"


def resolve_d1c_v3_r1_source(source_dataset: Path) -> Path:
    matches: list[Path] = []
    for marker in sorted(source_dataset.rglob("aethel_kaggle_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        training_root = marker.parent
        source_root = training_root.parent
        if (
            payload.get("release") == EXPECTED_RELEASE
            and payload.get("d1c_v3_r1_execution_authorized") is False
            and payload.get("training_authorized") is False
            and (training_root / "run_kaggle_d1c_router_aux_loss_diagnostic.sh").is_file()
            and (training_root / "summarize_d1a_router_metrics.py").is_file()
            and (training_root / "aethel_d1c_v3_r1_authorization_contract.json").is_file()
            and (source_root / "engine" / "train_aethel_gpu.py").is_file()
        ):
            matches.append(source_root)
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente un release D1C V3-R1 '{EXPECTED_RELEASE}'. Candidatos:\n"
            f"{rendered}\nNo se activó GPU, no se leyó Dataset y no se ejecutó retry."
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
            f"{rendered}\nNo se activó GPU ni retry."
        )
    return matches[0]


source_input = resolve_d1c_v3_r1_source(SOURCE_DATASET)
print("CELDA 10 — D1C V3-R1: retry aislado con perfil de release, ejecución autorizada")
print(f"SOURCE_INPUT_D1C_V3_R1: {source_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")

pending_values = {
    "D1C_V3_R1_RETRY_ENABLED": D1C_V3_R1_RETRY_ENABLED,
    "D1C_V3_R1_RETRY_RUN_CONFIRMATION": D1C_V3_R1_RETRY_RUN_CONFIRMATION,
    "D1C_V3_R1_RETRY_GPU_CONFIRMATION": D1C_V3_R1_RETRY_GPU_CONFIRMATION,
    "D1C_V3_R1_RETRY_FINAL_TOKEN": D1C_V3_R1_RETRY_FINAL_TOKEN,
    "D1C_V3_R1_RETRY_PYTORCH_FALLBACK_CONFIRMATION": D1C_V3_R1_RETRY_PYTORCH_FALLBACK_CONFIRMATION,
    "D1C_V3_R1_RELEASE_PROFILE_CONFIRMATION": D1C_V3_R1_RELEASE_PROFILE_CONFIRMATION,
}
approved_values = {
    "D1C_V3_R1_RETRY_ENABLED": True,
    "D1C_V3_R1_RETRY_RUN_CONFIRMATION": "APPROVED_D1C_V3_R1_RETRY_RUN",
    "D1C_V3_R1_RETRY_GPU_CONFIRMATION": "APPROVED_D1C_V3_R1_RETRY_GPU",
    "D1C_V3_R1_RETRY_FINAL_TOKEN": "APPROVED_FINAL_D1C_V3_R1_RETRY",
    "D1C_V3_R1_RETRY_PYTORCH_FALLBACK_CONFIRMATION": "APPROVED_D1C_V3_R1_RETRY_PYTORCH_FALLBACK",
    "D1C_V3_R1_RELEASE_PROFILE_CONFIRMATION": "APPROVED_D1C_V3_R1_RELEASE_PROFILE",
}

if pending_values != approved_values:
    raise RuntimeError("Las seis puertas de D1C V3-R1 no están aprobadas; no se ejecuta retry.")
elif WORK_ROOT.exists():
    raise RuntimeError(
        f"D1C V3-R1 retry bloqueado: el directorio de trabajo ya existe: {WORK_ROOT}. "
        "No se borra ni se reutiliza una salida previa."
    )
else:
    data_input = resolve_data_root()
    shutil.copytree(source_input, SOURCE_WORK)
    os.environ["AETHEL_SOURCE_DIR"] = str(SOURCE_WORK)
    os.environ["AETHEL_DATA_DIR"] = str(data_input)
    os.environ["AETHEL_D1C_OUTPUT_DIR"] = str(OUTPUT_DIR)
    os.environ["AETHEL_D1C_EXPECTED_RELEASE"] = EXPECTED_RELEASE
    os.environ["AETHEL_D1C_RELEASE_PROFILE_AUTHORIZED"] = "YES"
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
            "D1C V3-R1 retry no completó. No evalúes holdout, no reanudes pesos y no inicies D2; "
            "comparte el output completo para revisión."
        )
    print("D1C_DIAGNOSTIC_COMPLETE")
