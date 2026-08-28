# =============================================================================
# CELDA 12 — D1D: diagnóstico train-only con regularización de entropía densa
# Autorización recibida: una única corrida nueva, 768 pasos, sin holdout ni reanudación.
# =============================================================================
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

D1D_EXECUTION_ENABLED = True
D1D_RUN_CONFIRMATION = "APPROVED_D1D_RUN"
D1D_GPU_CONFIRMATION = "APPROVED_D1D_GPU"
D1D_FINAL_EXECUTION_TOKEN = "APPROVED_FINAL_D1D_EXECUTION"
D1D_PYTORCH_FALLBACK_CONFIRMATION = "APPROVED_D1D_PYTORCH_FALLBACK"

EXPECTED_RELEASE = "d1d-v1-router-entropy-train-only"
DATA_DATASET = "aethel-nextgen-data-v1"
INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working/aethel-d1d-router-entropy-001-run-v1")
SOURCE_WORK = WORK_ROOT / "aethel-nextgen-source"
OUTPUT_DIR = WORK_ROOT / "output"


def resolve_source() -> Path:
    matches: list[Path] = []
    for marker in sorted(INPUT_ROOT.rglob("aethel_d1d_v5_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        root = marker.parent.parent
        if (
            payload.get("release") == EXPECTED_RELEASE
            and payload.get("d1d_execution_authorized") is False
            and payload.get("training_authorized") is False
            and (root / "training" / "run_kaggle_d1d_router_entropy_diagnostic.sh").is_file()
            and (root / "engine" / "train_aethel_gpu.py").is_file()
        ):
            matches.append(root)
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente el release D1D '{EXPECTED_RELEASE}'. Candidatos:\n{rendered}\n"
            "No se inició GPU ni entrenamiento."
        )
    return matches[0]


def resolve_data() -> Path:
    matches = sorted(
        manifest.parent
        for manifest in INPUT_ROOT.rglob("package_manifest.json")
        if DATA_DATASET in str(manifest) and (manifest.parent / "tokenizer.json").is_file()
    )
    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente el Dataset '{DATA_DATASET}'. Candidatos:\n{rendered}\n"
            "No se inició GPU ni entrenamiento."
        )
    return matches[0]


print("CELDA 12 — D1D: diagnóstico train-only con regularización de entropía")
source_input = resolve_source()
print(f"SOURCE_INPUT_D1D: {source_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")

approved = {
    "D1D_EXECUTION_ENABLED": D1D_EXECUTION_ENABLED,
    "D1D_RUN_CONFIRMATION": D1D_RUN_CONFIRMATION,
    "D1D_GPU_CONFIRMATION": D1D_GPU_CONFIRMATION,
    "D1D_FINAL_EXECUTION_TOKEN": D1D_FINAL_EXECUTION_TOKEN,
    "D1D_PYTORCH_FALLBACK_CONFIRMATION": D1D_PYTORCH_FALLBACK_CONFIRMATION,
}
expected = {
    "D1D_EXECUTION_ENABLED": True,
    "D1D_RUN_CONFIRMATION": "APPROVED_D1D_RUN",
    "D1D_GPU_CONFIRMATION": "APPROVED_D1D_GPU",
    "D1D_FINAL_EXECUTION_TOKEN": "APPROVED_FINAL_D1D_EXECUTION",
    "D1D_PYTORCH_FALLBACK_CONFIRMATION": "APPROVED_D1D_PYTORCH_FALLBACK",
}
if approved != expected:
    raise RuntimeError("D1D bloqueado: faltan autorizaciones exactas.")
if WORK_ROOT.exists():
    raise RuntimeError(f"D1D bloqueado: la salida ya existe: {WORK_ROOT}. No se reutiliza ni se borra.")

data_input = resolve_data()
shutil.copytree(source_input, SOURCE_WORK)
os.environ["AETHEL_SOURCE_DIR"] = str(SOURCE_WORK)
os.environ["AETHEL_DATA_DIR"] = str(data_input)
os.environ["AETHEL_D1D_OUTPUT_DIR"] = str(OUTPUT_DIR)
os.environ["AETHEL_D1D_RUN_AUTHORIZED"] = "YES"
os.environ["AETHEL_D1D_GPU_AUTHORIZED"] = "YES"
os.environ["AETHEL_D1D_ALLOW_PYTORCH_FALLBACK"] = "YES"
os.environ.pop("AETHEL_RESUME_CHECKPOINT", None)

launcher = SOURCE_WORK / "training" / "run_kaggle_d1d_router_entropy_diagnostic.sh"
completed = subprocess.run(["bash", str(launcher)], text=True, capture_output=True)
print(completed.stdout)
if completed.stderr:
    print(completed.stderr)
if completed.returncode != 0 or "D1D_DIAGNOSTIC_COMPLETE" not in completed.stdout:
    raise RuntimeError("D1D no completó. No ejecutes holdout ni reanudación; comparte la salida completa.")
print("D1D_DIAGNOSTIC_COMPLETE")
