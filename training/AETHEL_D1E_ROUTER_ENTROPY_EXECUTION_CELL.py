# =============================================================================
# CELDA 12 — D1E: diagnóstico train-only con regularización de entropía reforzada
# Autorización: inicio fresco, 768 pasos, seed 17, sin checkpoint ni holdout.
# Esta celda ejecuta únicamente D1E y genera una salida nueva.
# =============================================================================

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
EXPECTED_RELEASE = "d1e-v1-router-entropy-strength-train-only"
EXPECTED_ENTROPY_WEIGHT = 0.03
EXPECTED_STEPS = 768
EXPECTED_SEED = 17
DIAGNOSTIC_ID = "D1E"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_source() -> Path:
    candidates: list[tuple[Path, str]] = []
    for marker in sorted(INPUT_ROOT.rglob("aethel_d1e_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_root = marker.parent.parent
        if (
            payload.get("release") == EXPECTED_RELEASE
            and payload.get("router_entropy_loss_weight") == EXPECTED_ENTROPY_WEIGHT
            and payload.get("d1e_execution_authorized") is False
            and (source_root / "engine" / "train_aethel_gpu.py").is_file()
            and (source_root / "training" / "run_kaggle_d1e_router_entropy_strength_diagnostic.sh").is_file()
            and (source_root / "training" / "summarize_d1a_router_metrics.py").is_file()
        ):
            candidates.append((source_root, sha256_file(marker)))
    if not candidates:
        raise RuntimeError(f"No se encontró el release D1E '{EXPECTED_RELEASE}'. No se inició entrenamiento.")
    if len({marker_hash for _, marker_hash in candidates}) != 1:
        rendered = "\n".join(f"- {path} ({marker_hash})" for path, marker_hash in candidates)
        raise RuntimeError(f"Hay releases D1E diferentes montados:\n{rendered}\nNo se inició entrenamiento.")
    return candidates[0][0]


def resolve_data_root() -> Path:
    candidates: list[tuple[Path, str]] = []
    for manifest in sorted(INPUT_ROOT.rglob("package_manifest.json")):
        root = manifest.parent
        if (root / "corpus").is_dir() and (root / "tokenizer.json").is_file():
            candidates.append((root, sha256_file(manifest)))
    if not candidates:
        raise RuntimeError("No se encontró el Dataset privado con corpus y tokenizer. No se inició entrenamiento.")
    if len({manifest_hash for _, manifest_hash in candidates}) != 1:
        rendered = "\n".join(f"- {path} ({manifest_hash})" for path, manifest_hash in candidates)
        raise RuntimeError(f"Hay Datasets diferentes montados:\n{rendered}\nNo se inició entrenamiento.")
    return candidates[0][0]


source_input = resolve_source()
data_input = resolve_data_root()
output_root = WORK_ROOT / "aethel-d1e-router-entropy-strength-v3"
if output_root.exists():
    entries = list(output_root.iterdir())
    rendered = "\n".join(f"- {entry}" for entry in entries[:20]) or "(vacía)"
    raise RuntimeError(
        f"La salida protegida ya existe: {output_root}. No se borra ni se sobrescribe.\n"
        f"Contenido detectado:\n{rendered}\nCambia OUTPUT_ROOT sólo a una ruta inédita después de conservar esta evidencia."
    )
output_root.mkdir(parents=True, exist_ok=False)

print("CELDA 12 — D1E: preflight integrado y diagnóstico train-only")
print(f"SOURCE_INPUT_D1E: {source_input}")
print(f"DATA_INPUT_D1E: {data_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")
print(f"ROUTER_ENTROPY_LOSS_WEIGHT: {EXPECTED_ENTROPY_WEIGHT}")
print(f"OUTPUT_ROOT_D1E: {output_root}")
print("D1E_EXECUTION_AUTHORIZED: true")
print("D1E_SCOPE: fresh-init train-only; no checkpoint; no holdout")

launcher = source_input / "training" / "run_kaggle_d1e_router_entropy_strength_diagnostic.sh"
env = os.environ.copy()
env.update(
    {
        "SOURCE_ROOT": str(source_input),
        "DATA_ROOT": str(data_input),
        "OUTPUT_ROOT": str(output_root),
    }
)

completed = subprocess.run(
    ["bash", str(launcher)],
    cwd=str(source_input),
    env=env,
    text=True,
    capture_output=True,
    check=False,
)
print(completed.stdout)
if completed.returncode != 0:
    print(completed.stderr)
    raise RuntimeError(
        f"D1E terminó con código {completed.returncode}; no se autoriza ninguna evaluación adicional."
    )

summary_path = output_root / "router_diagnostic.json"
if not summary_path.is_file():
    raise RuntimeError(
        f"No se encontró el resumen esperado {summary_path}; no se declara D1E completa."
    )

print(f"D1E_SUMMARY: {summary_path}")
print("D1E_DIAGNOSTIC_COMPLETE")
print("D1E límites: no holdout, no reanudación, no promoción, no serving.")
