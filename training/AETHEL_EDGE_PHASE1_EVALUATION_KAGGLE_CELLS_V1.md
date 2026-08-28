# Aethel Edge — evaluación aislada de fase 1

> **Alcance autorizado:** carga de `latest.pt` exclusivamente para evaluación, lectura de un máximo de 256 segmentos EN y 256 ES del holdout preparado y dos generaciones de 32 tokens. No hay entrenamiento, backward, optimizador, reanudación, red, promoción ni subida de pesos.

Usa un cuaderno nuevo llamado **`Aethel — Evaluación Edge Fase 1`** con exactamente estos tres inputs privados: `aethel-direct-train-source-v1` (release `edge-phase1-canonical-artifact-evaluation-v1`), `aethel-edge-corpus-v1` y `aethel-edge-phase1-artifacts-v1`. El cuaderno de entrenamiento original conserva sus dos inputs y no se modifica.

## CELDA 1 — resolver los tres inputs, sin cargar pesos

```python
# CELDA 1 — Aethel Edge: resolver inputs de evaluación, sin cargar pesos ni entrenar
from pathlib import Path
import json

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
EXPECTED_REVISION = "edge-phase1-canonical-artifact-evaluation-v1"

def unique_marker(filename, expected):
    found = []
    for path in INPUT_ROOT.rglob(filename):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("revision") == expected:
            found.append(path.parent.parent)
    if len(found) != 1:
        raise RuntimeError(f"Se esperaba exactamente un release {expected}; candidatos: {found}")
    return found[0]

SOURCE_ROOT = unique_marker("aethel_direct_train_source_release.json", EXPECTED_REVISION)
ARTIFACT_DATASET = "aethel-edge-phase1-artifacts-v1"
CANONICAL_ARTIFACT_FOLDER = "aethel-edge-phase-1-183680-v1"
artifact_candidates = []
excluded_tar_copies = []
for path in INPUT_ROOT.rglob("latest.pt"):
    root = path.parent
    if root.name != CANONICAL_ARTIFACT_FOLDER or not (root / "tokenizer.json").is_file():
        continue
    if ARTIFACT_DATASET not in str(root):
        continue
    if any("preservation" in part for part in root.parts):
        excluded_tar_copies.append(root)
        continue
    artifact_candidates.append(root)
if len(artifact_candidates) != 1:
    raise RuntimeError(
        "Se esperaba exactamente un checkpoint canónico de artefactos Edge; "
        f"canónicos={artifact_candidates}; copias_TAR_excluidas={excluded_tar_copies}"
    )
EDGE_ARTIFACT_ROOT = artifact_candidates[0]
data_candidates = [path.parent for path in INPUT_ROOT.rglob("prepared_manifest.json") if "aethel-edge-corpus-v1" in str(path)]
if len(data_candidates) != 1:
    raise RuntimeError(f"Se esperaba exactamente un corpus Edge; candidatos: {data_candidates}")
EDGE_DATA_ROOT = data_candidates[0]
if not any((EDGE_DATA_ROOT / name).is_file() for name in ("validation.jsonl", "validation.jsonl.gz")):
    raise RuntimeError("No existe validation.jsonl compatible en el corpus Edge.")
EVALUATION_OUTPUT = WORK_ROOT / "aethel-edge-phase-1-evaluation-v1"
if EVALUATION_OUTPUT.exists():
    raise RuntimeError(f"La salida ya existe; no se reutiliza: {EVALUATION_OUTPUT}")
print("CELDA 1 — INPUTS_EDGE_EVALUATION_READY")
print("SOURCE_ROOT:", SOURCE_ROOT)
print("EDGE_ARTIFACT_ROOT:", EDGE_ARTIFACT_ROOT)
print("TAR_COPIES_EXCLUDED:", excluded_tar_copies)
print("EDGE_DATA_ROOT:", EDGE_DATA_ROOT)
print("EVALUATION_OUTPUT:", EVALUATION_OUTPUT)
print("No se cargaron pesos, no se leyó holdout y no se entrenó.")
```

## CELDA 2 — preflight GPU, sin cargar pesos

```python
# CELDA 2 — Aethel Edge: preflight de evaluación, sin cargar pesos ni entrenar
import torch

if not torch.cuda.is_available():
    raise RuntimeError("CUDA no está disponible; no se iniciará la evaluación.")
print("CELDA 2 — AETHEL_EDGE_EVALUATION_PRECHECK_READY")
print("CUDA_AVAILABLE:", torch.cuda.is_available())
print("GPU_COUNT:", torch.cuda.device_count())
print("GPU_0:", torch.cuda.get_device_name(0))
print("EVALUATION_LIMITS: 256 segmentos EN + 256 segmentos ES; 2 generaciones de 32 tokens")
print("No se cargaron pesos, no se leyó holdout y no se entrenó.")
```

## CELDA 3 — evaluación autorizada, sin entrenamiento

```python
# CELDA 3 — Aethel Edge: evaluación aislada autorizada; no entrena ni reanuda
import os
import subprocess

environment = os.environ.copy()
environment.update({
    "SOURCE_ROOT": str(SOURCE_ROOT),
    "EDGE_ARTIFACT_ROOT": str(EDGE_ARTIFACT_ROOT),
    "EDGE_DATA_ROOT": str(EDGE_DATA_ROOT),
    "EVALUATION_OUTPUT": str(EVALUATION_OUTPUT),
})
subprocess.run(
    ["bash", str(SOURCE_ROOT / "training" / "run_kaggle_edge_checkpoint_eval_v1.sh")],
    check=True,
    env=environment,
)
receipt = EVALUATION_OUTPUT / "edge_evaluation_receipt.json"
print(receipt.read_text(encoding="utf-8"))
```

Al finalizar, usa **Save Version → Save & Run All** sólo si lanzas la evaluación como ejecución guardada. Después comparte el contenido de `edge_evaluation_receipt.json`; no inicies una sesión de entrenamiento 2 hasta interpretar los resultados.
