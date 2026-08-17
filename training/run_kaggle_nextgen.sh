#!/usr/bin/env bash
set -euo pipefail

# Puerta explícita para una ejecución de Aethel NextGen desde un Kaggle Notebook.
# Este script no inicia la instalación, CUDA ni entrenamiento sin autorización final.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${AETHEL_SOURCE_DIR:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${AETHEL_DATA_DIR:=/kaggle/input/aethel-nextgen-data}"
: "${AETHEL_WORK_DIR:=/kaggle/working/aethel-nextgen}"
: "${AETHEL_OUTPUT_DIR:=/kaggle/working/aethel-runs/nextgen-pilot}"
: "${AETHEL_KAGGLE_DATASET:=}"
: "${AETHEL_RESUME_CHECKPOINT:=}"
: "${AETHEL_RUN_AUTHORIZED:=}"

if [[ "$AETHEL_RUN_AUTHORIZED" != "YES" ]]; then
  echo "BLOCKED: define AETHEL_RUN_AUTHORIZED=YES solo después de confirmar Save Version → Save & Run All." >&2
  exit 3
fi

legacy_paths="$AETHEL_SOURCE_DIR:$AETHEL_DATA_DIR:$AETHEL_RESUME_CHECKPOINT"
if [[ "$legacy_paths" == *"aethel-v3"* || "$legacy_paths" == *"aethel_v3"* ]]; then
  echo "BLOCKED: Aethel NextGen no acepta rutas o checkpoints de Aethel V3." >&2
  exit 4
fi

for required in \
  "$AETHEL_SOURCE_DIR/engine/train_aethel_gpu.py" \
  "$AETHEL_SOURCE_DIR/training/validate_training_readiness.py" \
  "$AETHEL_SOURCE_DIR/training/inspect_checkpoint.py" \
  "$AETHEL_SOURCE_DIR/TRAINING_CURRICULUM.md"; do
  if [[ ! -f "$required" ]]; then
    echo "BLOCKED: falta una fuente NextGen verificable: $required" >&2
    exit 5
  fi
done

export AETHEL_SOURCE_DIR AETHEL_DATA_DIR AETHEL_WORK_DIR AETHEL_OUTPUT_DIR
export AETHEL_KAGGLE_DATASET AETHEL_RESUME_CHECKPOINT
exec bash "$AETHEL_SOURCE_DIR/training/run_kaggle_aethel.sh"
