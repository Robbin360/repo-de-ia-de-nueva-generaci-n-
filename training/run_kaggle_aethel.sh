#!/usr/bin/env bash
set -euo pipefail

# Ejecutar dentro de un Kaggle Notebook con Accelerator=GPU.
# Monte el código como un Kaggle Dataset de entrada o cópielo al working directory.
: "${AETHEL_SOURCE_DIR:=/kaggle/input/aethel-source}"
: "${AETHEL_WORK_DIR:=/kaggle/working/aethel}"
: "${AETHEL_DATA_DIR:=/kaggle/input/aethel-data}"
: "${AETHEL_OUTPUT_DIR:=/kaggle/working/aethel-runs/nextgen-pilot}"
: "${AETHEL_KAGGLE_DATASET:=}"

if [[ -z "$AETHEL_KAGGLE_DATASET" ]]; then
  echo "Falta AETHEL_KAGGLE_DATASET=usuario/dataset-privado; no se permite iniciar sin persistencia de artefactos." >&2
  exit 2
fi

rm -rf "$AETHEL_WORK_DIR"
cp -R "$AETHEL_SOURCE_DIR" "$AETHEL_WORK_DIR"
cd "$AETHEL_WORK_DIR"

python -m pip install --quiet -r training/requirements.txt
python - <<'PY'
import torch
assert torch.cuda.is_available(), "Activa Accelerator=GPU en la configuración del Kaggle Notebook."
print({"gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)})
PY

# BF16 requiere una GPU Ampere o posterior. P100/T4 usan FP16; se puede forzar con AETHEL_PRECISION.
if [[ -z "${AETHEL_PRECISION:-}" ]]; then
  AETHEL_PRECISION=$(python - <<'PY'
import torch
from engine.select_precision import select_precision
print(select_precision(torch.cuda.get_device_capability(0)[0]))
PY
)
fi
echo "Precision elegida: $AETHEL_PRECISION"

# El corpus preparado debe venir de un manifiesto aprobado. No descargues fuentes masivas en una sesión gratuita.
test -f "$AETHEL_DATA_DIR/prepared/prepared_manifest.json"
test -f "$AETHEL_DATA_DIR/tokenizer/aethel-bpe.json"

python engine/train_aethel_gpu.py \
  --corpus-dir "$AETHEL_DATA_DIR/prepared" \
  --tokenizer "$AETHEL_DATA_DIR/tokenizer/aethel-bpe.json" \
  --output "$AETHEL_OUTPUT_DIR" \
  --max-steps "${AETHEL_MAX_STEPS:-12000}" \
  --dim "${AETHEL_DIM:-512}" \
  --layers "${AETHEL_LAYERS:-4}" \
  --heads "${AETHEL_HEADS:-8}" \
  --kv-heads "${AETHEL_KV_HEADS:-2}" \
  --experts "${AETHEL_EXPERTS:-8}" \
  --active-experts 2 \
  --seq-len "${AETHEL_SEQ_LEN:-1024}" \
  --batch-size "${AETHEL_BATCH_SIZE:-2}" \
  --gradient-accumulation "${AETHEL_GRAD_ACCUM:-16}" \
  --precision "$AETHEL_PRECISION" \
  --save-every "${AETHEL_SAVE_EVERY:-500}" \
  --resume

# Versiona automáticamente el resultado en un Dataset privado del dueño del Notebook.
# El Kaggle API token debe existir solo en los secretos del Notebook. No se escribe en Git.
python engine/export_artifacts.py \
  --source "$AETHEL_OUTPUT_DIR" \
  --mode kaggle-dataset \
  --staging /kaggle/working/aethel-persist-staging \
  --dataset "$AETHEL_KAGGLE_DATASET"
