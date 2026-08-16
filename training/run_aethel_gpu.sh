#!/usr/bin/env bash
set -euo pipefail

# Variables que debe definir el operador de la instancia GPU.
: "${AETHEL_DATA_DIR:?Defina AETHEL_DATA_DIR con el volumen persistente de datos}"
: "${AETHEL_RUN_DIR:?Defina AETHEL_RUN_DIR con el volumen persistente de checkpoints}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python -m pip install --upgrade pip
python -m pip install -r training/requirements.txt

# El manifiesto incluye fuentes desactivadas. Actívelas tras revisar sus licencias y límites de descarga.
python engine/prepare_corpus.py \
  --manifest "$AETHEL_DATA_DIR/corpus_manifest.json" \
  --output "$AETHEL_DATA_DIR/prepared" \
  --allow-network \
  --max-documents "${AETHEL_MAX_DOCUMENTS:?Defina un límite explícito de documentos}"

python engine/train_tokenizer.py \
  --corpus-dir "$AETHEL_DATA_DIR/prepared" \
  --output "$AETHEL_DATA_DIR/tokenizer/aethel-bpe.json" \
  --vocab-size "${AETHEL_VOCAB_SIZE:-32000}"

GPUS="${AETHEL_GPUS:-1}"
STRATEGY="${AETHEL_STRATEGY:-single}"
if [[ "$GPUS" -gt 1 && "$STRATEGY" == "single" ]]; then STRATEGY="ddp"; fi

torchrun --standalone --nproc_per_node="$GPUS" engine/train_aethel_gpu.py \
  --strategy "$STRATEGY" \
  --corpus-dir "$AETHEL_DATA_DIR/prepared" \
  --tokenizer "$AETHEL_DATA_DIR/tokenizer/aethel-bpe.json" \
  --output "$AETHEL_RUN_DIR" \
  --max-steps "${AETHEL_MAX_STEPS:-100000}" \
  --dim "${AETHEL_DIM:-1024}" \
  --layers "${AETHEL_LAYERS:-16}" \
  --heads "${AETHEL_HEADS:-16}" \
  --kv-heads "${AETHEL_KV_HEADS:-4}" \
  --experts "${AETHEL_EXPERTS:-8}" \
  --active-experts 2 \
  --seq-len "${AETHEL_SEQ_LEN:-2048}" \
  --batch-size "${AETHEL_BATCH_SIZE:-2}" \
  --gradient-accumulation "${AETHEL_GRAD_ACCUM:-16}" \
  --precision "${AETHEL_PRECISION:-bf16}" \
  --resume
