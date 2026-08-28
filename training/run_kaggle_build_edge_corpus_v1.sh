#!/usr/bin/env bash
set -euo pipefail

# Construye únicamente el corpus Edge aprobado. No entrena ni carga checkpoints.
: "${SOURCE_ROOT:?SOURCE_ROOT debe apuntar al bundle limpio de código}"
: "${EDGE_DATA_OUTPUT:?EDGE_DATA_OUTPUT debe ser una ruta inédita en /kaggle/working}"
: "${BASE_DATA_ROOT:?BASE_DATA_ROOT debe apuntar al dataset base que contiene tokenizer.json}"

MANIFEST="${SOURCE_ROOT}/training/aethel_edge_v1.manifest.json"
if [[ -e "$EDGE_DATA_OUTPUT" ]]; then
  echo "La salida de corpus ya existe y no se reutiliza: $EDGE_DATA_OUTPUT" >&2
  exit 2
fi
test -f "$MANIFEST"
test -f "${BASE_DATA_ROOT}/tokenizer.json"

python "${SOURCE_ROOT}/engine/prepare_bilingual_corpus.py" \
  --manifest "$MANIFEST" \
  --output "$EDGE_DATA_OUTPUT" \
  --shard-documents 25000 \
  --seed 17 \
  --approved-data-plan \
  --allow-network

test -f "${EDGE_DATA_OUTPUT}/prepared_manifest.json"
test -f "${EDGE_DATA_OUTPUT}/validation.jsonl.gz"
find "$EDGE_DATA_OUTPUT" -maxdepth 1 -name 'train-*.jsonl.gz' -print -quit | grep -q .
cp "${BASE_DATA_ROOT}/tokenizer.json" "${EDGE_DATA_OUTPUT}/tokenizer.json"
cmp --silent "${BASE_DATA_ROOT}/tokenizer.json" "${EDGE_DATA_OUTPUT}/tokenizer.json"
test -f "${EDGE_DATA_OUTPUT}/tokenizer.json"
printf 'AETHEL_EDGE_CORPUS_READY\nEDGE_DATA_OUTPUT=%s\n' "$EDGE_DATA_OUTPUT"
