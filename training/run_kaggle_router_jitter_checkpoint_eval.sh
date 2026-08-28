#!/usr/bin/env bash
set -euo pipefail

# Evaluación aislada: no entrena, no reanuda y no abre corpus ni holdout.
ROOT="${SOURCE_ROOT:?SOURCE_ROOT debe apuntar al código limpio montado}"
CHECKPOINT="${CHECKPOINT_PATH:-/kaggle/working/aethel-direct-train-router-jitter-v1/latest.pt}"
OUTPUT="${OUTPUT_ROOT:-/kaggle/working/aethel-direct-train-router-jitter-v1-inference-check-v1}"

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Falta el checkpoint recuperable: $CHECKPOINT" >&2
  exit 2
fi
if [[ ! -f "$(dirname "$CHECKPOINT")/tokenizer.json" ]]; then
  echo "Falta el tokenizer.json junto al checkpoint: $(dirname "$CHECKPOINT")/tokenizer.json" >&2
  exit 2
fi
if [[ -e "$OUTPUT" ]]; then
  echo "La salida de evaluación ya existe y no se reutiliza: $OUTPUT" >&2
  exit 2
fi

python3 "$ROOT/training/evaluate_router_jitter_checkpoint.py" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --device cuda \
  --max-new-tokens 32

RECEIPT="$OUTPUT/checkpoint_generation_receipt.json"
if [[ ! -f "$RECEIPT" ]]; then
  echo "Falta el recibo de evaluación: $RECEIPT" >&2
  exit 2
fi
grep -q '"status": "CHECKPOINT_GENERATION_READY"' "$RECEIPT"
printf 'AETHEL_ROUTER_JITTER_CHECKPOINT_EVALUATION_COMPLETE\nRECEIPT=%s\n' "$RECEIPT"
