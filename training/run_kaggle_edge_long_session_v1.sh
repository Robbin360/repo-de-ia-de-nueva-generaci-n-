#!/usr/bin/env bash
set -euo pipefail

# Primera fase Edge o continuación de una fase previamente preservada.
: "${SOURCE_ROOT:?SOURCE_ROOT debe apuntar al bundle limpio de código}"
: "${EDGE_DATA_ROOT:?EDGE_DATA_ROOT debe apuntar al corpus Edge preparado}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT debe ser una salida inédita}"
: "${PRESERVATION_PACKAGE:?PRESERVATION_PACKAGE debe ser un TAR inexistente}"
: "${PHASE_ID:?PHASE_ID identifica la fase Edge}"
: "${SESSION_TARGET_STEP:?SESSION_TARGET_STEP es el paso global al terminar esta sesión}"
: "${SCHEDULE_TOTAL_STEPS:?SCHEDULE_TOTAL_STEPS es el horizonte global inmutable}"

EDGE_CORPUS_ROOT="${EDGE_CORPUS_ROOT:-$EDGE_DATA_ROOT}"
EDGE_MANIFEST_PATH="${EDGE_MANIFEST_PATH:-$EDGE_DATA_ROOT/prepared_manifest.json}"
EDGE_TOKENIZER_PATH="${EDGE_TOKENIZER_PATH:-$EDGE_DATA_ROOT/tokenizer.json}"

if [[ -e "$OUTPUT_ROOT" || -e "$PRESERVATION_PACKAGE" ]]; then
  echo "La salida y el paquete deben ser inéditos; no se reanuda ni sobrescribe aquí." >&2
  exit 2
fi
if (( SESSION_TARGET_STEP <= 0 || SCHEDULE_TOTAL_STEPS < SESSION_TARGET_STEP )); then
  echo "Los límites de sesión/scheduler son inválidos." >&2
  exit 2
fi
test -f "$EDGE_MANIFEST_PATH"
test -f "$EDGE_TOKENIZER_PATH"
compressed_shard=$(find "$EDGE_CORPUS_ROOT" -maxdepth 1 -name 'train-*.jsonl.gz' -print -quit)
plaintext_shard=$(find "$EDGE_CORPUS_ROOT" -maxdepth 1 -name 'train-*.jsonl' -print -quit)
if [[ -n "$compressed_shard" && -n "$plaintext_shard" ]]; then
  echo "El corpus mezcla shards comprimidos y descomprimidos; el formato debe ser único." >&2
  exit 2
fi
if [[ -z "$compressed_shard" && -z "$plaintext_shard" ]]; then
  echo "No se encontraron shards train-*.jsonl.gz ni train-*.jsonl." >&2
  exit 2
fi

resume_args=()
if [[ -n "${RESUME_CHECKPOINT:-}" ]]; then
  test -f "$RESUME_CHECKPOINT"
  resume_args=(--resume-checkpoint "$RESUME_CHECKPOINT")
fi

python "${SOURCE_ROOT}/engine/train_aethel_gpu.py" \
  --corpus-dir "$EDGE_CORPUS_ROOT" \
  --tokenizer "$EDGE_TOKENIZER_PATH" \
  --data-manifest "$EDGE_MANIFEST_PATH" \
  --output "$OUTPUT_ROOT" \
  --seed 17 \
  --max-steps "$SESSION_TARGET_STEP" \
  --schedule-total-steps "$SCHEDULE_TOTAL_STEPS" \
  --save-every 4000 \
  --metrics-every 256 \
  --console-every 4000 \
  --keep-snapshots 3 \
  --seq-len 1024 \
  --batch-size 1 \
  --gradient-accumulation 16 \
  --dim 512 \
  --layers 4 \
  --heads 8 \
  --kv-heads 2 \
  --experts 8 \
  --active-experts 2 \
  --memory-slots 512 \
  --replay-capacity 8192 \
  --precision bf16 \
  --router-aux-loss-weight 0.05 \
  --router-entropy-loss-weight 0.03 \
  --router-jitter-noise 0.01 \
  --allow-pytorch-fallback \
  "${resume_args[@]}"

python "${SOURCE_ROOT}/training/summarize_d1a_router_metrics.py" \
  --metrics "${OUTPUT_ROOT}/metrics_rank_0.jsonl" \
  --output "${OUTPUT_ROOT}/router_diagnostic.json" \
  --diagnostic-id AETHEL_EDGE_LONG_PHASE_V1
python "${SOURCE_ROOT}/training/validate_direct_train_pillars.py" \
  --metrics "${OUTPUT_ROOT}/metrics_rank_0.jsonl" \
  --output "$OUTPUT_ROOT" \
  --expected-steps "$SESSION_TARGET_STEP"
python "${SOURCE_ROOT}/training/package_edge_session.py" \
  --output "$OUTPUT_ROOT" \
  --package "$PRESERVATION_PACKAGE" \
  --phase-id "$PHASE_ID" \
  --session-target-step "$SESSION_TARGET_STEP" \
  --schedule-total-steps "$SCHEDULE_TOTAL_STEPS" \
  --data-manifest "$EDGE_MANIFEST_PATH"
test -f "${OUTPUT_ROOT}/edge_session_preservation_receipt.json"
test -f "${OUTPUT_ROOT}/SAVE_KAGGLE_VERSION_NOW.txt"
test -s "$PRESERVATION_PACKAGE"
sync
printf 'AETHEL_EDGE_LONG_SESSION_COMPLETE\nOUTPUT=%s\nPACKAGE=%s\nSAVE_VERSION_NOW=Use Save Version inmediatamente en Kaggle.\n' "$OUTPUT_ROOT" "$PRESERVATION_PACKAGE"
