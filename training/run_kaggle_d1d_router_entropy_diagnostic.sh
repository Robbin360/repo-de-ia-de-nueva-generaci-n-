#!/usr/bin/env bash
set -euo pipefail

if [[ "${AETHEL_D1D_RUN_AUTHORIZED:-NO}" != "YES" ]]; then
  echo "D1D bloqueado: AETHEL_D1D_RUN_AUTHORIZED debe ser YES." >&2
  exit 2
fi
if [[ "${AETHEL_D1D_GPU_AUTHORIZED:-NO}" != "YES" ]]; then
  echo "D1D bloqueado: AETHEL_D1D_GPU_AUTHORIZED debe ser YES." >&2
  exit 2
fi
if [[ "${AETHEL_D1D_ALLOW_PYTORCH_FALLBACK:-NO}" != "YES" ]]; then
  echo "D1D bloqueado: el fallback PyTorch requiere autorización explícita." >&2
  exit 2
fi
if [[ -n "${AETHEL_RESUME_CHECKPOINT:-}" ]]; then
  echo "D1D bloqueado: no acepta reanudación ni AETHEL_RESUME_CHECKPOINT." >&2
  exit 2
fi

SOURCE_DIR="${AETHEL_SOURCE_DIR:?Falta AETHEL_SOURCE_DIR}"
DATA_DIR="${AETHEL_DATA_DIR:?Falta AETHEL_DATA_DIR}"
OUTPUT_DIR="${AETHEL_D1D_OUTPUT_DIR:?Falta AETHEL_D1D_OUTPUT_DIR}"
EXPECTED_RELEASE="d1d-v1-router-entropy-train-only"
RELEASE_FILE="$SOURCE_DIR/training/aethel_d1d_v5_source_release.json"

[[ -f "$RELEASE_FILE" ]] || { echo "D1D bloqueado: falta el marcador de release." >&2; exit 2; }
grep -Fq '"release": "d1d-v1-router-entropy-train-only"' "$RELEASE_FILE" || { echo "D1D bloqueado: release incorrecto." >&2; exit 2; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "D1D bloqueado: la salida ya existe; se exige una salida nueva." >&2; exit 2; }
for required in \
  "$SOURCE_DIR/training/validate_aethel_train_only_mount.py" \
  "$SOURCE_DIR/training/summarize_d1a_router_metrics.py" \
  "$SOURCE_DIR/engine/test_liquid_device_alignment.py" \
  "$SOURCE_DIR/engine/train_aethel_gpu.py"; do
  [[ -f "$required" ]] || { echo "D1D bloqueado: falta $required." >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
python3 "$SOURCE_DIR/training/validate_aethel_train_only_mount.py" \
  --package-dir "$DATA_DIR" \
  --report "$OUTPUT_DIR/d1d_train_only_preflight.json"
python3 "$SOURCE_DIR/engine/test_liquid_device_alignment.py"
python3 "$SOURCE_DIR/engine/train_aethel_gpu.py" \
  --corpus-dir "$DATA_DIR/corpus" \
  --tokenizer "$DATA_DIR/tokenizer.json" \
  --output "$OUTPUT_DIR" \
  --max-steps 768 --seq-len 1024 --batch-size 2 --gradient-accumulation 16 \
  --dim 512 --layers 4 --heads 8 --kv-heads 2 --experts 8 --active-experts 2 \
  --memory-slots 512 --replay-capacity 8192 --router-bias-step 0.05 --router-bias-limit 0.5 \
  --router-aux-loss-weight 0.05 --router-entropy-loss-weight 0.01 \
  --learning-rate 0.0003 --min-learning-rate 0.00003 --warmup-steps 500 \
  --precision fp16 --strategy single --seed 17 --save-every 768 --keep-snapshots 1 \
  --allow-pytorch-fallback
python3 "$SOURCE_DIR/training/summarize_d1a_router_metrics.py" \
  --metrics "$OUTPUT_DIR/metrics_rank_0.jsonl" \
  --output "$OUTPUT_DIR/router_diagnostic.json" \
  --diagnostic-id D1D
echo "D1D_DIAGNOSTIC_COMPLETE"
