#!/usr/bin/env bash
set -euo pipefail

# No ejecutar sin una autorización futura, explícita y separada para D1C.
if [[ "${AETHEL_D1C_RUN_AUTHORIZED:-NO}" != "YES" ]]; then
  echo "D1C bloqueado: AETHEL_D1C_RUN_AUTHORIZED debe ser YES." >&2
  exit 2
fi
if [[ "${AETHEL_D1C_GPU_AUTHORIZED:-NO}" != "YES" ]]; then
  echo "D1C bloqueado: AETHEL_D1C_GPU_AUTHORIZED debe ser YES." >&2
  exit 2
fi
if [[ -n "${AETHEL_RESUME_CHECKPOINT:-}" ]]; then
  echo "D1C bloqueado: no acepta AETHEL_RESUME_CHECKPOINT ni reanudación." >&2
  exit 2
fi
if [[ "${AETHEL_D1C_ALLOW_PYTORCH_FALLBACK:-NO}" != "YES" ]]; then
  echo "D1C bloqueado: el fallback PyTorch experimental requiere autorización separada." >&2
  exit 2
fi

SOURCE_DIR="${AETHEL_SOURCE_DIR:?Falta AETHEL_SOURCE_DIR}"
DATA_DIR="${AETHEL_DATA_DIR:?Falta AETHEL_DATA_DIR}"
OUTPUT_DIR="${AETHEL_D1C_OUTPUT_DIR:?Falta AETHEL_D1C_OUTPUT_DIR}"
EXPECTED_RELEASE="d1c-v1-router-aux-loss-005-train-only"
EXPECTED_RELEASE="${AETHEL_D1C_EXPECTED_RELEASE:-$EXPECTED_RELEASE}"
V3_R1_RELEASE="d1c-v4-v3-r1-launcher-profile-train-only"

case "$EXPECTED_RELEASE" in
  "d1c-v1-router-aux-loss-005-train-only")
    ;;
  "$V3_R1_RELEASE")
    if [[ "${AETHEL_D1C_RELEASE_PROFILE_AUTHORIZED:-NO}" != "YES" ]]; then
      echo "D1C bloqueado: el perfil V3-R1 requiere autorización separada de release." >&2
      exit 2
    fi
    ;;
  *)
    echo "D1C bloqueado: perfil de release no permitido: $EXPECTED_RELEASE" >&2
    exit 2
    ;;
esac

RELEASE_FILE="$SOURCE_DIR/training/aethel_kaggle_source_release.json"

if [[ ! -f "$RELEASE_FILE" ]] || ! grep -Fq "\"release\": \"$EXPECTED_RELEASE\"" "$RELEASE_FILE"; then
  echo "D1C bloqueado: el release fuente exacto $EXPECTED_RELEASE no está montado." >&2
  exit 2
fi
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "D1C bloqueado: AETHEL_D1C_OUTPUT_DIR ya existe; se requiere una salida nueva y vacía." >&2
  exit 2
fi
for required in \
  "$SOURCE_DIR/training/validate_aethel_train_only_mount.py" \
  "$SOURCE_DIR/training/summarize_d1a_router_metrics.py" \
  "$SOURCE_DIR/engine/test_liquid_device_alignment.py" \
  "$SOURCE_DIR/engine/train_aethel_gpu.py"; do
  [[ -f "$required" ]] || { echo "D1C bloqueado: falta $required." >&2; exit 2; }
done

mkdir -p "$OUTPUT_DIR"
python3 "$SOURCE_DIR/training/validate_aethel_train_only_mount.py" \
  --package-dir "$DATA_DIR" \
  --report "$OUTPUT_DIR/d1c_train_only_preflight.json"
python3 "$SOURCE_DIR/engine/test_liquid_device_alignment.py"
python3 "$SOURCE_DIR/engine/train_aethel_gpu.py" \
  --corpus-dir "$DATA_DIR/corpus" \
  --tokenizer "$DATA_DIR/tokenizer.json" \
  --output "$OUTPUT_DIR" \
  --max-steps 768 --seq-len 1024 --batch-size 2 --gradient-accumulation 16 \
  --dim 512 --layers 4 --heads 8 --kv-heads 2 --experts 8 --active-experts 2 \
  --memory-slots 512 --replay-capacity 8192 --router-bias-step 0.05 --router-bias-limit 0.5 \
  --router-aux-loss-weight 0.05 \
  --learning-rate 0.0003 --min-learning-rate 0.00003 --warmup-steps 500 \
  --precision fp16 --strategy single --seed 17 --save-every 768 --keep-snapshots 1 \
  --allow-pytorch-fallback
python3 "$SOURCE_DIR/training/summarize_d1a_router_metrics.py" \
  --metrics "$OUTPUT_DIR/metrics_rank_0.jsonl" \
  --output "$OUTPUT_DIR/router_diagnostic.json" \
  --diagnostic-id D1C
echo "D1C_DIAGNOSTIC_COMPLETE"
