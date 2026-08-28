#!/usr/bin/env bash
set -euo pipefail

# Repetición aislada desde cero. No reanuda ni reemplaza resultados previos.
: "${SOURCE_ROOT:?SOURCE_ROOT debe apuntar al bundle limpio de código}"
: "${DATA_ROOT:?DATA_ROOT debe apuntar a aethel-nextgen-data-v1}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT debe ser una ruta inédita}"
: "${PRESERVATION_PACKAGE:?PRESERVATION_PACKAGE debe ser un TAR inexistente en /kaggle/working}"

if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "La salida inédita ya existe y no se reutiliza: $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ -e "$PRESERVATION_PACKAGE" ]]; then
  echo "El paquete de preservación ya existe y no se sobrescribe: $PRESERVATION_PACKAGE" >&2
  exit 2
fi

python "${SOURCE_ROOT}/engine/train_aethel_gpu.py" \
  --corpus-dir "${DATA_ROOT}/corpus" \
  --tokenizer "${DATA_ROOT}/tokenizer.json" \
  --output "${OUTPUT_ROOT}" \
  --seed 17 \
  --max-steps 768 \
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
  --allow-pytorch-fallback

python "${SOURCE_ROOT}/training/summarize_d1a_router_metrics.py" \
  --metrics "${OUTPUT_ROOT}/metrics_rank_0.jsonl" \
  --output "${OUTPUT_ROOT}/router_diagnostic.json" \
  --diagnostic-id DIRECT_TRAIN_ROUTER_JITTER_V1

python "${SOURCE_ROOT}/training/validate_direct_train_pillars.py" \
  --metrics "${OUTPUT_ROOT}/metrics_rank_0.jsonl" \
  --output "${OUTPUT_ROOT}" \
  --expected-steps 768

python "${SOURCE_ROOT}/training/package_router_jitter_rerun.py" \
  --output "${OUTPUT_ROOT}" \
  --package "${PRESERVATION_PACKAGE}"

test -f "${OUTPUT_ROOT}/checkpoint_preservation_receipt.json"
test -f "${OUTPUT_ROOT}/SAVE_KAGGLE_VERSION_NOW.txt"
test -s "${PRESERVATION_PACKAGE}"
sync
printf 'AETHEL_ROUTER_JITTER_RERUN_COMPLETE\nOUTPUT=%s\nPACKAGE=%s\nSAVE_VERSION_NOW=Use Save Version en Kaggle antes de cerrar, reiniciar, cambiar de sesión o solicitar otra acción.\n' "$OUTPUT_ROOT" "$PRESERVATION_PACKAGE"
