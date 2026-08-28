#!/usr/bin/env bash
set -euo pipefail

# Corrida única correctiva del router con jitter de selección. Inicia desde
# cero, lee sólo shards train del Dataset v1 y no evalúa holdout ni reanuda pesos.
: "${SOURCE_ROOT:?SOURCE_ROOT debe apuntar al bundle limpio de código}"
: "${DATA_ROOT:?DATA_ROOT debe apuntar a aethel-nextgen-data-v1}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT debe ser una ruta inédita}"

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

echo "AETHEL_DIRECT_TRAINING_COMPLETE"
