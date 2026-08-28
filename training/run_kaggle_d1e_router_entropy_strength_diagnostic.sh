#!/usr/bin/env bash
set -euo pipefail

# D1E: una única sonda de fuerza de entropía; no reanuda ni evalúa holdout.
: "${SOURCE_ROOT:?SOURCE_ROOT debe apuntar a la raíz del bundle D1E}"
: "${DATA_ROOT:?DATA_ROOT debe apuntar al Dataset privado de datos}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT debe ser un directorio inédito}"

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
  --allow-pytorch-fallback

python "${SOURCE_ROOT}/training/summarize_d1a_router_metrics.py" \
  --metrics "${OUTPUT_ROOT}/metrics_rank_0.jsonl" \
  --output "${OUTPUT_ROOT}/router_diagnostic.json" \
  --diagnostic-id D1E
python "${SOURCE_ROOT}/training/validate_direct_train_pillars.py" \
  --metrics "${OUTPUT_ROOT}/metrics_rank_0.jsonl" \
  --output "${OUTPUT_ROOT}" \
  --expected-steps 768
