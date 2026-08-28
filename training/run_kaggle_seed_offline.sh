#!/usr/bin/env bash
set -euo pipefail

# Ruta de Aethel Seed para un paquete de conocimiento ya congelado. No descarga
# corpus, no reconstruye el tokenizador y no relaja el contrato Triton por defecto.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${AETHEL_SOURCE_DIR:=$(cd "$SCRIPT_DIR/.." && pwd)}"
: "${AETHEL_DATA_DIR:=/kaggle/input/aethel-knowledge-corpus-v1-package}"
: "${AETHEL_OUTPUT_DIR:=/kaggle/working/aethel-runs/aethel-seed}"
: "${AETHEL_RUN_AUTHORIZED:=}"
: "${AETHEL_LAB_FALLBACK_AUTHORIZED:=NO}"
: "${AETHEL_MAX_STEPS:=4992}"
: "${AETHEL_DIM:=512}"
: "${AETHEL_LAYERS:=4}"
: "${AETHEL_HEADS:=8}"
: "${AETHEL_KV_HEADS:=2}"
: "${AETHEL_EXPERTS:=8}"
: "${AETHEL_SEQ_LEN:=1024}"
: "${AETHEL_BATCH_SIZE:=2}"
: "${AETHEL_GRAD_ACCUM:=16}"
: "${AETHEL_SAVE_EVERY:=192}"
: "${AETHEL_KEEP_SNAPSHOTS:=3}"

if [[ "$AETHEL_RUN_AUTHORIZED" != "YES" ]]; then
  echo "BLOCKED: defina AETHEL_RUN_AUTHORIZED=YES sólo tras autorizar una corrida GPU real." >&2
  exit 3
fi

for required in \
  "$AETHEL_SOURCE_DIR/engine/train_aethel_gpu.py" \
  "$AETHEL_SOURCE_DIR/engine/evaluate_nextgen.py" \
  "$AETHEL_SOURCE_DIR/engine/test_triton_gpu.py" \
  "$AETHEL_SOURCE_DIR/engine/test_liquid_device_alignment.py" \
  "$AETHEL_SOURCE_DIR/training/validate_aethel_knowledge_package.py" \
  "$AETHEL_SOURCE_DIR/training/inspect_checkpoint.py"; do
  if [[ ! -f "$required" ]]; then
    echo "BLOCKED: falta un componente Seed verificable: $required" >&2
    exit 4
  fi
done

if [[ ! -d "$AETHEL_DATA_DIR" ]]; then
  echo "BLOCKED: el paquete congelado no está montado en AETHEL_DATA_DIR=$AETHEL_DATA_DIR" >&2
  exit 5
fi

mkdir -p "$AETHEL_OUTPUT_DIR"

# Esta validación lee sólo el Dataset montado, verifica hashes y certifica que
# la procedencia del tokenizador se limita a train. No ejecuta peticiones de red.
python "$AETHEL_SOURCE_DIR/training/validate_aethel_knowledge_package.py" \
  --package-dir "$AETHEL_DATA_DIR" \
  --report "$AETHEL_OUTPUT_DIR/package_preflight.json"

test -f "$AETHEL_DATA_DIR/tokenizer.json"

resolve_holdout() {
  local language="$1"
  local compressed="$AETHEL_DATA_DIR/corpus/holdout-${language}-00000.jsonl.gz"
  local plaintext="$AETHEL_DATA_DIR/corpus/holdout-${language}-00000.jsonl"
  if [[ -f "$compressed" && -f "$plaintext" ]]; then
    echo "BLOCKED: el holdout ${language} mezcla .jsonl.gz y .jsonl; el montaje debe usar un formato único verificable." >&2
    exit 7
  fi
  if [[ -f "$compressed" ]]; then
    printf '%s\n' "$compressed"
    return
  fi
  if [[ -f "$plaintext" ]]; then
    printf '%s\n' "$plaintext"
    return
  fi
  echo "BLOCKED: falta el holdout ${language} en formato .jsonl.gz o .jsonl." >&2
  exit 7
}

HOLDOUT_EN="$(resolve_holdout en)"
HOLDOUT_ES="$(resolve_holdout es)"

python - <<'PY'
import torch
assert torch.cuda.is_available(), "Activa Accelerator=GPU antes de iniciar Aethel Seed."
print({"gpu": torch.cuda.get_device_name(0), "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)})
PY

# El proyecto no puede afirmar soporte GPU completo hasta validar kernels Triton
# de prefill causal y dispatch/combina MoE. El fallback existe sólo para un E0 de
# laboratorio explícitamente autorizado; nunca habilita promoción comercial.
if [[ "$AETHEL_LAB_FALLBACK_AUTHORIZED" != "YES" ]]; then
  echo "BLOCKED: el contrato Triton estricto sigue activo. Aethel Seed no entrena en CUDA hasta validar prefill causal y dispatch/combina MoE completos. Para un E0 experimental con operadores PyTorch, se requiere AETHEL_LAB_FALLBACK_AUTHORIZED=YES y sus resultados no son promocionables." >&2
  exit 6
fi

cd "$AETHEL_SOURCE_DIR/engine"
python test_triton_gpu.py
python test_liquid_device_alignment.py

cd "$AETHEL_SOURCE_DIR"
python - <<'PY'
import importlib
import sys

required = ("torch", "tokenizers", "triton")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(
        "BLOCKED: faltan dependencias ya requeridas por E0: " + ", ".join(missing) +
        ". El lanzador no instala paquetes ni realiza peticiones de red."
    )
import torch
print({"offline_dependency_check": "passed", "torch": torch.__version__, "cuda": torch.version.cuda})
PY

python engine/train_aethel_gpu.py \
  --corpus-dir "$AETHEL_DATA_DIR/corpus" \
  --tokenizer "$AETHEL_DATA_DIR/tokenizer.json" \
  --output "$AETHEL_OUTPUT_DIR" \
  --max-steps "$AETHEL_MAX_STEPS" \
  --dim "$AETHEL_DIM" \
  --layers "$AETHEL_LAYERS" \
  --heads "$AETHEL_HEADS" \
  --kv-heads "$AETHEL_KV_HEADS" \
  --experts "$AETHEL_EXPERTS" \
  --active-experts 2 \
  --seq-len "$AETHEL_SEQ_LEN" \
  --batch-size "$AETHEL_BATCH_SIZE" \
  --gradient-accumulation "$AETHEL_GRAD_ACCUM" \
  --save-every "$AETHEL_SAVE_EVERY" \
  --keep-snapshots "$AETHEL_KEEP_SNAPSHOTS" \
  --allow-pytorch-fallback \
  --resume

test -f "$AETHEL_OUTPUT_DIR/latest.pt"
test -f "$AETHEL_OUTPUT_DIR/recovery_receipt.json"
test -f "$AETHEL_OUTPUT_DIR/metrics_rank_0.jsonl"
python "$AETHEL_SOURCE_DIR/training/inspect_checkpoint.py" \
  "$AETHEL_OUTPUT_DIR/latest.pt" \
  --require-reproducible \
  --output "$AETHEL_OUTPUT_DIR/checkpoint_inspection.json"

python "$AETHEL_SOURCE_DIR/engine/evaluate_nextgen.py" \
  --checkpoint "$AETHEL_OUTPUT_DIR/latest.pt" \
  --corpus "$HOLDOUT_EN" \
  --tokenizer "$AETHEL_DATA_DIR/tokenizer.json" \
  --seq-len "$AETHEL_SEQ_LEN" \
  > "$AETHEL_OUTPUT_DIR/evaluation_holdout_en.json"
python "$AETHEL_SOURCE_DIR/engine/evaluate_nextgen.py" \
  --checkpoint "$AETHEL_OUTPUT_DIR/latest.pt" \
  --corpus "$HOLDOUT_ES" \
  --tokenizer "$AETHEL_DATA_DIR/tokenizer.json" \
  --seq-len "$AETHEL_SEQ_LEN" \
  > "$AETHEL_OUTPUT_DIR/evaluation_holdout_es.json"

printf 'PERSISTED_AS=notebook-output\nOUTPUT_DIR=%s\nCHECKPOINT=%s\nPRELIGHT=%s\nMETRICS=%s\nEVALUATION_EN=%s\nEVALUATION_ES=%s\n' \
  "$AETHEL_OUTPUT_DIR" \
  "$AETHEL_OUTPUT_DIR/latest.pt" \
  "$AETHEL_OUTPUT_DIR/package_preflight.json" \
  "$AETHEL_OUTPUT_DIR/metrics_rank_0.jsonl" \
  "$AETHEL_OUTPUT_DIR/evaluation_holdout_en.json" \
  "$AETHEL_OUTPUT_DIR/evaluation_holdout_es.json" \
  | tee "$AETHEL_OUTPUT_DIR/persistence_receipt.txt"
