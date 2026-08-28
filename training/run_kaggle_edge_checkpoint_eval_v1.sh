#!/usr/bin/env bash
set -euo pipefail

# Evaluación aislada: no entrena, no reanuda ni copia/modifica el checkpoint.
for variable in SOURCE_ROOT EDGE_ARTIFACT_ROOT EDGE_DATA_ROOT EVALUATION_OUTPUT; do
  if [[ -z "${!variable:-}" ]]; then
    echo "Falta $variable" >&2
    exit 2
  fi
done
CHECKPOINT="$EDGE_ARTIFACT_ROOT/latest.pt"
TOKENIZER="$EDGE_ARTIFACT_ROOT/tokenizer.json"
MANIFEST="$EDGE_DATA_ROOT/prepared_manifest.json"
VALIDATION=""
for candidate in "$EDGE_DATA_ROOT/validation.jsonl" "$EDGE_DATA_ROOT/validation.jsonl.gz"; do
  if [[ -f "$candidate" ]]; then VALIDATION="$candidate"; break; fi
done
[[ -f "$CHECKPOINT" && -f "$TOKENIZER" && -f "$MANIFEST" && -n "$VALIDATION" ]] || { echo "Faltan artefactos Edge requeridos." >&2; exit 2; }
[[ ! -e "$EVALUATION_OUTPUT" ]] || { echo "La salida de evaluación ya existe y no se reutiliza: $EVALUATION_OUTPUT" >&2; exit 2; }

PYTHONDONTWRITEBYTECODE=1 python "$SOURCE_ROOT/training/evaluate_edge_checkpoint.py" \
  --checkpoint "$CHECKPOINT" \
  --tokenizer "$TOKENIZER" \
  --data-manifest "$MANIFEST" \
  --validation "$VALIDATION" \
  --output "$EVALUATION_OUTPUT" \
  --device cuda \
  --seq-len 1024 \
  --max-segments-per-language 256 \
  --max-new-tokens 32

test -f "$EVALUATION_OUTPUT/edge_evaluation_receipt.json"
sync
echo "AETHEL_EDGE_EVALUATION_COMPLETE"
echo "RECEIPT=$EVALUATION_OUTPUT/edge_evaluation_receipt.json"
