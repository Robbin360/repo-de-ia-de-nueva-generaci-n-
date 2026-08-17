#!/usr/bin/env bash
# Construye el artefacto privado aethel-nextgen-data; no inicia entrenamiento ni usa CUDA.
set -euo pipefail

ROOT="${AETHEL_SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OUT="${AETHEL_NEXTGEN_DATA_OUTPUT:-${ROOT}/artifacts/aethel-nextgen-data}"
MANIFEST="${AETHEL_NEXTGEN_MANIFEST:-${ROOT}/training/aethel_nextgen_bilingual_pilot.manifest.json}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ "${AETHEL_ALLOW_NETWORK:-}" != "YES" ]]; then
  echo "BLOCKED: establezca AETHEL_ALLOW_NETWORK=YES para descargar fuentes reales aprobadas." >&2
  exit 2
fi
if [[ -e "${OUT}/prepared/prepared_manifest.json" || -e "${OUT}/tokenizer/aethel-bpe.json" ]]; then
  echo "BLOCKED: ${OUT} ya contiene artefactos; use una ruta vacía para preservar trazabilidad." >&2
  exit 2
fi

mkdir -p "${OUT}"
"${PYTHON_BIN}" "${ROOT}/engine/prepare_bilingual_corpus.py" --manifest "${MANIFEST}" --output "${OUT}/prepared" --allow-network --approved-data-plan
"${PYTHON_BIN}" "${ROOT}/engine/train_tokenizer.py" --corpus-dir "${OUT}/prepared" --output "${OUT}/tokenizer/aethel-bpe.json" --vocab-size 32000 --max-documents 120000
"${PYTHON_BIN}" "${ROOT}/training/prepare_bilingual_evaluations.py" --output "${OUT}/evaluation" --allow-network

cat > "${OUT}/evaluation/evaluation_plan.json" <<EOF
{
  "approved": true,
  "holdout_path": "${OUT}/prepared/validation.jsonl.gz",
  "tokenizer_path": "${OUT}/tokenizer/aethel-bpe.json",
  "seed": 17,
  "benchmark_references": {
    "mgsm_en_test": "${OUT}/evaluation/mgsm-en-test.jsonl",
    "mgsm_es_test": "${OUT}/evaluation/mgsm-es-test.jsonl",
    "belebele_eng_test": "${OUT}/evaluation/belebele-eng_Latn-test.jsonl",
    "belebele_spa_test": "${OUT}/evaluation/belebele-spa_Latn-test.jsonl"
  }
}
EOF

tar -C "${OUT}" -czf "${OUT}/aethel-nextgen-data.tar.gz" prepared tokenizer evaluation
sha256sum "${OUT}/aethel-nextgen-data.tar.gz" > "${OUT}/aethel-nextgen-data.tar.gz.sha256"
echo "READY: ${OUT}/aethel-nextgen-data.tar.gz"
