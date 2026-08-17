#!/usr/bin/env bash
set -euo pipefail

# Ruta integral para Kaggle: construye datos reales aprobados en /kaggle/working,
# prepara evaluación retenida y sólo después delega en el lanzador GPU protegido.
: "${AETHEL_SOURCE_DIR:=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${AETHEL_DATA_DIR:=/kaggle/working/aethel-nextgen-data}"
: "${AETHEL_RUN_AUTHORIZED:=}"
: "${AETHEL_BUILD_DATA_IN_KAGGLE:=}"
: "${AETHEL_PERSISTENCE_MODE:=notebook-output}"

if [[ "$AETHEL_RUN_AUTHORIZED" != "YES" ]]; then
  echo "BLOCKED: define AETHEL_RUN_AUTHORIZED=YES únicamente en una versión Save & Run All autorizada." >&2
  exit 3
fi
if [[ "$AETHEL_BUILD_DATA_IN_KAGGLE" != "YES" ]]; then
  echo "BLOCKED: AETHEL_BUILD_DATA_IN_KAGGLE=YES confirma la construcción de fuentes reales en Kaggle." >&2
  exit 3
fi
if [[ "$AETHEL_SOURCE_DIR:$AETHEL_DATA_DIR" == *"aethel-v3"* || "$AETHEL_SOURCE_DIR:$AETHEL_DATA_DIR" == *"aethel_v3"* ]]; then
  echo "BLOCKED: la ruta integral de NextGen rechaza cualquier ruta de Aethel V3." >&2
  exit 4
fi
if [[ -e "$AETHEL_DATA_DIR/prepared/prepared_manifest.json" || -e "$AETHEL_DATA_DIR/tokenizer/aethel-bpe.json" ]]; then
  echo "BLOCKED: $AETHEL_DATA_DIR ya contiene datos; use una ruta vacía para conservar trazabilidad." >&2
  exit 5
fi

cd "$AETHEL_SOURCE_DIR"
# Sólo dependencias CPU de preparación; CUDA aún no se consulta ni reserva.
python -m pip install --quiet -r training/requirements.txt
AETHEL_SOURCE_ROOT="$AETHEL_SOURCE_DIR" \
  AETHEL_NEXTGEN_DATA_OUTPUT="$AETHEL_DATA_DIR" \
  AETHEL_ALLOW_NETWORK=YES \
  bash training/build_aethel_nextgen_data.sh

export AETHEL_DATA_DIR AETHEL_RUN_AUTHORIZED AETHEL_PERSISTENCE_MODE
exec bash training/run_kaggle_nextgen.sh
