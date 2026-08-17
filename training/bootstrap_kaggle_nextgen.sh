#!/usr/bin/env bash
set -euo pipefail

# Entrada estable invocada por el Notebook, una vez descomprimido el bundle de fuentes.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export AETHEL_SOURCE_DIR="${AETHEL_SOURCE_DIR:-$ROOT}"
exec bash "$ROOT/training/run_kaggle_nextgen.sh"
