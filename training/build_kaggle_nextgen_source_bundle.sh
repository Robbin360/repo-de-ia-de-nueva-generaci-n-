#!/usr/bin/env bash
set -euo pipefail

# Construye un paquete de código para subir manualmente como Dataset privado de Kaggle.
# No descarga datos, no publica un Dataset y no incluye pesos ni checkpoints.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-${AETHEL_KAGGLE_BUNDLE_DIR:-$ROOT/../aethel-kaggle-bundles}}"
PACKAGE_NAME="aethel-nextgen-source"
STAGING="$(mktemp -d)"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"
SOURCE_ROOT="$STAGING/$PACKAGE_NAME"
mkdir -p "$SOURCE_ROOT"

cp -R "$ROOT/engine" "$SOURCE_ROOT/engine"
cp -R "$ROOT/training" "$SOURCE_ROOT/training"
for document in TRAINING_CURRICULUM.md ARCHITECTURE_ALIGNMENT.md ARCHITECTURE_EXPERIMENTS.md; do
  if [[ -f "$ROOT/$document" ]]; then cp "$ROOT/$document" "$SOURCE_ROOT/$document"; fi
done

# El paquete de ejecución no arrastra resultados locales, bytecode ni el entrenador V3 legado.
rm -rf "$SOURCE_ROOT/engine/__pycache__" "$SOURCE_ROOT/training/__pycache__"
rm -rf "$SOURCE_ROOT/engine/artifacts" "$SOURCE_ROOT/engine/corpora"
rm -f "$SOURCE_ROOT/engine/train_aethel_v3.py"
# Las celdas de bootstrap son artefactos de operación de Kaggle, no código fuente
# del bundle; excluirlas evita que sus nombres históricos activen el guard V3.
rm -f "$SOURCE_ROOT"/training/KAGGLE_NEXTGEN_CELL_*.py
find "$SOURCE_ROOT" -type d -name '__pycache__' -prune -exec rm -rf {} +
find "$SOURCE_ROOT" -type f -name '*.pyc' -delete

if find "$SOURCE_ROOT" -iname '*v3*' -print -quit | grep -q .; then
  echo "El paquete contiene referencias V3 y se rechaza para evitar reusar el flujo heredado." >&2
  exit 2
fi

ARCHIVE="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
MANIFEST="$OUTPUT_DIR/$PACKAGE_NAME.manifest.json"
tar -C "$STAGING" -czf "$ARCHIVE" "$PACKAGE_NAME"

cat > "$MANIFEST" <<EOF
{
  "schema": "aethel-kaggle-source/v1",
  "package": "$(basename "$ARCHIVE")",
  "sha256": "$(sha256sum "$ARCHIVE" | awk '{print $1}')",
  "entrypoint": "training/bootstrap_kaggle_nextgen.sh",
  "trainer": "engine/train_aethel_gpu.py",
  "excluded": ["Aethel V3 checkpoints", "legacy V3 trainer", "local artifacts", "local corpora"],
  "requires_separate_input": "aethel-nextgen-data"
}
EOF

printf 'SOURCE_BUNDLE=%s\nMANIFEST=%s\nSHA256=%s\n' \
  "$ARCHIVE" "$MANIFEST" "$(sha256sum "$ARCHIVE" | awk '{print $1}')"
