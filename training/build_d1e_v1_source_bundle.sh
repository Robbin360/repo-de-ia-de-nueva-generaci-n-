#!/usr/bin/env bash
set -euo pipefail

# Bundle local de código D1E. No lee Dataset ni publica recursos.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT/../aethel-private-transfer}"
PACKAGE_NAME="aethel-nextgen-source-d1e-v1-router-entropy-strength-train-only"
RELEASE_FILE="$ROOT/training/aethel_d1e_source_release.json"
EXPECTED_RELEASE="d1e-v1-router-entropy-strength-train-only"
STAGING="$(mktemp -d)"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"
SOURCE_RELEASE="$(sed -nE 's/^[[:space:]]*"release"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$RELEASE_FILE")"
if [[ "$SOURCE_RELEASE" != "$EXPECTED_RELEASE" ]]; then
  echo "El marcador local no corresponde al release D1E esperado." >&2
  exit 2
fi

SOURCE_ROOT="$STAGING/$PACKAGE_NAME/aethel-nextgen-source"
mkdir -p "$SOURCE_ROOT"

while IFS= read -r -d '' source_file; do
  relative_path="${source_file#"$ROOT"/}"
  case "$relative_path" in
    engine/artifacts/*|engine/corpora/*|*/__pycache__/*) continue ;;
  esac
  case "$source_file" in
    *.py|*.sh|*.json|*.md|*.toml|*.txt)
      destination="$SOURCE_ROOT/$relative_path"
      mkdir -p "$(dirname "$destination")"
      cp "$source_file" "$destination"
      ;;
  esac
done < <(find "$ROOT/engine" "$ROOT/training" -type f -print0)

if find "$SOURCE_ROOT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.jsonl' -o -name '*.jsonl.gz' -o -name '*.pyc' \) -print -quit | grep -q .; then
  echo "El bundle D1E contiene un dato o artefacto prohibido." >&2
  exit 2
fi

if ! grep -Fq "\"release\": \"$EXPECTED_RELEASE\"" "$SOURCE_ROOT/training/aethel_d1e_source_release.json"; then
  echo "El bundle D1E no preservó el marcador exacto." >&2
  exit 2
fi

TAR_ARCHIVE="$OUTPUT_DIR/$PACKAGE_NAME.tar.gz"
ZIP_ARCHIVE="$OUTPUT_DIR/$PACKAGE_NAME.zip"
MANIFEST="$OUTPUT_DIR/$PACKAGE_NAME.manifest.json"
rm -f "$TAR_ARCHIVE" "$ZIP_ARCHIVE" "$MANIFEST"
tar -C "$STAGING" -czf "$TAR_ARCHIVE" "$PACKAGE_NAME"
(cd "$STAGING" && zip -q -r "$ZIP_ARCHIVE" "$PACKAGE_NAME")

cat > "$MANIFEST" <<EOF
{
  "schema": "aethel-d1e-v1-source-bundle/v1",
  "release": "$SOURCE_RELEASE",
  "tar": "$(basename "$TAR_ARCHIVE")",
  "tar_sha256": "$(sha256sum "$TAR_ARCHIVE" | awk '{print $1}')",
  "zip": "$(basename "$ZIP_ARCHIVE")",
  "zip_sha256": "$(sha256sum "$ZIP_ARCHIVE" | awk '{print $1}')",
  "excluded": ["raw corpus and JSONL", "weights and checkpoints", "metrics JSONL", "bytecode and caches"],
  "execution_authorized": false
}
EOF

printf 'D1E_V1_BUNDLE_LOCAL_READY\nTAR=%s\nZIP=%s\nMANIFEST=%s\n' "$TAR_ARCHIVE" "$ZIP_ARCHIVE" "$MANIFEST"
