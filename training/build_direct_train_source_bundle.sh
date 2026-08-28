#!/usr/bin/env bash
set -euo pipefail

# Construye sólo el bundle limpio para el nuevo cuaderno. No lee corpus, no
# incluye pesos ni métricas y no publica el archivo fuera del entorno local.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT/../aethel-private-transfer}"
PACKAGE_NAME="aethel-direct-train-source-v1"
RELEASE_FILE="$ROOT/training/aethel_direct_train_source_release.json"
EXPECTED_RELEASE="aethel-direct-train-source-v1"
STAGING="$(mktemp -d)"
cleanup() { rm -rf "$STAGING"; }
trap cleanup EXIT

SOURCE_RELEASE="$(sed -nE 's/^[[:space:]]*"release"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' "$RELEASE_FILE")"
if [[ "$SOURCE_RELEASE" != "$EXPECTED_RELEASE" ]]; then
  echo "El marcador del bundle directo no corresponde al release esperado." >&2
  exit 2
fi

SOURCE_ROOT="$STAGING/$PACKAGE_NAME/aethel-direct-train-source"
mkdir -p "$SOURCE_ROOT/engine" "$SOURCE_ROOT/training" "$OUTPUT_DIR"

FILES=(
  engine/aethel_model.py
  engine/aethel_nextgen.py
  engine/aethel_resume.py
  engine/router_auxiliary.py
  engine/router_health.py
  engine/train_aethel_gpu.py
  engine/prepare_bilingual_corpus.py
  engine/triton_bridge.py
  training/aethel_direct_train_source_release.json
  training/run_kaggle_direct_train_v1.sh
  training/run_kaggle_router_jitter_rerun_v1.sh
  training/package_router_jitter_rerun.py
  training/run_kaggle_router_jitter_checkpoint_eval.sh
  training/evaluate_router_jitter_checkpoint.py
  training/run_kaggle_build_edge_corpus_v1.sh
  training/run_kaggle_edge_long_session_v1.sh
  training/package_edge_session.py
  training/run_kaggle_edge_checkpoint_eval_v1.sh
  training/evaluate_edge_checkpoint.py
  training/summarize_d1a_router_metrics.py
  training/validate_direct_train_pillars.py
  training/AETHEL_META_CAPABILITY_AUDIT_2026-08-26.md
  training/AETHEL_ROUTER_CORRECTION_PROTOCOL_2026-08-26.md
  training/AETHEL_ROUTER_JITTER_PROTOCOL_2026-08-26.md
  training/AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_PROTOCOL_2026-08-26.md
  training/AETHEL_ROUTER_JITTER_CHECKPOINT_EVALUATION_PROTOCOL_2026-08-26.md
  training/AETHEL_TRAINING_RESUME_CONTRACT_V1.md
  training/AETHEL_EDGE_LONG_PHASE_BUDGET_2026-08-26.md
  training/AETHEL_EDGE_DATA_SOURCES_RESEARCH_2026-08-26.md
  training/AETHEL_EDGE_CORPUS_DESIGN_2026-08-26.md
  training/AETHEL_EDGE_CORPUS_BUILD_KAGGLE_CELLS_V1.md
  training/AETHEL_EDGE_CORPUS_BUILD_FAILURE_2026-08-26.md
  training/AETHEL_EDGE_LONG_TRAIN_KAGGLE_CELLS_V1.md
  training/AETHEL_EDGE_MOUNT_INVENTORY_CELL_V1.md
  training/AETHEL_EDGE_PHASE1_ARTIFACT_DATASET_V1.md
  training/AETHEL_EDGE_PHASE1_EVALUATION_KAGGLE_CELLS_V1.md
  training/aethel_edge_v1.manifest.json
  engine/test_router_bias_selection_contract.py
  engine/test_router_jitter_contract.py
  engine/test_compact_telemetry.py
  engine/test_hf_config_preflight.py
  training/test_evaluate_router_jitter_checkpoint.py
  training/test_package_router_jitter_rerun.py
  training/test_package_edge_session.py
  training/test_edge_long_session_contract.py
  training/test_edge_data_manifest_authorization.py
  training/test_edge_corpus_build_cells_contract.py
  training/test_edge_long_train_cells_contract.py
  engine/test_resume_contract.py
  engine/test_training_resume_e2e.py
  engine/test_prepare_bilingual_corpus.py
  engine/test_corpus_records_formats.py
  training/test_edge_mount_inventory_cell_contract.py
  training/test_evaluate_edge_checkpoint.py
  training/test_edge_phase1_evaluation_cells_contract.py
)
for relative_path in "${FILES[@]}"; do
  if [[ ! -f "$ROOT/$relative_path" ]]; then
    echo "Falta archivo obligatorio: $relative_path" >&2
    exit 2
  fi
  mkdir -p "$(dirname "$SOURCE_ROOT/$relative_path")"
  cp "$ROOT/$relative_path" "$SOURCE_ROOT/$relative_path"
done

if find "$SOURCE_ROOT" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.jsonl' -o -name '*.jsonl.gz' -o -name '*.pyc' \) -print -quit | grep -q .; then
  echo "El bundle directo contiene un artefacto prohibido." >&2
  exit 2
fi
if find "$SOURCE_ROOT" -type f -name 'aethel_d1*_source_release.json' -print -quit | grep -q .; then
  echo "El bundle directo contiene un marcador D1 histórico prohibido." >&2
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
  "schema": "aethel-direct-train-source-bundle/v1",
  "release": "$SOURCE_RELEASE",
  "tar": "$(basename "$TAR_ARCHIVE")",
  "tar_sha256": "$(sha256sum "$TAR_ARCHIVE" | awk '{print $1}')",
  "zip": "$(basename "$ZIP_ARCHIVE")",
  "zip_sha256": "$(sha256sum "$ZIP_ARCHIVE" | awk '{print $1}')",
  "included_files": ${#FILES[@]},
  "excluded": ["raw corpus and JSONL", "weights and checkpoints", "metrics JSONL", "bytecode and caches", "historical D1 release markers"],
  "execution_authorized": false
}
EOF
printf 'AETHEL_DIRECT_TRAIN_SOURCE_BUNDLE_READY\nTAR=%s\nZIP=%s\nMANIFEST=%s\n' "$TAR_ARCHIVE" "$ZIP_ARCHIVE" "$MANIFEST"
