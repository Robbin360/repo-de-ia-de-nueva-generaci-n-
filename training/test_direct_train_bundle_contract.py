#!/usr/bin/env python3
"""Contrato estático del bundle limpio de entrenamiento directo."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"


def main() -> None:
    marker = json.loads((TRAINING / "aethel_direct_train_source_release.json").read_text(encoding="utf-8"))
    assert marker["release"] == "aethel-direct-train-source-v1"
    assert marker["revision"] == "edge-phase1-canonical-artifact-evaluation-v1"
    assert marker["dataset_required"] == "aethel-edge-corpus-v1"
    assert marker["edge_training_inputs"] == ["aethel-direct-train-source-v1", "aethel-edge-corpus-v1"]
    assert marker["gpu_execution_authorized"] is False
    assert marker["training_authorized"] is False
    assert marker["checkpoint_loading_authorized"] is False
    assert marker["holdout_read_authorized"] is False
    assert marker["promotion_authorized"] is False
    assert (TRAINING / "run_kaggle_direct_train_v1.sh").is_file()
    assert (TRAINING / "run_kaggle_router_jitter_rerun_v1.sh").is_file()
    assert (TRAINING / "package_router_jitter_rerun.py").is_file()
    assert (TRAINING / "run_kaggle_router_jitter_checkpoint_eval.sh").is_file()
    assert (TRAINING / "evaluate_router_jitter_checkpoint.py").is_file()
    assert (TRAINING / "build_direct_train_source_bundle.sh").is_file()
    assert (ROOT / "engine" / "prepare_bilingual_corpus.py").is_file()
    assert (TRAINING / "run_kaggle_build_edge_corpus_v1.sh").is_file()
    assert (TRAINING / "run_kaggle_edge_long_session_v1.sh").is_file()
    assert (TRAINING / "package_edge_session.py").is_file()
    assert (TRAINING / "run_kaggle_edge_checkpoint_eval_v1.sh").is_file()
    assert (TRAINING / "evaluate_edge_checkpoint.py").is_file()
    runner = (TRAINING / "run_kaggle_direct_train_v1.sh").read_text(encoding="utf-8")
    assert "DIRECT_TRAIN_ROUTER_JITTER_V1" in runner
    assert "--router-jitter-noise 0.01" in runner
    assert "run_kaggle_d1e" not in runner
    summary = (TRAINING / "summarize_d1a_router_metrics.py").read_text(encoding="utf-8")
    assert "DIRECT_TRAIN_ROUTER_JITTER_V1" in summary
    assert (TRAINING / "AETHEL_ROUTER_CORRECTION_PROTOCOL_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_ROUTER_JITTER_PROTOCOL_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_PROTOCOL_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_ROUTER_JITTER_CHECKPOINT_EVALUATION_PROTOCOL_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_TRAINING_RESUME_CONTRACT_V1.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_LONG_PHASE_BUDGET_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_DATA_SOURCES_RESEARCH_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_CORPUS_DESIGN_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_CORPUS_BUILD_KAGGLE_CELLS_V1.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_CORPUS_BUILD_FAILURE_2026-08-26.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_LONG_TRAIN_KAGGLE_CELLS_V1.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_MOUNT_INVENTORY_CELL_V1.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_PHASE1_ARTIFACT_DATASET_V1.md").is_file()
    assert (TRAINING / "AETHEL_EDGE_PHASE1_EVALUATION_KAGGLE_CELLS_V1.md").is_file()
    assert (TRAINING / "aethel_edge_v1.manifest.json").is_file()
    assert (ROOT / "engine" / "test_router_bias_selection_contract.py").is_file()
    assert (ROOT / "engine" / "test_router_jitter_contract.py").is_file()
    assert (ROOT / "engine" / "test_compact_telemetry.py").is_file()
    assert (ROOT / "engine" / "test_hf_config_preflight.py").is_file()
    assert (TRAINING / "test_evaluate_router_jitter_checkpoint.py").is_file()
    assert (TRAINING / "test_package_router_jitter_rerun.py").is_file()
    assert (TRAINING / "test_package_edge_session.py").is_file()
    assert (TRAINING / "test_edge_long_session_contract.py").is_file()
    assert (TRAINING / "test_edge_data_manifest_authorization.py").is_file()
    assert (TRAINING / "test_edge_corpus_build_cells_contract.py").is_file()
    assert (TRAINING / "test_edge_long_train_cells_contract.py").is_file()
    assert (ROOT / "engine" / "aethel_resume.py").is_file()
    assert (ROOT / "engine" / "test_resume_contract.py").is_file()
    assert (ROOT / "engine" / "test_training_resume_e2e.py").is_file()
    assert (ROOT / "engine" / "test_prepare_bilingual_corpus.py").is_file()
    assert (ROOT / "engine" / "test_corpus_records_formats.py").is_file()
    assert (TRAINING / "test_edge_mount_inventory_cell_contract.py").is_file()
    assert (TRAINING / "test_evaluate_edge_checkpoint.py").is_file()
    assert (TRAINING / "test_edge_phase1_evaluation_cells_contract.py").is_file()
    print("AETHEL_DIRECT_TRAIN_BUNDLE_CONTRACT_OK")


if __name__ == "__main__":
    main()
