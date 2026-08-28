from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "training" / "aethel_d1e_source_release.json"
BLOCKED_CELL = ROOT / "training" / "AETHEL_D1E_ROUTER_ENTROPY_BLOCKED_CELL.py"
EXECUTION_CELL = ROOT / "training" / "AETHEL_D1E_ROUTER_ENTROPY_EXECUTION_CELL.py"
LAUNCHER = ROOT / "training" / "run_kaggle_d1e_router_entropy_strength_diagnostic.sh"
PROTOCOL = ROOT / "training" / "AETHEL_D1E_ROUTER_ENTROPY_STRENGTH_PROTOCOL_2026-08-25.md"


def test_d1e_manifest_is_single_predefined_probe() -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert payload["diagnostic_id"] == "D1E"
    assert payload["release"] == "d1e-v1-router-entropy-strength-train-only"
    assert payload["router_entropy_loss_weight"] == 0.03
    assert payload["router_aux_loss_weight"] == 0.05
    assert payload["steps"] == 768
    assert payload["seed"] == 17
    assert payload["fresh_initialization"] is True
    for key in ("d1e_execution_authorized", "training_authorized", "notebook_edit_authorized", "gpu_authorized", "promotion_authorized"):
        assert payload[key] is False


def test_d1e_blocked_cell_has_no_open_gate() -> None:
    text = BLOCKED_CELL.read_text(encoding="utf-8")
    assert 'EXPECTED_RELEASE = "d1e-v1-router-entropy-strength-train-only"' in text
    assert "EXPECTED_ENTROPY_WEIGHT = 0.03" in text
    assert "D1E_EXECUTION_ENABLED = False" in text
    assert "D1E_PENDING_NOTEBOOK_EDIT_AND_RUN_AUTHORIZATION" in text
    assert "train" in text.lower()


def test_d1e_execution_cell_delegates_to_guarded_launcher() -> None:
    text = EXECUTION_CELL.read_text(encoding="utf-8")
    assert 'EXPECTED_RELEASE = "d1e-v1-router-entropy-strength-train-only"' in text
    assert "EXPECTED_ENTROPY_WEIGHT = 0.03" in text
    assert "EXPECTED_STEPS = 768" in text
    assert "EXPECTED_SEED = 17" in text
    assert '"--train-only"' not in text
    assert '"--no-resume"' not in text
    assert "run_kaggle_d1e_router_entropy_strength_diagnostic.sh" in text
    assert "D1E_DIAGNOSTIC_COMPLETE" in text


def test_d1e_launcher_forbids_resume_and_holdout() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert "--router-entropy-loss-weight 0.03" in text
    assert "--max-steps 768" in text
    assert "--seed 17" in text
    assert "--allow-pytorch-fallback" in text
    assert '--metrics "${OUTPUT_ROOT}/metrics_rank_0.jsonl"' in text
    assert '--output "${OUTPUT_ROOT}/router_diagnostic.json"' in text
    assert "--diagnostic-id D1E" in text
    assert "--resume" not in text
    assert "holdout" not in text.lower()


def test_d1e_protocol_predefines_failure_rule() -> None:
    text = PROTOCOL.read_text(encoding="utf-8")
    assert "D1E_ROUTER_HEALTHY_CANDIDATE" in text
    assert "D1E_ROUTER_NOT_IMPROVED" in text
    assert "No se permite barrer pesos" in text
    assert "holdout" in text.lower()
