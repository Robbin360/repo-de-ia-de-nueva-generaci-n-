"""Contrato estático de D1C: no usa Dataset, GPU, PyTorch, Kaggle ni artefactos."""
from __future__ import annotations

from pathlib import Path

from summarize_d1a_router_metrics import summarize


def fixture_events() -> list[dict[str, object]]:
    return [{"step": 1, "loss": 4.0, "tokens_seen": 2048, "router_health": {"healthy": True}, "routing": [{"entropy": 0.7, "max_load": 0.3, "imbalance": 0.1, "bias": [0.01, -0.01]}]}]


def test_d1c_summary_identity_and_limits() -> None:
    report = summarize(fixture_events(), diagnostic_id="D1C")
    assert report["schema"] == "aethel-d1c-router-diagnostic/v1"
    assert report["status"] == "D1C_METRICS_SUMMARIZED"
    assert report["limits"]["checkpoint_loaded"] is False
    assert report["limits"]["holdout_content_read"] is False
    assert report["limits"]["promotion_authorized"] is False


def test_d1c_cli_choice_is_explicitly_supported() -> None:
    root = Path(__file__).resolve().parents[1]
    summarizer = (root / "training" / "summarize_d1a_router_metrics.py").read_text(encoding="utf-8")
    assert 'choices=("D1A", "D1B", "D1C", "D1D")' in summarizer


def test_d1c_launcher_is_single_change_and_strictly_blocked() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "training" / "run_kaggle_d1c_router_aux_loss_diagnostic.sh").read_text(encoding="utf-8")
    assert 'EXPECTED_RELEASE="d1c-v1-router-aux-loss-005-train-only"' in launcher
    assert 'V3_R1_RELEASE="d1c-v4-v3-r1-launcher-profile-train-only"' in launcher
    assert "AETHEL_D1C_RELEASE_PROFILE_AUTHORIZED" in launcher
    assert "AETHEL_D1C_RUN_AUTHORIZED" in launcher
    assert "AETHEL_D1C_GPU_AUTHORIZED" in launcher
    assert "AETHEL_D1C_ALLOW_PYTORCH_FALLBACK" in launcher
    assert "--router-bias-step 0.05 --router-bias-limit 0.5" in launcher
    assert "--router-aux-loss-weight 0.05" in launcher
    assert "--max-steps 768 --seq-len 1024 --batch-size 2" in launcher
    assert "--diagnostic-id D1C" in launcher
    assert 'if [[ -e "$OUTPUT_DIR" ]]' in launcher
    assert "AETHEL_RESUME_CHECKPOINT" in launcher
    assert "evaluate_nextgen.py" not in launcher
    assert "inspect_checkpoint.py" not in launcher
    assert "--resume" not in launcher


if __name__ == "__main__":
    test_d1c_summary_identity_and_limits()
    test_d1c_cli_choice_is_explicitly_supported()
    test_d1c_launcher_is_single_change_and_strictly_blocked()
    print("D1C_LOCAL_CONTRACT_TESTS_PASSED")
