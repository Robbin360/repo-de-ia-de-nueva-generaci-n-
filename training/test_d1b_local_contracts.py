"""Pruebas locales D1B: no usan Dataset Aethel, GPU, Kaggle ni pesos."""
from __future__ import annotations

from pathlib import Path

from summarize_d1a_router_metrics import summarize


def fixture_events() -> list[dict[str, object]]:
    return [
        {
            "step": 1,
            "loss": 4.0,
            "tokens_seen": 2048,
            "config": {"router_bias_step": 0.01, "router_bias_limit": 0.5},
            "router_health": {"healthy": True},
            "routing": [{"entropy": 0.7, "max_load": 0.3, "imbalance": 0.1, "bias": [0.01, -0.01]}],
        }
    ]


def test_d1b_summary_has_explicit_identity_and_limits() -> None:
    report = summarize(fixture_events(), diagnostic_id="D1B")
    assert report["schema"] == "aethel-d1b-router-diagnostic/v1"
    assert report["status"] == "D1B_METRICS_SUMMARIZED"
    assert report["diagnostic_id"] == "D1B"
    assert report["limits"]["checkpoint_loaded"] is False
    assert report["limits"]["holdout_content_read"] is False
    assert report["limits"]["promotion_authorized"] is False


def test_summary_rejects_unrecognized_diagnostic_identity() -> None:
    try:
        summarize(fixture_events(), diagnostic_id="D2")
    except ValueError as error:
        assert "D1A, D1B o D1C" in str(error)
    else:
        raise AssertionError("Se debía rechazar una identidad diagnóstica no autorizada.")


def test_d1b_launcher_keeps_single_change_and_blockers() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    launcher = (repository_root / "training" / "run_kaggle_d1b_router_bias_diagnostic.sh").read_text(encoding="utf-8")
    assert 'EXPECTED_RELEASE="d1b-v1-router-bias-step-001-train-only"' in launcher
    assert "AETHEL_D1B_RUN_AUTHORIZED" in launcher
    assert "AETHEL_D1B_GPU_AUTHORIZED" in launcher
    assert "AETHEL_D1B_ALLOW_PYTORCH_FALLBACK" in launcher
    assert "--router-bias-step 0.01 --router-bias-limit 0.5" in launcher
    assert "--max-steps 768 --seq-len 1024 --batch-size 2" in launcher
    assert "--diagnostic-id D1B" in launcher
    assert 'if [[ -e "$OUTPUT_DIR" ]]' in launcher
    assert "AETHEL_RESUME_CHECKPOINT" in launcher
    assert "evaluate_nextgen.py" not in launcher
    assert "inspect_checkpoint.py" not in launcher
    assert "--resume" not in launcher


if __name__ == "__main__":
    test_d1b_summary_has_explicit_identity_and_limits()
    test_summary_rejects_unrecognized_diagnostic_identity()
    test_d1b_launcher_keeps_single_change_and_blockers()
    print("D1B_LOCAL_CONTRACT_TESTS_PASSED")
