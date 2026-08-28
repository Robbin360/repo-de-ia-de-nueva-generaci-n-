"""Contratos estáticos del diagnóstico D1D; no ejecuta GPU, datos ni entrenamiento."""
from __future__ import annotations

from pathlib import Path

from summarize_d1a_router_metrics import summarize


def fixture_events() -> list[dict[str, object]]:
    return [{"step": 1, "loss": 4.0, "tokens_seen": 2048, "router_health": {"healthy": True}, "routing": [{"entropy": 0.7, "max_load": 0.3, "imbalance": 0.1, "bias": [0.01, -0.01]}]}]


def test_d1d_summary_identity() -> None:
    report = summarize(fixture_events(), diagnostic_id="D1D")
    assert report["schema"] == "aethel-d1d-router-diagnostic/v1"
    assert report["status"] == "D1D_METRICS_SUMMARIZED"
    assert report["limits"]["checkpoint_loaded"] is False
    assert report["limits"]["holdout_content_read"] is False


def test_d1d_launcher_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    launcher = (root / "training" / "run_kaggle_d1d_router_entropy_diagnostic.sh").read_text(encoding="utf-8")
    assert 'EXPECTED_RELEASE="d1d-v1-router-entropy-train-only"' in launcher
    assert "AETHEL_D1D_RUN_AUTHORIZED" in launcher
    assert "AETHEL_D1D_GPU_AUTHORIZED" in launcher
    assert "AETHEL_D1D_ALLOW_PYTORCH_FALLBACK" in launcher
    assert "AETHEL_RESUME_CHECKPOINT" in launcher
    assert "--router-aux-loss-weight 0.05 --router-entropy-loss-weight 0.01" in launcher
    assert "--max-steps 768 --seq-len 1024 --batch-size 2" in launcher
    assert "--diagnostic-id D1D" in launcher
    assert "--resume" not in launcher
    assert "evaluate_nextgen.py" not in launcher


def test_d1d_execution_cell_requires_fresh_output() -> None:
    root = Path(__file__).resolve().parents[1]
    cell = (root / "training" / "AETHEL_D1D_ROUTER_ENTROPY_EXECUTION_CELL.py").read_text(encoding="utf-8")
    assert 'D1D_EXECUTION_ENABLED = True' in cell
    assert "WORK_ROOT.exists()" in cell
    assert "shutil.copytree" in cell
    assert "resolve_data" in cell
    assert "AETHEL_RESUME_CHECKPOINT" in cell


if __name__ == "__main__":
    test_d1d_summary_identity()
    test_d1d_launcher_contract()
    test_d1d_execution_cell_requires_fresh_output()
    print("D1D_EXECUTION_CONTRACT_TESTS_PASSED")
