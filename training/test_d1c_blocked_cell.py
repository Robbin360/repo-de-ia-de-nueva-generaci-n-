"""Contrato estático de la CELDA 6 D1C; no ejecuta la celda ni usa Kaggle."""
from __future__ import annotations

from pathlib import Path


def test_d1c_blocked_cell_is_numbered_and_inert() -> None:
    root = Path(__file__).resolve().parents[1]
    cell = (root / "training" / "AETHEL_D1C_ROUTER_AUX_LOSS_BLOCKED_CELL.py").read_text(encoding="utf-8")
    assert "CELDA 6 — Verificar el release D1C de peso auxiliar en modo bloqueado" in cell
    assert "D1C_EXECUTION_ENABLED = False" in cell
    assert 'EXPECTED_RELEASE = "d1c-v1-router-aux-loss-005-train-only"' in cell
    assert "D1C_CELL_PREPARED_NOT_EXECUTED" in cell
    assert "D1C_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT" in cell
    assert "subprocess" not in cell
    assert "torch" not in cell
    assert "cuda" not in cell.lower()
    assert "AETHEL_D1C_RUN_AUTHORIZED" not in cell
    assert "run_kaggle_d1c_router_aux_loss_diagnostic.sh" in cell


if __name__ == "__main__":
    test_d1c_blocked_cell_is_numbered_and_inert()
    print("D1C_BLOCKED_CELL_LOCAL_TESTS_PASSED")
