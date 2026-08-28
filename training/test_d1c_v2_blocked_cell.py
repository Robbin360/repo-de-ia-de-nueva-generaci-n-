"""Contrato estático de la CELDA 8 D1C V2: no usa Dataset, GPU ni ejecución."""
from __future__ import annotations

import ast
from pathlib import Path


def test_d1c_v2_blocked_cell_is_numbered_and_has_no_execution_path() -> None:
    root = Path(__file__).resolve().parents[1]
    cell_path = root / "training" / "AETHEL_D1C_V2_SUMMARY_FIX_BLOCKED_CELL.py"
    source = cell_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    assert source.startswith("# CELDA 8 —")
    assert 'REQUIRED_RELEASE = "d1c-v2-summary-cli-fix-train-only"' in source
    assert "D1C_V2_CELL_PREPARED_NOT_EXECUTED" in source
    assert "D1C_V2_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT" in source
    assert "subprocess" not in source
    assert "torch" not in source
    assert "/kaggle/input/aethel-nextgen-data-v1" not in source
    assert "OUTPUT_DIR" not in source

    imported = {
        alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert imported == {"json"}


if __name__ == "__main__":
    test_d1c_v2_blocked_cell_is_numbered_and_has_no_execution_path()
    print("D1C_V2_BLOCKED_CELL_LOCAL_CONTRACTS_PASSED")
