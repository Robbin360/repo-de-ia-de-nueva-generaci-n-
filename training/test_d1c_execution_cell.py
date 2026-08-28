from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "training" / "AETHEL_D1C_ROUTER_AUX_LOSS_EXECUTION_CELL.py"


def main() -> None:
    source = CELL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert "CELDA 7 — D1C" in source
    assert "D1C_EXECUTION_ENABLED = False" in source
    assert 'D1C_RUN_CONFIRMATION = "PENDING_D1C_RUN"' in source
    assert 'D1C_GPU_CONFIRMATION = "PENDING_D1C_GPU"' in source
    assert 'D1C_FINAL_EXECUTION_TOKEN = "PENDING_FINAL_D1C_EXECUTION"' in source
    assert 'D1C_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1C_PYTORCH_FALLBACK"' in source
    assert "D1C_EXECUTION_PENDING_FINAL_AUTHORIZATION" in source
    assert "D1C_DIAGNOSTIC_COMPLETE" in source
    assert source.index("source_input = resolve_d1c_source()") < source.index(
        "if pending_values != approved_values:"
    )
    assert source.index("data_input = resolve_data_root()") > source.index(
        "if pending_values != approved_values:"
    )
    assert source.index("shutil.copytree(source_input, SOURCE_WORK)") > source.index(
        "if pending_values != approved_values:"
    )
    assert source.index("subprocess.run(") > source.index("if pending_values != approved_values:")
    assert any(isinstance(node, ast.If) for node in ast.walk(tree))
    print("D1C_EXECUTION_CELL_LOCAL_CONTRACTS_PASSED")


if __name__ == "__main__":
    main()
