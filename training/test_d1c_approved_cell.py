from __future__ import annotations

import ast
from pathlib import Path


APPROVED_CELL = Path("/home/ubuntu/aethel-private-transfer/AETHEL_CELDA_7_D1C_APROBADA.py")


def assignments(source: str) -> dict[str, object]:
    tree = ast.parse(source)
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except ValueError:
                continue
    return values


def main() -> None:
    source = APPROVED_CELL.read_text(encoding="utf-8")
    values = assignments(source)
    assert values["D1C_EXECUTION_ENABLED"] is True
    assert values["D1C_RUN_CONFIRMATION"] == "APPROVED_D1C_RUN"
    assert values["D1C_GPU_CONFIRMATION"] == "APPROVED_D1C_GPU"
    assert values["D1C_FINAL_EXECUTION_TOKEN"] == "APPROVED_FINAL_D1C_EXECUTION"
    assert values["D1C_PYTORCH_FALLBACK_CONFIRMATION"] == "APPROVED_D1C_PYTORCH_FALLBACK"
    assert 'os.environ.pop("AETHEL_RESUME_CHECKPOINT", None)' in source
    assert "D1C_DIAGNOSTIC_COMPLETE" in source
    assert "prepared_validation_holdout" not in source
    print("D1C_APPROVED_CELL_LOCAL_CONTRACTS_PASSED")


if __name__ == "__main__":
    main()
