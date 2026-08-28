"""Prueba estática del contrato de la CELDA 11 D1D bloqueada."""

import ast
from pathlib import Path


CELL = Path(__file__).with_name("AETHEL_D1D_ROUTER_ENTROPY_BLOCKED_CELL.py")


def test_d1d_cell_has_closed_gates_and_no_training_call() -> None:
    source = CELL.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, (ast.Constant, ast.NameConstant))
    }
    assert assignments["D1D_EXECUTION_ENABLED"] is False
    assert assignments["D1D_NOTEBOOK_EDIT_CONFIRMATION"].startswith("PENDING_")
    assert assignments["D1D_RUN_CONFIRMATION"].startswith("PENDING_")
    assert assignments["D1D_GPU_CONFIRMATION"].startswith("PENDING_")
    assert assignments["D1D_FINAL_TOKEN"].startswith("PENDING_")
    assert assignments["D1D_PYTORCH_FALLBACK_CONFIRMATION"].startswith("PENDING_")
    assert "train_aethel_gpu" not in source
    assert "torch.load" not in source
    assert "subprocess" not in source


def test_d1d_release_and_protocol_are_pinned() -> None:
    source = CELL.read_text(encoding="utf-8")
    assert 'EXPECTED_RELEASE = "d1d-v1-router-entropy-train-only"' in source
    assert "AETHEL_D1D_ROUTER_ENTROPY_PROTOCOL_2026-08-25.md" in source
