#!/usr/bin/env python3
"""Contrato estático de la CELDA 9 D1C V3: sólo verificación bloqueada."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "training" / "AETHEL_D1C_V3_RETRY_BLOCKED_CELL.py"


def require(text: str, fragment: str) -> None:
    if fragment not in text:
        raise AssertionError(f"Falta el contrato requerido: {fragment}")


def main() -> None:
    text = CELL.read_text(encoding="utf-8")
    require(text, "# CELDA 9 — D1C V3")
    require(text, 'SOURCE_DATASET = Path("/kaggle/input/datasets/felixtremigual/aethel-nextgen-source-e0-v1")')
    require(text, 'EXPECTED_RELEASE = "d1c-v3-retry-cell-train-only"')
    require(text, 'payload.get("d1c_v3_execution_authorized") is False')
    require(text, 'payload.get("training_authorized") is False')
    require(text, "D1C_V3_RETRY_ENABLED = False")
    require(text, 'D1C_V3_RETRY_RUN_CONFIRMATION = "PENDING_D1C_V3_RETRY_RUN"')
    require(text, 'D1C_V3_RETRY_GPU_CONFIRMATION = "PENDING_D1C_V3_RETRY_GPU"')
    require(text, 'D1C_V3_RETRY_FINAL_TOKEN = "PENDING_FINAL_D1C_V3_RETRY"')
    require(
        text,
        'D1C_V3_RETRY_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1C_V3_RETRY_PYTORCH_FALLBACK"',
    )
    require(text, "D1C_V3_CELL_PREPARED_NOT_EXECUTED")
    require(text, "D1C_V3_RETRY_PENDING_FINAL_AUTHORIZATION")
    require(text, "resolve_d1c_v3_source(SOURCE_DATASET)")
    for forbidden in ("resolve_data_root", "subprocess.run", "shutil.copytree", "AETHEL_RESUME_CHECKPOINT"):
        if forbidden in text:
            raise AssertionError(f"La CELDA 9 V3 bloqueada no debe contener: {forbidden}")
    print("D1C_V3_BLOCKED_CELL_STATIC_CONTRACT_OK")


if __name__ == "__main__":
    main()
