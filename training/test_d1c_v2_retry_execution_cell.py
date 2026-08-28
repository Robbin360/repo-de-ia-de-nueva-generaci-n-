#!/usr/bin/env python3
"""Contrato estático de la CELDA 9 D1C V2: cinco puertas cerradas, sin ejecución."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CELL = ROOT / "training" / "AETHEL_D1C_V2_RETRY_EXECUTION_CELL.py"


def require(text: str, fragment: str) -> None:
    if fragment not in text:
        raise AssertionError(f"Falta el contrato requerido: {fragment}")


def main() -> None:
    text = CELL.read_text(encoding="utf-8")
    require(text, "# CELDA 9 — D1C V2")
    require(text, 'EXPECTED_RELEASE = "d1c-v2-summary-cli-fix-train-only"')
    require(text, "D1C_V2_RETRY_ENABLED = False")
    require(text, 'D1C_V2_RETRY_RUN_CONFIRMATION = "PENDING_D1C_V2_RETRY_RUN"')
    require(text, 'D1C_V2_RETRY_GPU_CONFIRMATION = "PENDING_D1C_V2_RETRY_GPU"')
    require(text, 'D1C_V2_RETRY_FINAL_TOKEN = "PENDING_FINAL_D1C_V2_RETRY"')
    require(
        text,
        'D1C_V2_RETRY_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1C_V2_RETRY_PYTORCH_FALLBACK"',
    )
    require(text, "D1C_V2_RETRY_PENDING_FINAL_AUTHORIZATION")
    require(text, "AETHEL_RESUME_CHECKPOINT")
    require(text, "os.environ.pop")
    require(text, "D1C_DIAGNOSTIC_COMPLETE")

    gate_position = text.index("if pending_values != approved_values:")
    data_position = text.index("data_input = resolve_data_root()")
    run_position = text.index("subprocess.run")
    if not gate_position < data_position < run_position:
        raise AssertionError("La puerta D1C V2 debe bloquear datos y ejecución antes de alcanzarlos.")
    print("D1C_V2_RETRY_EXECUTION_CELL_STATIC_CONTRACT_OK")


if __name__ == "__main__":
    main()
