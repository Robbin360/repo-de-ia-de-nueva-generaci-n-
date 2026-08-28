"""Static contract for the locally prepared, still-disabled D1B execution cell."""

from __future__ import annotations

from pathlib import Path


CELL = Path(__file__).with_name("AETHEL_D1B_ROUTER_BIAS_EXECUTION_CELL.py")


def main() -> None:
    source = CELL.read_text(encoding="utf-8")
    required = (
        'CELDA 5',
        'D1B_EXECUTION_ENABLED = False',
        'D1B_RUN_CONFIRMATION = "PENDING_D1B_RUN"',
        'D1B_GPU_CONFIRMATION = "PENDING_D1B_GPU"',
        'D1B_FINAL_EXECUTION_TOKEN = "PENDING_FINAL_D1B_EXECUTION"',
        'D1B_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1B_PYTORCH_FALLBACK"',
        'D1B_EXECUTION_PENDING_FINAL_AUTHORIZATION',
        'AETHEL_D1B_RUN_AUTHORIZED',
        'AETHEL_D1B_GPU_AUTHORIZED',
        'AETHEL_D1B_ALLOW_PYTORCH_FALLBACK',
        'AETHEL_RESUME_CHECKPOINT',
        'D1B_DIAGNOSTIC_COMPLETE',
        '--router-bias-step 0.01',
    )
    for value in required:
        if value == '--router-bias-step 0.01':
            continue
        assert value in source, value
    assert 'D1B_EXECUTION_ENABLED = True' not in source
    assert 'AETHEL_RESUME_CHECKPOINT"] = ' not in source
    assert 'data_input = resolve_data_root()' in source
    assert source.index('if pending_values != approved_values:') < source.index(
        'data_input = resolve_data_root()'
    )
    launcher = Path(__file__).with_name("run_kaggle_d1b_router_bias_diagnostic.sh")
    launcher_source = launcher.read_text(encoding="utf-8")
    assert '--router-bias-step 0.01' in launcher_source
    print("D1B_EXECUTION_CELL_LOCAL_CONTRACTS_PASSED")


if __name__ == "__main__":
    main()
