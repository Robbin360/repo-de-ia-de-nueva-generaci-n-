"""Contrato estático de CELDA 10 autorizada; nunca ejecuta subprocess ni entrenamiento."""

from pathlib import Path


CELL = Path(__file__).with_name("AETHEL_D1C_V3_R1_AUTHORIZED_EXECUTION_CELL.py")


def main() -> None:
    text = CELL.read_text(encoding="utf-8")
    required = (
        'D1C_V3_R1_RETRY_ENABLED = True',
        'D1C_V3_R1_RETRY_RUN_CONFIRMATION = "APPROVED_D1C_V3_R1_RETRY_RUN"',
        'D1C_V3_R1_RETRY_GPU_CONFIRMATION = "APPROVED_D1C_V3_R1_RETRY_GPU"',
        'D1C_V3_R1_RETRY_FINAL_TOKEN = "APPROVED_FINAL_D1C_V3_R1_RETRY"',
        'D1C_V3_R1_RETRY_PYTORCH_FALLBACK_CONFIRMATION = "APPROVED_D1C_V3_R1_RETRY_PYTORCH_FALLBACK"',
        'D1C_V3_R1_RELEASE_PROFILE_CONFIRMATION = "APPROVED_D1C_V3_R1_RELEASE_PROFILE"',
        'EXPECTED_RELEASE = "d1c-v4-v3-r1-launcher-profile-train-only"',
        'WORK_ROOT = Path("/kaggle/working/aethel-d1c-v3-r1-retry-run")',
        'os.environ.pop("AETHEL_RESUME_CHECKPOINT", None)',
        'if completed.returncode != 0 or "D1C_DIAGNOSTIC_COMPLETE" not in completed.stdout:',
    )
    for fragment in required:
        assert fragment in text, f"Falta el contrato esperado: {fragment}"
    assert text.count("subprocess.run(") == 1
    assert text.index("data_input = resolve_data_root()") < text.index("subprocess.run(")
    print("D1C_V3_R1_AUTHORIZED_CELL_STATIC_CONTRACT_OK")


if __name__ == "__main__":
    main()
