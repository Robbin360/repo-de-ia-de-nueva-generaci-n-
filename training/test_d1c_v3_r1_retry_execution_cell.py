"""Contrato estático de CELDA 10 V3-R1; no usa Dataset, GPU ni subprocess."""

from pathlib import Path


CELL = Path(__file__).with_name("AETHEL_D1C_V3_R1_RETRY_EXECUTION_CELL.py")


def main() -> None:
    text = CELL.read_text(encoding="utf-8")
    for fragment in (
        "CELDA 10 — D1C V3-R1",
        'EXPECTED_RELEASE = "d1c-v4-v3-r1-launcher-profile-train-only"',
        "D1C_V3_R1_RETRY_ENABLED = False",
        'D1C_V3_R1_RETRY_RUN_CONFIRMATION = "PENDING_D1C_V3_R1_RETRY_RUN"',
        'D1C_V3_R1_RETRY_GPU_CONFIRMATION = "PENDING_D1C_V3_R1_RETRY_GPU"',
        'D1C_V3_R1_RETRY_FINAL_TOKEN = "PENDING_FINAL_D1C_V3_R1_RETRY"',
        'D1C_V3_R1_RETRY_PYTORCH_FALLBACK_CONFIRMATION = "PENDING_D1C_V3_R1_RETRY_PYTORCH_FALLBACK"',
        'D1C_V3_R1_RELEASE_PROFILE_CONFIRMATION = "PENDING_D1C_V3_R1_RELEASE_PROFILE"',
        "D1C_V3_R1_CELL_PREPARED_NOT_EXECUTED",
        "D1C_V3_R1_RETRY_PENDING_FINAL_AUTHORIZATION",
        "training_root = marker.parent",
        "source_root = training_root.parent",
        "AETHEL_D1C_EXPECTED_RELEASE",
        "AETHEL_D1C_RELEASE_PROFILE_AUTHORIZED",
        'os.environ.pop("AETHEL_RESUME_CHECKPOINT", None)',
        "No se borra ni se reutiliza una salida previa.",
    ):
        assert fragment in text, f"Falta el contrato: {fragment}"

    gate_index = text.index("if pending_values != approved_values:")
    data_index = text.index("data_input = resolve_data_root()")
    copy_index = text.index("shutil.copytree(source_input, SOURCE_WORK)")
    subprocess_index = text.index("subprocess.run")
    assert gate_index < data_index < copy_index < subprocess_index
    assert "evaluate_nextgen.py" not in text
    assert "inspect_checkpoint.py" not in text
    assert "--resume" not in text
    print("D1C_V3_R1_RETRY_EXECUTION_CELL_STATIC_CONTRACT_OK")


if __name__ == "__main__":
    main()
