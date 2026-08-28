"""Contrato estático del empaquetador local D1C V4; no construye ni publica archivos."""

from pathlib import Path


BUILDER = Path(__file__).with_name("build_d1c_v4_source_bundle.sh")


def main() -> None:
    text = BUILDER.read_text(encoding="utf-8")
    for fragment in (
        'EXPECTED_RELEASE="d1c-v4-v3-r1-launcher-profile-train-only"',
        'engine/artifacts/*|engine/corpora/*|*/__pycache__/*',
        "*.pt",
        "*.pth",
        "*.ckpt",
        "*.safetensors",
        "*.jsonl",
        "*.pyc",
        "execution_authorized\": false",
        "D1C_V4_BUNDLE_LOCAL_READY",
    ):
        assert fragment in text, f"Falta el contrato de bundle V4: {fragment}"
    assert "kaggle datasets" not in text.lower()
    assert "curl " not in text.lower()
    assert "wget " not in text.lower()
    print("D1C_V4_BUNDLE_BUILDER_STATIC_CONTRACT_OK")


if __name__ == "__main__":
    main()
