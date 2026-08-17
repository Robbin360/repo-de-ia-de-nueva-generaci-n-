"""Comprueba que Kaggle conserva la puerta de datos antes de toda instalación o uso CUDA."""
from pathlib import Path


def main() -> None:
    script = (Path(__file__).parent / "run_kaggle_aethel.sh").read_text(encoding="utf-8")
    gate = script.index("validate_training_readiness.py")
    install = script.index("python -m pip install")
    cuda = script.index("torch.cuda.is_available")
    assert gate < install < cuda, "La puerta de preparación debe ejecutarse antes de instalar o usar CUDA"
    assert "--require-approved" in script
    assert "AETHEL_EVALUATION_CONFIG" in script
    print("PASS: Kaggle readiness gate precedes installation and CUDA use")


if __name__ == "__main__":
    main()
