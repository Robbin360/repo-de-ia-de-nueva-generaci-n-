"""Pruebas del empaquetado y de la puerta de autorización del flujo limpio NextGen."""
from __future__ import annotations

import os
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"


def main() -> None:
    runner = TRAINING / "run_kaggle_nextgen.sh"
    result = subprocess.run(["bash", str(runner)], text=True, capture_output=True, env={**os.environ, "AETHEL_RUN_AUTHORIZED": ""})
    assert result.returncode == 3
    assert "AETHEL_RUN_AUTHORIZED=YES" in result.stderr

    legacy = subprocess.run(
        ["bash", str(runner)],
        text=True,
        capture_output=True,
        env={**os.environ, "AETHEL_RUN_AUTHORIZED": "YES", "AETHEL_SOURCE_DIR": "/tmp/aethel-v3-checkpoints"},
    )
    assert legacy.returncode == 4
    assert "no acepta rutas o checkpoints de Aethel V3" in legacy.stderr

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory)
        packaged = subprocess.run(["bash", str(TRAINING / "build_kaggle_nextgen_source_bundle.sh"), str(output)], text=True, capture_output=True, check=True)
        archive = output / "aethel-nextgen-source.tar.gz"
        assert archive.is_file(), packaged.stdout
        with tarfile.open(archive, "r:gz") as bundle:
            names = bundle.getnames()
        assert "aethel-nextgen-source/engine/train_aethel_gpu.py" in names
        assert "aethel-nextgen-source/training/bootstrap_kaggle_nextgen.sh" in names
        assert not any("train_aethel_v3.py" in name or "aethel-v3" in name for name in names)
    print("PASS: NextGen Kaggle scripts reject V3 and package clean sources")


if __name__ == "__main__":
    main()
