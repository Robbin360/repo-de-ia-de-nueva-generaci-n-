"""Prueba de la misma ruta Kaggle sin usar red ni credenciales reales."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

from export_artifacts import export_kaggle_dataset


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    runner = project_root / "training" / "run_kaggle_aethel.sh"
    blocked = subprocess.run(["bash", str(runner)], capture_output=True, text=True, cwd=project_root)
    assert blocked.returncode == 2, (blocked.returncode, blocked.stderr)
    assert "AETHEL_KAGGLE_DATASET" in blocked.stderr

    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        source = root_path / "run"
        source.mkdir()
        (source / "latest.pt").write_bytes(b"checkpoint-for-kaggle-simulation")
        (source / "metrics.jsonl").write_text('{"step":2,"loss":0.5}\n', encoding="utf-8")
        fake_bin = root_path / "bin"
        fake_bin.mkdir()
        invocation = root_path / "kaggle-invocation.txt"
        fake_cli = fake_bin / "kaggle"
        fake_cli.write_text(f"#!/usr/bin/env bash\nprintf '%s\\n' \"$*\" > {invocation}\n", encoding="utf-8")
        fake_cli.chmod(0o755)

        staging = root_path / f"aethel-kaggle-test-{uuid.uuid4().hex}"
        old_path = os.environ.get("PATH", "")
        old_simulation = os.environ.get("AETHEL_KAGGLE_SIMULATION")
        os.environ["PATH"] = f"{fake_bin}{os.pathsep}{old_path}"
        os.environ["AETHEL_KAGGLE_SIMULATION"] = "1"
        try:
            result = export_kaggle_dataset(source, staging, "test-owner/aethel-artifacts-private")
        finally:
            os.environ["PATH"] = old_path
            if old_simulation is None:
                os.environ.pop("AETHEL_KAGGLE_SIMULATION", None)
            else:
                os.environ["AETHEL_KAGGLE_SIMULATION"] = old_simulation
            shutil.rmtree(staging, ignore_errors=True)

        assert Path(result["archive"]).is_file() is False, "El staging se limpia tras verificar el empaquetado simulado"
        assert "datasets version" in invocation.read_text(encoding="utf-8")
        assert result["dataset"] == "test-owner/aethel-artifacts-private"
        assert len(result["archive_sha256"]) == 64
        print('{"kaggle_export_simulation_verified":true,"runner_blocks_without_destination":true}')


if __name__ == "__main__":
    main()
