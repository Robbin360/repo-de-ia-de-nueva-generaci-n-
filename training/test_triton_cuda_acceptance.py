"""La herramienta CUDA debe bloquearse explícitamente y emitir informe en un host sin GPU."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "training" / "run_triton_cuda_acceptance.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        output = Path(temporary) / "acceptance.json"
        completed = subprocess.run(
            ["python3", str(RUNNER), "--output", str(output)],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        report = json.loads(output.read_text(encoding="utf-8"))
    assert completed.returncode == 2
    assert report["status"] == "NOT_RUN"
    assert report["cuda_available"] is False
    assert "no se habilitó" in report["blocker"]
    print("PASS: ejecutor CUDA emite bloqueo verificable sin hardware")


if __name__ == "__main__":
    main()
