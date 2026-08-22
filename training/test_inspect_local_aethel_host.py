"""Contrato de bloqueo del inspector local cuando no existe Dataset/GPU autorizado."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
INSPECTOR = ROOT / "training" / "inspect_local_aethel_host.py"


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        report_path = root / "inspection.json"
        completed = subprocess.run(
            [
                "python3",
                str(INSPECTOR),
                "--source-dir",
                str(ROOT),
                "--data-dir",
                str(root / "missing-dataset"),
                "--output-dir",
                str(root / "output"),
                "--report",
                str(report_path),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        report = json.loads(report_path.read_text(encoding="utf-8"))
    assert completed.returncode == 2
    assert report["seed_status"] == "BLOCKED"
    assert "dataset_directory_missing" in report["blockers"]
    assert report["edge_status"] == "REQUIRES_SEPARATE_CAPACITY_PLAN"
    assert report["storage"]["writable"] is True
    print("PASS: inspector local bloquea host sin Dataset/GPU antes de Seed")


if __name__ == "__main__":
    main()
