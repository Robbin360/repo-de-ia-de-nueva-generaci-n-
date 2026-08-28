"""Prueba CPU del paquete de preservación sin cargar checkpoints."""
from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
PACKAGER = ROOT / "training" / "package_router_jitter_rerun.py"
REQUIRED = (
    "latest.pt",
    "tokenizer.json",
    "metrics_rank_0.jsonl",
    "router_diagnostic.json",
    "recovery_receipt.json",
    "aethel_direct_validation.json",
)
SAVE_VERSION_GATE = "SAVE_KAGGLE_VERSION_NOW.txt"


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        output = root / "output"
        output.mkdir()
        for index, name in enumerate(REQUIRED):
            (output / name).write_bytes(f"fixture-{index}".encode("utf-8"))
        package = root / "preservation.tar.gz"
        completed = subprocess.run(
            [sys.executable, str(PACKAGER), "--output", str(output), "--package", str(package)],
            check=True,
            text=True,
            capture_output=True,
        )
        assert "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_READY" in completed.stdout
        receipt = json.loads((output / "checkpoint_preservation_receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_READY"
        assert receipt["PERSISTENCE_ACTION_REQUIRED"] == "SAVE_KAGGLE_VERSION"
        assert receipt["limits"]["checkpoint_uploaded"] is False
        assert (output / SAVE_VERSION_GATE).is_file()
        assert "Save Version" in (output / SAVE_VERSION_GATE).read_text(encoding="utf-8")
        with tarfile.open(package, "r:gz") as archive:
            archived = {Path(member.name).name for member in archive.getmembers() if member.isfile()}
        assert archived == {*REQUIRED, SAVE_VERSION_GATE}
    print("AETHEL_ROUTER_JITTER_RERUN_PACKAGER_CPU_VALIDATED")


if __name__ == "__main__":
    main()
