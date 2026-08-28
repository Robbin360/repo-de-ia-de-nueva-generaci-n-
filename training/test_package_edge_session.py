import hashlib
import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        output = root / "edge-session"
        output.mkdir()
        contents = {
            "latest.pt": b"checkpoint-bytes",
            "tokenizer.json": b'{"model":"test"}\n',
            "metrics_rank_0.jsonl": b'{"step":1}\n',
            "router_diagnostic.json": b'{"status":"ok"}\n',
            "recovery_receipt.json": b'{"reason":"final"}\n',
            "aethel_direct_validation.json": b'{"status":"MEASURED_NOT_PROMOTED"}\n',
        }
        for name, content in contents.items():
            (output / name).write_bytes(content)
        data_manifest = root / "prepared_manifest.json"
        data_manifest.write_text('{"schema_version":1,"shards":[]}\n', encoding="utf-8")

        package = root / "edge-session.tar.gz"
        command = [
            sys.executable,
            str(ROOT / "package_edge_session.py"),
            "--output",
            str(output),
            "--package",
            str(package),
            "--phase-id",
            "edge-v1-session-001",
            "--session-target-step",
            "9000",
            "--schedule-total-steps",
            "90000",
            "--data-manifest",
            str(data_manifest),
        ]
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        assert "AETHEL_EDGE_SESSION_PRESERVATION_READY" in completed.stdout
        assert package.is_file()

        receipt = json.loads((output / "edge_session_preservation_receipt.json").read_text(encoding="utf-8"))
        assert receipt["limits"]["checkpoint_uploaded"] is False
        assert receipt["checkpoint"]["sha256"] == sha256_file(output / "latest.pt")
        assert receipt["data_manifest"]["sha256"] == sha256_file(data_manifest)
        assert receipt["package"]["sha256"] == sha256_file(package)
        assert receipt["required_manual_action"] == "SAVE_KAGGLE_VERSION_IMMEDIATELY"

        with tarfile.open(package, "r:gz") as archive:
            archived = set(archive.getnames())
        assert "edge-session/SAVE_KAGGLE_VERSION_NOW.txt" in archived
        assert "edge-session/latest.pt" in archived
        assert "edge-session/metrics_rank_0.jsonl" in archived
        assert "edge-session/prepared_manifest.json" in archived
        assert "edge_session_preservation_receipt.json" not in archived

    print("AETHEL_EDGE_SESSION_PACKAGE_TEST_OK")


if __name__ == "__main__":
    main()
