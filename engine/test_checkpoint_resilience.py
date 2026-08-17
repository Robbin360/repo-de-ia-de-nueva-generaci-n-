"""Contratos sin GPU para checkpoints portátiles y reanudación segura."""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).parent))
from train_aethel_gpu import prune_portable_snapshots, snapshot_tokenizer, validate_resume_metadata


def test_tokenizer_snapshot_is_portable_and_hashed() -> None:
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        source = root / "source.json"
        source.write_text('{"version":"bpe"}\n', encoding="utf-8")
        artifact, digest = snapshot_tokenizer(source, root / "run")
        assert artifact.name == "tokenizer.json"
        assert artifact.read_text(encoding="utf-8") == source.read_text(encoding="utf-8")
        assert digest == hashlib.sha256(source.read_bytes()).hexdigest()


def test_resume_rejects_mismatched_config_or_tokenizer() -> None:
    payload = {"config": {"dim": 64}, "tokenizer_sha256": "abc"}
    validate_resume_metadata(payload, {"dim": 64}, "abc")
    try:
        validate_resume_metadata(payload, {"dim": 128}, "abc")
    except ValueError as error:
        assert "configuración" in str(error)
    else:
        raise AssertionError("Se aceptó una configuración incompatible")
    try:
        validate_resume_metadata(payload, {"dim": 64}, "def")
    except ValueError as error:
        assert "tokenizador" in str(error)
    else:
        raise AssertionError("Se aceptó un tokenizador incompatible")


def test_snapshot_retention_keeps_latest_portable_points() -> None:
    with TemporaryDirectory() as temporary:
        output = Path(temporary)
        for step in (192, 384, 576, 768):
            (output / f"step_{step:08d}.pt").write_bytes(b"portable")
        (output / "latest.pt").write_bytes(b"full-state")
        retained = prune_portable_snapshots(output, keep_snapshots=2)
        assert retained == ["step_00000576.pt", "step_00000768.pt"]
        assert (output / "latest.pt").exists()
        assert not (output / "step_00000192.pt").exists()
        assert not (output / "step_00000384.pt").exists()


if __name__ == "__main__":
    test_tokenizer_snapshot_is_portable_and_hashed()
    test_resume_rejects_mismatched_config_or_tokenizer()
    test_snapshot_retention_keeps_latest_portable_points()
    print("OK")
