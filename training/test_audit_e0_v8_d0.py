#!/usr/bin/env python3
"""Regresiones directas para el auditor D0 de solo lectura."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from hashlib import sha256

from audit_e0_v8_d0 import D0AuditError, EXPECTED_SOURCE_RELEASE, audit


ROOT = Path(__file__).resolve().parent


def copy_json(source: Path, target: Path) -> None:
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")


def write_contract_fixture(training: Path, data: Path) -> None:
    evidence = json.loads((ROOT / "e0_v8_d0_evidence.json").read_text(encoding="utf-8"))
    contract = evidence["dataset_contract"]
    manifest = {
        "dataset_id": contract["dataset_id"],
        "counts": contract["counts"],
        "tokenizer": {"sha256": contract["tokenizer_sha256"]},
        "holdout_excluded_from_tokenizer": contract["holdout_excluded_from_tokenizer"],
    }
    manifest_text = json.dumps(manifest, sort_keys=True) + "\n"
    (data / "package_manifest.json").write_text(manifest_text, encoding="utf-8")
    evidence["dataset_contract"]["package_manifest_sha256"] = sha256(manifest_text.encode("utf-8")).hexdigest()
    (training / "e0_v8_d0_evidence.json").write_text(
        json.dumps(evidence), encoding="utf-8"
    )


def test_happy_path() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        training = source / "training"
        training.mkdir(parents=True)
        (training / "aethel_kaggle_source_release.json").write_text(
            json.dumps({"release": EXPECTED_SOURCE_RELEASE}), encoding="utf-8"
        )
        data = root / "data"
        data.mkdir()
        write_contract_fixture(training, data)
        report = audit(source, data, root / "output")
        assert report["status"] == "D0_AUDIT_READY"
        assert report["checkpoint"]["checkpoint_loaded"] is False
        assert report["holdout_scope"]["holdout_content_read"] is False
        assert report["dataset"]["manifest_metadata_verified"] is True
        assert report["router_final_healthy"] is False
        assert (root / "output" / "d0_audit.json").is_file()


def test_rejects_wrong_release() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "source"
        training = source / "training"
        training.mkdir(parents=True)
        (training / "aethel_kaggle_source_release.json").write_text(
            json.dumps({"release": "wrong-release"}), encoding="utf-8"
        )
        copy_json(ROOT / "e0_v8_d0_evidence.json", training / "e0_v8_d0_evidence.json")
        data = root / "data"
        data.mkdir()
        (data / "package_manifest.json").write_text("{}", encoding="utf-8")
        try:
            audit(source, data, root / "output")
        except D0AuditError as exc:
            assert "Release D0 incorrecto" in str(exc)
        else:
            raise AssertionError("D0 debe rechazar release de código incorrecto")


def main() -> None:
    test_happy_path()
    test_rejects_wrong_release()
    print("D0_LOCAL_AUDIT_TESTS_PASSED")


if __name__ == "__main__":
    main()
