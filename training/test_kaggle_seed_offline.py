"""Contrato estático de la ruta Seed offline; no reserva GPU ni inicia entrenamiento."""
from __future__ import annotations

import os
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
RUNNER = ROOT / "training" / "run_kaggle_seed_offline.sh"


def main() -> None:
    blocked = subprocess.run(
        ["bash", str(RUNNER)],
        text=True,
        capture_output=True,
        env={**os.environ, "AETHEL_RUN_AUTHORIZED": ""},
    )
    assert blocked.returncode == 3
    assert "AETHEL_RUN_AUTHORIZED=YES" in blocked.stderr

    content = RUNNER.read_text(encoding="utf-8")
    required_fragments = (
        "validate_aethel_knowledge_package.py",
        '"$AETHEL_DATA_DIR/corpus"',
        '"$AETHEL_DATA_DIR/tokenizer.json"',
        "AETHEL_LAB_FALLBACK_AUTHORIZED",
        "--allow-pytorch-fallback",
        "holdout-en-00000.jsonl.gz",
        "holdout-es-00000.jsonl.gz",
        "recovery_receipt.json",
        "checkpoint_inspection.json",
    )
    for fragment in required_fragments:
        assert fragment in content, f"Falta contrato Seed offline: {fragment}"

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "output"
        missing_controls = subprocess.run(
            ["bash", str(RUNNER)],
            text=True,
            capture_output=True,
            env={**os.environ, "AETHEL_RUN_AUTHORIZED": "YES", "AETHEL_DATA_DIR": directory, "AETHEL_OUTPUT_DIR": str(output)},
        )
        assert missing_controls.returncode == 1
        assert "Faltan archivos de control obligatorios" in missing_controls.stdout
        assert "Activa Accelerator=GPU" not in missing_controls.stderr

    with tempfile.TemporaryDirectory() as directory:
        package = Path(directory)
        output = package / "output"
        (package / "package_manifest.json").write_text(json.dumps({
            "dataset_id": "aethel-knowledge-reasoning-bilingual-v1",
            "source_documents_sha256": "source-hash",
            "corpus_files": [],
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": "0" * 64,
                "derived_from": "train split only",
                "source_documents_sha256": "source-hash",
            },
        }), encoding="utf-8")
        (package / "metadata.json").write_text(json.dumps({
            "dataset_id": "aethel-knowledge-reasoning-bilingual-v1",
            "documents_sha256": "source-hash",
            "document_count": 0,
        }), encoding="utf-8")
        (package / "validation_report.json").write_text(json.dumps({"valid": True, "network_requests": 0}), encoding="utf-8")
        invalid_tokenizer = subprocess.run(
            ["bash", str(RUNNER)],
            text=True,
            capture_output=True,
            env={**os.environ, "AETHEL_RUN_AUTHORIZED": "YES", "AETHEL_DATA_DIR": str(package), "AETHEL_OUTPUT_DIR": str(output)},
        )
        assert invalid_tokenizer.returncode == 1
        assert "Tokenizador ausente o con hash inválido" in invalid_tokenizer.stdout
        assert "Activa Accelerator=GPU" not in invalid_tokenizer.stderr
    print("PASS: la ruta Seed offline exige autorización, valida el paquete y conserva evidencia reanudable")


if __name__ == "__main__":
    main()
