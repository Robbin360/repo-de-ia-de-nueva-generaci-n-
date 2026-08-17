"""Pruebas unitarias del preparador bilingüe sin descargar fuentes ni usar GPU."""
from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "engine" / "prepare_bilingual_corpus.py"
SPEC = importlib.util.spec_from_file_location("prepare_bilingual_corpus", MODULE_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class PrepareBilingualCorpusTest(unittest.TestCase):
    def test_round_robin_quota_hashes_and_language_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = {
                "approval_required": False,
                "purpose": "test",
                "minimum_documents_by_language": {"en": 2, "es": 2},
                "sources": [
                    {"id": "en", "kind": "hf_text", "language": "en", "document_limit": 2, "license": "test", "provenance_url": "https://example.invalid/en", "revision": "a" * 40, "enabled": True},
                    {"id": "es", "kind": "hf_text", "language": "es", "document_limit": 2, "license": "test", "provenance_url": "https://example.invalid/es", "revision": "b" * 40, "enabled": True},
                ],
                "filters": {"min_characters": 10, "max_characters": 1000, "deduplicate_exact": True, "validation_percent": 0.0},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            texts = {"en": iter([{"text": "English reference document A."}, {"text": "English reference document B."}]), "es": iter([{"text": "Documento español de referencia A."}, {"text": "Documento español de referencia B."}])}

            def fake_rows(source: dict):
                return texts[source["id"]]

            with patch.object(MODULE, "source_rows", side_effect=fake_rows):
                MODULE.run(argparse.Namespace(manifest=str(manifest_path), output=str(root / "output"), shard_documents=1, seed=17, allow_network=True))

            report = json.loads((root / "output" / "prepared_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(report["language_counts"]["en"]["accepted"], 2)
            self.assertEqual(report["language_counts"]["es"]["accepted"], 2)
            self.assertEqual(len(report["shards"]), 4)
            rows = []
            for shard in sorted((root / "output").glob("train-*.jsonl.gz")):
                with gzip.open(shard, "rt", encoding="utf-8") as handle:
                    rows.extend(json.loads(line) for line in handle)
            self.assertEqual({row["language"] for row in rows}, {"en", "es"})
            self.assertTrue(all(row["sha256"] for row in rows))

    def test_rejects_manifest_that_still_requires_approval(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = Path(temporary) / "manifest.json"
            manifest_path.write_text(json.dumps({"approval_required": True, "sources": []}), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exige aprobación"):
                MODULE.run(argparse.Namespace(manifest=str(manifest_path), output=str(Path(temporary) / "output"), shard_documents=1, seed=17, allow_network=True))


if __name__ == "__main__":
    unittest.main()
