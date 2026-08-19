import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from validate_aethel_knowledge_corpus import validate


class OfflineValidatorTests(unittest.TestCase):
    def test_validates_a_minimal_traced_offline_corpus(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            raw.mkdir()
            source_file = raw / "source.bz2"
            source_file.write_bytes(b"official-source-placeholder-for-test")
            source_hash = hashlib.sha256(source_file.read_bytes()).hexdigest()
            source = {"id": "source-en", "language": "en", "required": True, "url": "https://example.test/source.bz2", "license": "CC-BY-SA-4.0", "target_documents": 1, "min_characters": 1, "max_characters": 100}
            manifest = {
                "dataset_id": "test-corpus",
                "sources": [source],
                "splits": {"holdout_percent": 0, "seed": "test", "train_percent": 100},
                "acceptance": {"minimum_documents_per_language": 1, "required_languages": ["en"], "minimum_total_documents": 1, "minimum_domain_coverage_per_language": ["general"], "require_holdout": False},
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            text = "Verifiable original source text."
            record = {"document_id": "document", "split": "train", "language": "en", "source_id": "source-en", "source_url": source["url"], "license": source["license"], "title": "Title", "revision_id": "1", "domains": ["general"], "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "text": text}
            documents = root / "documents.jsonl"
            documents.write_text(json.dumps(record) + "\n", encoding="utf-8")
            metadata = {"dataset_id": "test-corpus", "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "documents_sha256": hashlib.sha256(documents.read_bytes()).hexdigest(), "document_count": 1, "sources": [{"id": "source-en", "url": source["url"], "license": source["license"], "path": source_file.name, "sha256": source_hash}]}
            (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            report = validate(root, manifest_path)
            self.assertTrue(report["valid"])
            self.assertEqual(report["network_requests"], 0)


if __name__ == "__main__":
    unittest.main()
