import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from package_aethel_knowledge_corpus import package


class PackageCorpusTests(unittest.TestCase):
    def test_train_only_tokenizer_input_and_holdout_corpus_are_partitioned(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_dir = Path(temp_dir) / "dataset"
            dataset_dir.mkdir()
            metadata = {"dataset_id": "test", "documents_sha256": "source-hash"}
            (dataset_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            (dataset_dir / "validation_report.json").write_text(json.dumps({"valid": True, "network_requests": 0}), encoding="utf-8")
            rows = [
                {"document_id": "a", "language": "en", "split": "train", "text": "train text"},
                {"document_id": "b", "language": "en", "split": "holdout", "text": "holdout text"},
            ]
            (dataset_dir / "documents.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output_dir = Path(temp_dir) / "package"
            result = package(dataset_dir, output_dir, documents_per_shard=1, include_tokenizer_input=True)
            self.assertTrue(result["holdout_excluded_from_tokenizer"])
            tokenizer_rows = []
            for path in (output_dir / "tokenizer-input").glob("*.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    tokenizer_rows.extend(json.loads(line) for line in handle)
            self.assertEqual([row["document_id"] for row in tokenizer_rows], ["a"])
            corpus_rows = []
            for path in (output_dir / "corpus").glob("*.gz"):
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    corpus_rows.extend(json.loads(line) for line in handle)
            self.assertEqual({row["document_id"] for row in corpus_rows}, {"a", "b"})


if __name__ == "__main__":
    unittest.main()
