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

            def fake_rows(source: dict, cache_dir=None):
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


    def test_retry_delay_honors_numeric_and_http_date_retry_after(self) -> None:
        numeric = MODULE.urllib.error.HTTPError("https://example.invalid", 429, "busy", {"Retry-After": "17"}, None)
        self.assertEqual(MODULE.retry_delay(numeric, 0, {}), 17.0)
        with patch.object(MODULE.time, "time", return_value=1_000.0):
            dated = MODULE.urllib.error.HTTPError("https://example.invalid", 503, "busy", {"Retry-After": "Thu, 01 Jan 1970 00:16:45 GMT"}, None)
            self.assertEqual(MODULE.retry_delay(dated, 0, {}), 5.0)

    def test_hf_rows_api_reuses_cached_page_after_retry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            source = {
                "id": "test-api",
                "kind": "hf_rows_api",
                "dataset": "test/dataset",
                "config": "default",
                "revision": "a" * 40,
                "batch_size": 2,
                "max_retries": 2,
                "retry_backoff_seconds": 0,
                "max_retry_delay_seconds": 1,
            }
            response = {"rows": [{"row": {"text": "Cached reference document."}}]}
            with patch.object(MODULE.urllib.request, "urlopen", return_value=DummyResponse(response)) as open_mock:
                rows = list(MODULE.hf_rows_api(source, cache_dir))
                self.assertEqual(rows, [{"text": "Cached reference document."}])
                self.assertEqual(open_mock.call_count, 1)
            with patch.object(MODULE.urllib.request, "urlopen") as second_open:
                self.assertEqual(list(MODULE.hf_rows_api(source, cache_dir)), rows)
                second_open.assert_not_called()

    def test_download_resumable_continues_partial_with_range(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache_dir = Path(temporary)
            source = {"id": "download", "url": "https://example.invalid/data.jsonl", "max_retries": 1}
            target = cache_dir / "download" / "data.jsonl"
            target.parent.mkdir(parents=True)
            partial = target.with_suffix(".jsonl.part")
            partial.write_bytes(b"first-")
            with patch.object(MODULE.urllib.request, "urlopen", return_value=DummyResponse(b"second", headers={"Content-Range": "bytes 6-11/12"})) as open_mock:
                self.assertEqual(MODULE.download_resumable(source, cache_dir), target)
                self.assertEqual(target.read_bytes(), b"first-second")
                request = open_mock.call_args.args[0]
                self.assertEqual(request.headers["Range"], "bytes=6-")


class DummyResponse:
    def __init__(self, payload: object, headers: dict[str, str] | None = None) -> None:
        self.headers = headers or {}
        self.payload = payload
        self._read = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self, size: int = -1) -> bytes:
        if self._read:
            return b""
        self._read = True
        if isinstance(self.payload, bytes):
            return self.payload
        return json.dumps(self.payload).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
