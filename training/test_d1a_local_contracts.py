"""Pruebas locales con fixtures sintéticas; no usan Dataset Aethel, GPU ni pesos."""
from __future__ import annotations

import gzip
import hashlib
import json
import tempfile
from pathlib import Path

from summarize_d1a_router_metrics import load_events, summarize
from validate_aethel_train_only_mount import validate_train_only


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_gzip(path: Path, value: bytes) -> None:
    with gzip.open(path, "wb") as handle:
        handle.write(value)


def test_train_only_validator_ignores_holdout_fixture() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        package = Path(temporary)
        corpus = package / "corpus"
        corpus.mkdir()
        train_en = corpus / "train-en-00000.jsonl.gz"
        train_es = corpus / "train-es-00000.jsonl.gz"
        holdout = corpus / "holdout-en-00000.jsonl.gz"
        write_gzip(train_en, b'{"synthetic": "train-en"}\n')
        write_gzip(train_es, b'{"synthetic": "train-es"}\n')
        write_gzip(holdout, b"unexpected holdout fixture is not opened\n")
        tokenizer = package / "tokenizer.json"
        tokenizer.write_text('{"synthetic": true}\n', encoding="utf-8")
        manifest = {
            "dataset_id": "fixture-only",
            "counts": {"train:en": 1, "train:es": 1, "holdout:en": 99},
            "tokenizer": {"path": "tokenizer.json", "sha256": sha256(tokenizer), "derived_from": "train split only"},
            "corpus_files": [
                {"path": "corpus/train-en-00000.jsonl.gz", "bytes": train_en.stat().st_size, "sha256": sha256(train_en)},
                {"path": "corpus/train-es-00000.jsonl.gz", "bytes": train_es.stat().st_size, "sha256": sha256(train_es)},
                {"path": "corpus/holdout-en-00000.jsonl.gz", "bytes": 1, "sha256": "not-read-by-test"},
            ],
        }
        (package / "package_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = validate_train_only(package)
        assert report["valid"] is True
        assert report["holdout_content_read"] is False
        assert report["train_shards_verified"] == ["corpus/train-en-00000.jsonl.gz", "corpus/train-es-00000.jsonl.gz"]


def test_metrics_summary_uses_only_jsonl_events() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        metrics = root / "metrics_rank_0.jsonl"
        events = [
            {"step": 1, "loss": 4.0, "tokens_seen": 2048, "config": {"experts": 8}, "router_health": {"healthy": True}, "routing": [{"entropy": 0.7, "max_load": 0.3, "imbalance": 0.1, "bias": [0.1, -0.1]}]},
            {"step": 2, "loss": 3.0, "tokens_seen": 4096, "config": {"experts": 8}, "router_health": {"healthy": False}, "routing": [{"entropy": 0.4, "max_load": 0.5, "imbalance": 0.2, "bias": [0.2, -0.2]}]},
        ]
        metrics.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
        report = summarize(load_events(metrics))
        assert report["status"] == "D1A_METRICS_SUMMARIZED"
        assert report["router_health"] == {"healthy_steps": 1, "unhealthy_steps": 1}
        assert report["layers"][0]["entropy"]["minimum"] == 0.4
        assert report["limits"]["checkpoint_loaded"] is False
        assert report["limits"]["holdout_content_read"] is False


def test_metrics_summary_rejects_checkpoint_path() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint = Path(temporary) / "latest.pt"
        checkpoint.write_bytes(b"not loaded")
        try:
            load_events(checkpoint)
        except ValueError as error:
            assert "metrics_rank_0.jsonl" in str(error)
        else:
            raise AssertionError("Se debía rechazar una ruta de checkpoint.")


def test_d1a_launcher_uses_supported_router_limit_argument() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    trainer = (repository_root / "engine" / "train_aethel_gpu.py").read_text(encoding="utf-8")
    launcher = (repository_root / "training" / "run_kaggle_d1a_router_diagnostic.sh").read_text(encoding="utf-8")
    assert 'parser.add_argument("--router-bias-limit", type=float, default=0.5)' in trainer
    assert "--router-bias-limit 0.5" in launcher
    assert 'if [[ -e "$OUTPUT_DIR" ]]' in launcher
    assert "evaluate_nextgen.py" not in launcher
    assert "inspect_checkpoint.py" not in launcher
    assert "--resume" not in launcher


if __name__ == "__main__":
    test_train_only_validator_ignores_holdout_fixture()
    test_metrics_summary_uses_only_jsonl_events()
    test_metrics_summary_rejects_checkpoint_path()
    test_d1a_launcher_uses_supported_router_limit_argument()
    print("D1A_LOCAL_CONTRACT_TESTS_PASSED")
