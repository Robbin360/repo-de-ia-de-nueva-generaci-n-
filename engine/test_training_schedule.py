"""Pruebas del calendario de aprendizaje sin iniciar entrenamiento ni usar GPU."""
from __future__ import annotations

import gzip
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_aethel_gpu import corpus_records, learning_rate_at_step


def test_warmup_and_cosine_floor_are_bounded() -> None:
    peak, floor = 3e-4, 3e-5
    assert learning_rate_at_step(1, 1000, peak, floor, 100) == peak / 100
    assert learning_rate_at_step(100, 1000, peak, floor, 100) == peak
    assert floor <= learning_rate_at_step(500, 1000, peak, floor, 100) <= peak
    assert learning_rate_at_step(1000, 1000, peak, floor, 100) == floor


def test_corpus_records_accepts_verified_plaintext_kaggle_mount() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        corpus = Path(temporary)
        (corpus / "train-en-00000.jsonl").write_text('{"text":"plain"}\n', encoding="utf-8")
        assert list(corpus_records(corpus)) == ["plain"]


def test_corpus_records_rejects_mixed_mount_formats() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        corpus = Path(temporary)
        (corpus / "train-en-00000.jsonl").write_text('{"text":"plain"}\n', encoding="utf-8")
        with gzip.open(corpus / "train-es-00000.jsonl.gz", "wt", encoding="utf-8") as handle:
            handle.write('{"text":"compressed"}\n')
        try:
            list(corpus_records(corpus))
        except RuntimeError as error:
            assert "mezcla shards" in str(error)
        else:
            raise AssertionError("El montaje mixto debe bloquearse")


if __name__ == "__main__":
    test_warmup_and_cosine_floor_are_bounded()
    test_corpus_records_accepts_verified_plaintext_kaggle_mount()
    test_corpus_records_rejects_mixed_mount_formats()
    print("OK")
