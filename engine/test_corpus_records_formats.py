from __future__ import annotations

import gzip
import tempfile
from pathlib import Path

from train_aethel_gpu import corpus_records


def write_records(path: Path, *, compressed: bool) -> None:
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8") as handle:
        handle.write('{"text":"uno"}\n')
        handle.write('{"text":"dos"}\n')


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_records(root / "train-00000.jsonl", compressed=False)
        assert list(corpus_records(root)) == ["uno", "dos"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_records(root / "train-00000.jsonl.gz", compressed=True)
        assert list(corpus_records(root)) == ["uno", "dos"]

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        write_records(root / "train-00000.jsonl", compressed=False)
        write_records(root / "train-00001.jsonl.gz", compressed=True)
        try:
            list(corpus_records(root))
        except RuntimeError as error:
            assert "mezcla shards" in str(error)
        else:
            raise AssertionError("Una mezcla de JSONL y gzip debe quedar bloqueada.")

    print("AETHEL_CORPUS_RECORDS_FORMATS_OK")


if __name__ == "__main__":
    main()
