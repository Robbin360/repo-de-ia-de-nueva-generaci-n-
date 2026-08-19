#!/usr/bin/env python3
"""Convierte un corpus validado a shards gzip para distribución y tokenización offline."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package(dataset_dir: Path, output_dir: Path, documents_per_shard: int, include_tokenizer_input: bool = False, tokenizer_path: Path | None = None) -> dict[str, Any]:
    report = json.loads((dataset_dir / "validation_report.json").read_text(encoding="utf-8"))
    if not report.get("valid") or report.get("network_requests") != 0:
        raise RuntimeError("El corpus debe superar primero la validación offline antes de empaquetarse.")
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_dir = output_dir / "corpus"
    corpus_dir.mkdir(exist_ok=True)
    tokenizer_dir = output_dir / "tokenizer-input"
    if include_tokenizer_input:
        tokenizer_dir.mkdir(exist_ok=True)
    writers: dict[tuple[str, str, int], Any] = {}
    counts = Counter()
    files: list[Path] = []

    def writer_for(section: str, language: str, index: int) -> Any:
        key = (section, language, index)
        if key not in writers:
            target_root = tokenizer_dir if section == "tokenizer-input" else corpus_dir
            prefix = "tokenizer" if section == "tokenizer-input" else section
            path = target_root / f"{prefix}-{language}-{index:05d}.jsonl.gz"
            writers[key] = gzip.open(path, "wt", encoding="utf-8")
            files.append(path)
        return writers[key]

    try:
        with (dataset_dir / "documents.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                language = record["language"]
                split = record["split"]
                ordinal = counts[f"{split}:{language}"]
                shard_index = ordinal // documents_per_shard
                writer_for(split, language, shard_index).write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                counts[f"{split}:{language}"] += 1
                if include_tokenizer_input and split == "train":
                    tokenizer_ordinal = counts[f"tokenizer:{language}"]
                    tokenizer_index = tokenizer_ordinal // documents_per_shard
                    writer_for("tokenizer-input", language, tokenizer_index).write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[f"tokenizer:{language}"] += 1
    finally:
        for handle in writers.values():
            handle.close()

    shutil.copy2(dataset_dir / "metadata.json", output_dir / "metadata.json")
    shutil.copy2(dataset_dir / "validation_report.json", output_dir / "validation_report.json")
    tokenizer_metadata: dict[str, Any] | None = None
    if tokenizer_path is not None:
        if not tokenizer_path.is_file():
            raise RuntimeError(f"No existe el tokenizador solicitado: {tokenizer_path}")
        tokenizer_destination = output_dir / "tokenizer.json"
        shutil.copy2(tokenizer_path, tokenizer_destination)
        tokenizer_metadata = {
            "path": tokenizer_destination.name,
            "sha256": sha256_file(tokenizer_destination),
            "derived_from": "train split only",
            "source_documents_sha256": metadata["documents_sha256"],
        }
        source_manifest = tokenizer_path.with_suffix(".manifest.json")
        if source_manifest.is_file():
            tokenizer_metadata["training_manifest"] = json.loads(source_manifest.read_text(encoding="utf-8"))
    package_manifest = {
        "dataset_id": metadata["dataset_id"],
        "source_documents_sha256": metadata["documents_sha256"],
        "documents_per_shard": documents_per_shard,
        "counts": dict(counts),
        "corpus_files": [
            {"path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(corpus_dir.glob("*.jsonl.gz"))
        ],
        "tokenizer_input_files": [
            {"path": str(path.relative_to(output_dir)), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in sorted(tokenizer_dir.glob("*.jsonl.gz"))
        ],
        "tokenizer_input_included": include_tokenizer_input,
        "holdout_excluded_from_tokenizer": True,
        "tokenizer": tokenizer_metadata,
        "offline_training_ready": True,
    }
    (output_dir / "package_manifest.json").write_text(json.dumps(package_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return package_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="/home/ubuntu/aethel-knowledge-corpus-v1")
    parser.add_argument("--output-dir", default="/home/ubuntu/aethel-knowledge-corpus-v1-package")
    parser.add_argument("--documents-per-shard", type=int, default=2000)
    parser.add_argument("--include-tokenizer-input", action="store_true", help="Incluye copias de train sólo para entrenar el tokenizador; no se recomienda para el Dataset final.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizador BPE ya entrenado localmente desde train; se copiará y se hasheará en el paquete final.")
    args = parser.parse_args()
    tokenizer = Path(args.tokenizer).resolve() if args.tokenizer else None
    result = package(Path(args.dataset_dir).resolve(), Path(args.output_dir).resolve(), args.documents_per_shard, args.include_tokenizer_input, tokenizer)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
