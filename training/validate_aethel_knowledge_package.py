#!/usr/bin/env python3
"""Verifica offline la integridad del paquete distribuible de Aethel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "package_manifest.json"
    metadata_path = package_dir / "metadata.json"
    report_path = package_dir / "validation_report.json"
    errors: list[str] = []
    if not all(path.is_file() for path in (manifest_path, metadata_path, report_path)):
        return {"valid": False, "network_requests": 0, "errors": ["Faltan archivos de control obligatorios."], "warnings": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not source_report.get("valid") or source_report.get("network_requests") != 0:
        errors.append("La validación de origen no certifica un corpus válido y offline.")
    if manifest.get("dataset_id") != metadata.get("dataset_id"):
        errors.append("dataset_id inconsistente entre manifiesto de paquete y metadata.")
    if manifest.get("source_documents_sha256") != metadata.get("documents_sha256"):
        errors.append("El hash de documentos de origen no coincide.")
    records = Counter()
    shard_count = 0
    for shard in manifest.get("corpus_files", []):
        path = package_dir / shard["path"]
        shard_count += 1
        if not path.is_file():
            errors.append(f"Falta el shard {shard['path']}.")
            continue
        if sha256_file(path) != shard["sha256"]:
            errors.append(f"Hash de shard inválido: {shard['path']}.")
            continue
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"JSONL inválido en {shard['path']}:{line_number}.")
                    break
                if row.get("split") not in {"train", "holdout"} or row.get("language") not in {"en", "es"} or not row.get("text"):
                    errors.append(f"Registro inválido en {shard['path']}:{line_number}.")
                    break
                records[f"{row['split']}:{row['language']}"] += 1
    if sum(records.values()) != metadata.get("document_count"):
        errors.append("El número de registros de los shards no coincide con metadata.")
    tokenizer = manifest.get("tokenizer")
    if not tokenizer:
        errors.append("El paquete no contiene el tokenizador versionado.")
    else:
        tokenizer_path = package_dir / tokenizer["path"]
        if not tokenizer_path.is_file() or sha256_file(tokenizer_path) != tokenizer["sha256"]:
            errors.append("Tokenizador ausente o con hash inválido.")
        elif tokenizer.get("derived_from") != "train split only" or tokenizer.get("source_documents_sha256") != metadata.get("documents_sha256"):
            errors.append("Procedencia de tokenizador inválida.")
    return {
        "valid": not errors,
        "network_requests": 0,
        "dataset_id": metadata.get("dataset_id"),
        "shard_count": shard_count,
        "record_counts": dict(records),
        "errors": errors,
        "warnings": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", default="/home/ubuntu/aethel-knowledge-corpus-v1-package")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    package_dir = Path(args.package_dir).resolve()
    result = validate(package_dir)
    report_path = Path(args.report).resolve() if args.report else package_dir / "package_validation_report.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))
    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
