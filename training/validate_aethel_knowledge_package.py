#!/usr/bin/env python3
"""Verifica offline la integridad del paquete distribuible de Aethel."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal


CONTRACT_FILENAME = "aethel_kaggle_decompressed_mount_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plaintext_path(compressed_path: str) -> str:
    if not compressed_path.endswith(".gz"):
        raise ValueError(f"El manifiesto contiene un shard no comprimido inesperado: {compressed_path}")
    return compressed_path.removesuffix(".gz")


def load_plaintext_contract(manifest: dict[str, Any], manifest_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    """Carga el contrato inmutable para verificar una descompresión hecha por Kaggle."""
    errors: list[str] = []
    contract_path = Path(__file__).with_name(CONTRACT_FILENAME)
    if not contract_path.is_file():
        return None, ["Falta el contrato de integridad para shards descomprimidos por Kaggle."]
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"No se puede leer el contrato de mount descomprimido: {error}"]

    if contract.get("schema") != "aethel-kaggle-decompressed-mount/v1":
        errors.append("Esquema inválido del contrato de mount descomprimido.")
    if contract.get("dataset_id") != manifest.get("dataset_id"):
        errors.append("dataset_id inconsistente entre el manifiesto y el contrato de mount descomprimido.")
    if contract.get("package_manifest_sha256") != sha256_file(manifest_path):
        errors.append("El contrato de mount no está vinculado al package_manifest.json montado.")

    expected_paths = {plaintext_path(entry["path"]) for entry in manifest.get("corpus_files", [])}
    plaintext_shards = contract.get("plaintext_shards")
    if not isinstance(plaintext_shards, dict) or set(plaintext_shards) != expected_paths:
        errors.append("Los shards del contrato de mount no coinciden exactamente con el manifiesto congelado.")
        return None, errors
    for path, details in plaintext_shards.items():
        if not isinstance(details, dict) or not isinstance(details.get("bytes"), int) or not isinstance(details.get("sha256"), str):
            errors.append(f"Contrato inválido para shard descomprimido: {path}.")
    return (contract if not errors else None), errors


def detect_mount_mode(package_dir: Path, manifest: dict[str, Any]) -> tuple[Literal["gzip", "plaintext", "invalid"], list[str]]:
    """Distingue el paquete original de la exposición .jsonl de Kaggle sin aceptar un estado mixto."""
    compressed_paths = [entry["path"] for entry in manifest.get("corpus_files", [])]
    plain_paths = [plaintext_path(path) for path in compressed_paths]
    compressed_present = [path for path in compressed_paths if (package_dir / path).is_file()]
    plain_present = [path for path in plain_paths if (package_dir / path).is_file()]
    if len(compressed_present) == len(compressed_paths) and not plain_present:
        return "gzip", []
    if len(plain_present) == len(plain_paths) and not compressed_present:
        return "plaintext", []
    errors = [
        "El montaje del corpus es incompleto o mezcla shards comprimidos y descomprimidos; se rechaza para evitar una validación ambigua.",
        f"Shards .jsonl.gz presentes: {len(compressed_present)}/{len(compressed_paths)}.",
        f"Shards .jsonl presentes: {len(plain_present)}/{len(plain_paths)}.",
    ]
    return "invalid", errors


def validate(package_dir: Path) -> dict[str, Any]:
    manifest_path = package_dir / "package_manifest.json"
    metadata_path = package_dir / "metadata.json"
    report_path = package_dir / "validation_report.json"
    errors: list[str] = []
    if not all(path.is_file() for path in (manifest_path, metadata_path, report_path)):
        return {
            "valid": False,
            "network_requests": 0,
            "errors": ["Faltan archivos de control obligatorios."],
            "warnings": [],
            "mount_format": "invalid",
        }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_report = json.loads(report_path.read_text(encoding="utf-8"))
    if not source_report.get("valid") or source_report.get("network_requests") != 0:
        errors.append("La validación de origen no certifica un corpus válido y offline.")
    if manifest.get("dataset_id") != metadata.get("dataset_id"):
        errors.append("dataset_id inconsistente entre manifiesto de paquete y metadata.")
    if manifest.get("source_documents_sha256") != metadata.get("documents_sha256"):
        errors.append("El hash de documentos de origen no coincide.")

    mount_format, mount_errors = detect_mount_mode(package_dir, manifest)
    errors.extend(mount_errors)
    plaintext_contract: dict[str, Any] | None = None
    if mount_format == "plaintext":
        plaintext_contract, contract_errors = load_plaintext_contract(manifest, manifest_path)
        errors.extend(contract_errors)

    records = Counter()
    shard_count = 0
    for shard in manifest.get("corpus_files", []):
        compressed_path = shard["path"]
        path = package_dir / (compressed_path if mount_format == "gzip" else plaintext_path(compressed_path))
        shard_count += 1
        if mount_format == "invalid" or not path.is_file():
            errors.append(f"Falta el shard {compressed_path}.")
            continue
        if mount_format == "gzip":
            expected_bytes = shard["bytes"]
            expected_sha256 = shard["sha256"]
        else:
            assert plaintext_contract is not None
            plain_details = plaintext_contract["plaintext_shards"][plaintext_path(compressed_path)]
            expected_bytes = plain_details["bytes"]
            expected_sha256 = plain_details["sha256"]
        if path.stat().st_size != expected_bytes:
            errors.append(f"Tamaño de shard inválido: {path.relative_to(package_dir)}.")
            continue
        if sha256_file(path) != expected_sha256:
            errors.append(f"Hash de shard inválido: {path.relative_to(package_dir)}.")
            continue
        opener = gzip.open if mount_format == "gzip" else Path.open
        with opener(path, "rt", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    errors.append(f"JSONL inválido en {path.relative_to(package_dir)}:{line_number}.")
                    break
                if row.get("split") not in {"train", "holdout"} or row.get("language") not in {"en", "es"} or not row.get("text"):
                    errors.append(f"Registro inválido en {path.relative_to(package_dir)}:{line_number}.")
                    break
                records[f"{row['split']}:{row['language']}"] += 1
    if sum(records.values()) != metadata.get("document_count"):
        errors.append("El número de registros de los shards no coincide con metadata.")
    expected_counts = manifest.get("counts")
    if dict(records) != expected_counts:
        errors.append("Los conteos por split e idioma no coinciden con package_manifest.json.")
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
        "mount_format": "gzip" if mount_format == "gzip" else "kaggle_plaintext_verified" if mount_format == "plaintext" else "invalid",
        "integrity_mode": "compressed_sha256" if mount_format == "gzip" else "plaintext_sha256_bound_to_manifest" if mount_format == "plaintext" else "none",
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
