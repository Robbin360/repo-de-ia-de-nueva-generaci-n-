#!/usr/bin/env python3
"""Valida offline un Dataset Aethel materializado, sin realizar solicitudes de red."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(dataset_dir: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata_path = dataset_dir / "metadata.json"
    documents_path = dataset_dir / "documents.jsonl"
    if not metadata_path.is_file() or not documents_path.is_file():
        raise RuntimeError("Faltan metadata.json o documents.jsonl en el Dataset materializado.")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    if metadata.get("dataset_id") != manifest.get("dataset_id"):
        errors.append("dataset_id no coincide con el manifiesto.")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if metadata.get("manifest_sha256") != manifest_hash:
        errors.append("El hash del manifiesto no coincide.")
    documents_hash = sha256_file(documents_path)
    if metadata.get("documents_sha256") != documents_hash:
        errors.append("El hash de documents.jsonl no coincide.")

    declared_sources = {source["id"]: source for source in manifest["sources"]}
    inventory_sources = {source["id"]: source for source in metadata.get("sources", [])}
    for source_id, source in declared_sources.items():
        inventory = inventory_sources.get(source_id)
        if inventory is None:
            errors.append(f"No existe inventario para la fuente requerida: {source_id}.")
            continue
        raw_path = dataset_dir / "raw" / inventory["path"]
        if not raw_path.is_file():
            errors.append(f"Falta el artefacto de fuente local: {raw_path.name}.")
        elif sha256_file(raw_path) != inventory.get("sha256"):
            errors.append(f"Hash incorrecto para el artefacto de fuente: {raw_path.name}.")
        if inventory.get("url") != source.get("url") or inventory.get("license") != source.get("license"):
            errors.append(f"La procedencia declarada no coincide para {source_id}.")

    languages = Counter()
    splits = Counter()
    domains: dict[str, Counter[str]] = {language: Counter() for language in manifest["acceptance"]["required_languages"]}
    source_counts = Counter()
    document_ids: set[str] = set()
    text_hashes: set[str] = set()
    line_count = 0
    with documents_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                errors.append(f"Línea vacía no permitida en documents.jsonl: {line_number}.")
                continue
            line_count += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                errors.append(f"JSON inválido en línea {line_number}: {error}.")
                continue
            required = {"document_id", "split", "language", "source_id", "source_url", "license", "title", "revision_id", "domains", "text_sha256", "text"}
            missing = sorted(required.difference(record))
            if missing:
                errors.append(f"Campos faltantes en línea {line_number}: {missing}.")
                continue
            document_id = record["document_id"]
            text_hash = record["text_sha256"]
            language = record["language"]
            if document_id in document_ids:
                errors.append(f"document_id duplicado: {document_id}.")
            document_ids.add(document_id)
            if text_hash in text_hashes:
                errors.append(f"Contenido duplicado: {text_hash}.")
            text_hashes.add(text_hash)
            computed_text_hash = hashlib.sha256(record["text"].encode("utf-8")).hexdigest()
            if text_hash != computed_text_hash:
                errors.append(f"text_sha256 incorrecto para {document_id}.")
            if language not in domains:
                errors.append(f"Idioma no permitido para {document_id}: {language}.")
            else:
                languages[language] += 1
                domains[language].update(record["domains"])
            if record["split"] not in {"train", "holdout"}:
                errors.append(f"Split inválido para {document_id}: {record['split']}.")
            else:
                splits[record["split"]] += 1
            source = declared_sources.get(record["source_id"])
            if source is None:
                errors.append(f"Fuente no declarada para {document_id}: {record['source_id']}.")
            elif record["source_url"] != source["url"] or record["license"] != source["license"]:
                errors.append(f"Procedencia inconsistente para {document_id}.")
            source_counts[record["source_id"]] += 1

    acceptance = manifest["acceptance"]
    if line_count != metadata.get("document_count"):
        errors.append(f"Conteo de líneas ({line_count}) no coincide con metadata ({metadata.get('document_count')}).")
    if sum(languages.values()) < acceptance["minimum_total_documents"]:
        errors.append("No se alcanza el mínimo total de documentos.")
    for language in acceptance["required_languages"]:
        if languages[language] < acceptance["minimum_documents_per_language"]:
            errors.append(f"No se alcanza el mínimo de documentos para {language}.")
        missing_domains = [domain for domain in acceptance["minimum_domain_coverage_per_language"] if domains[language][domain] == 0]
        if missing_domains:
            errors.append(f"Cobertura de dominio ausente para {language}: {missing_domains}.")
    if acceptance.get("require_holdout") and splits["holdout"] == 0:
        errors.append("No existe conjunto holdout.")
    observed_holdout_pct = (100 * splits["holdout"] / line_count) if line_count else 0
    expected_holdout_pct = manifest["splits"]["holdout_percent"]
    if line_count and abs(observed_holdout_pct - expected_holdout_pct) > 1.0:
        warnings.append(f"El holdout observado ({observed_holdout_pct:.2f}%) se aleja más de un punto porcentual del objetivo ({expected_holdout_pct}%).")

    return {
        "valid": not errors,
        "validated_utc": datetime.now(UTC).isoformat(),
        "offline_validation": True,
        "network_requests": 0,
        "dataset_id": manifest["dataset_id"],
        "documents_sha256": documents_hash,
        "document_count": line_count,
        "language_counts": dict(languages),
        "split_counts": dict(splits),
        "holdout_percent_observed": round(observed_holdout_pct, 4),
        "domain_counts": {language: dict(values) for language, values in domains.items()},
        "source_counts": dict(source_counts),
        "errors": errors,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", default="/home/ubuntu/aethel-knowledge-corpus-v1")
    parser.add_argument("--manifest", default="training/aethel_knowledge_corpus_v1.manifest.json")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir).resolve()
    report_path = Path(args.report).resolve() if args.report else dataset_dir / "validation_report.json"
    report = validate(dataset_dir, Path(args.manifest).resolve())
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
