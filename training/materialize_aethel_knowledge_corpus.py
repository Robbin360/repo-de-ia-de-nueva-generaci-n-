#!/usr/bin/env python3
"""Materializa un corpus Wikimedia real, bilingüe y trazable sin usar GPU.

Cada documento publicado proviene literalmente de un dump oficial. La
clasificación de dominios sólo añade metadatos para auditar cobertura; no crea,
traduce ni reescribe contenido. El resultado es un JSONL apto para empaquetar
como Dataset de Kaggle separado del código y del entrenamiento.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as element_tree
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable


USER_AGENT = "AethelKnowledgeCorpus/1.0 (traceable research materializer)"
WIKI_NAMESPACE = "{http://www.mediawiki.org/xml/export-0.11/}"
SPACE = re.compile(r"\s+")
TEMPLATE = re.compile(r"\{\{[^{}]{0,400}\}\}")
REF = re.compile(r"<ref[^>/]*?(?:/>|>.*?</ref>)", re.IGNORECASE | re.DOTALL)
TAG = re.compile(r"<[^>]{1,200}>")
LINK = re.compile(r"\[\[([^\]|]{1,200})(?:\|([^\]]{1,300}))?\]\]")
EXTERNAL_LINK = re.compile(r"\[https?://[^\s\]]+(?:\s+([^\]]{1,300}))?\]")
HEADER = re.compile(r"={2,6}\s*([^=]{1,200}?)\s*={2,6}")

DOMAIN_KEYWORDS = {
    "en": {
        "mathematics": ("mathematics", "algebra", "geometry", "calculus", "theorem", "equation", "probability", "statistics"),
        "science": ("physics", "chemistry", "biology", "astronomy", "geology", "science"),
        "engineering": ("engineering", "mechanical", "electrical", "civil engineering", "materials", "manufacturing"),
        "programming": ("programming", "software", "computer science", "algorithm", "computer", "python", "javascript"),
        "language": ("language", "linguistics", "grammar", "literature", "writing", "translation"),
    },
    "es": {
        "mathematics": ("matemática", "álgebra", "geometría", "cálculo", "teorema", "ecuación", "probabilidad", "estadística"),
        "science": ("física", "química", "biología", "astronomía", "geología", "ciencia"),
        "engineering": ("ingeniería", "mecánica", "eléctrica", "ingeniería civil", "materiales", "manufactura"),
        "programming": ("programación", "software", "informática", "algoritmo", "computadora", "ordenador", "python", "javascript"),
        "language": ("idioma", "lengua", "lingüística", "gramática", "literatura", "escritura", "traducción"),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_split(identifier: str, seed: str, train_percent: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()[:8], 16) % 100
    return "train" if bucket < train_percent else "holdout"


def clean_wikitext(raw: str) -> str:
    """Limpia marcado para lectura, conservando únicamente texto del documento fuente."""
    value = raw.replace("\r", "\n")
    previous = ""
    while value != previous:
        previous = value
        value = TEMPLATE.sub(" ", value)
    value = REF.sub(" ", value)
    value = TAG.sub(" ", value)
    value = EXTERNAL_LINK.sub(lambda match: match.group(1) or " ", value)
    value = LINK.sub(lambda match: match.group(2) or match.group(1), value)
    value = HEADER.sub(lambda match: f"\n{match.group(1)}\n", value)
    value = value.replace("'''", "").replace("''", "")
    value = re.sub(r"^\s*[*#;:]\s*", "", value, flags=re.MULTILINE)
    return SPACE.sub(" ", value).strip()


def domains_for(title: str, text: str, language: str) -> list[str]:
    haystack = f"{title} {text[:6000]}".lower()
    selected = ["general"]
    for domain, keywords in DOMAIN_KEYWORDS[language].items():
        if any(keyword in haystack for keyword in keywords):
            selected.append(domain)
    return sorted(set(selected))


def download_resumable(url: str, destination: Path, attempts: int = 6) -> dict[str, Any]:
    """Descarga un artefacto oficial con Range y conserva el .part ante una interrupción."""
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(attempts):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": USER_AGENT}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                status = getattr(response, "status", 200)
                if start and status != 206:
                    partial.unlink(missing_ok=True)
                    start = 0
                mode = "ab" if start else "wb"
                with partial.open(mode) as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
            partial.replace(destination)
            return {
                "url": url,
                "downloaded_utc": datetime.now(UTC).isoformat(),
                "bytes": destination.stat().st_size,
                "sha256": sha256_file(destination),
            }
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as error:
            if attempt == attempts - 1:
                raise RuntimeError(f"No se pudo descargar {url}: {error}") from error
            time.sleep(min(60, 2 ** (attempt + 2)))
    raise AssertionError("unreachable")


def iter_wiki_pages(source_path: Path) -> Iterable[tuple[str, str, str]]:
    """Emite título, revision-id y texto desde un XML bz2 oficial de Wikimedia."""
    with bz2.open(source_path, "rb") as handle:
        for _event, node in element_tree.iterparse(handle, events=("end",)):
            if node.tag != f"{WIKI_NAMESPACE}page":
                continue
            title = node.findtext(f"{WIKI_NAMESPACE}title") or ""
            redirect = node.find(f"{WIKI_NAMESPACE}redirect")
            text_node = node.find(f".//{WIKI_NAMESPACE}text")
            revision_id = node.findtext(f".//{WIKI_NAMESPACE}revision/{WIKI_NAMESPACE}id") or ""
            raw = text_node.text if text_node is not None and text_node.text else ""
            node.clear()
            if title and raw and redirect is None:
                yield title, revision_id, raw


def materialize_source(
    source: dict[str, Any],
    raw_dir: Path,
    output_handle: Any,
    seed: str,
    train_percent: int,
) -> tuple[dict[str, Any], Counter[str]]:
    """Escribe documentos reales de una fuente y devuelve su inventario trazable."""
    raw_name = source["url"].rsplit("/", 1)[-1]
    raw_path = raw_dir / raw_name
    inventory = download_resumable(source["url"], raw_path)
    inventory.update({"id": source["id"], "language": source["language"], "license": source["license"], "path": raw_name})

    counts: Counter[str] = Counter()
    seen_text_hashes: set[str] = set()
    target = int(source["target_documents"])
    minimum = int(source["min_characters"])
    maximum = int(source["max_characters"])
    for title, revision_id, raw in iter_wiki_pages(raw_path):
        text = clean_wikitext(raw)
        if len(text) < minimum:
            continue
        text = text[:maximum].strip()
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if text_hash in seen_text_hashes:
            continue
        seen_text_hashes.add(text_hash)
        document_id = hashlib.sha256(f"{source['id']}:{revision_id}:{title}".encode("utf-8")).hexdigest()
        domains = domains_for(title, text, source["language"])
        record = {
            "document_id": document_id,
            "split": stable_split(document_id, seed, train_percent),
            "language": source["language"],
            "source_id": source["id"],
            "source_url": source["url"],
            "license": source["license"],
            "title": title,
            "revision_id": revision_id,
            "domains": domains,
            "text_sha256": text_hash,
            "text": text,
        }
        output_handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        counts["documents"] += 1
        counts["characters"] += len(text)
        for domain in domains:
            counts[f"domain:{domain}"] += 1
        if counts["documents"] >= target:
            break
    return inventory, counts


def run(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    output_root = Path(args.output_dir).resolve()
    raw_dir = output_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    documents_path = output_root / "documents.jsonl"
    metadata_path = output_root / "metadata.json"
    if documents_path.exists() and not args.force:
        raise RuntimeError(f"Ya existe {documents_path}; usa --force para reconstruir de forma explícita.")

    sources: list[dict[str, Any]] = manifest["sources"]
    acceptance = manifest["acceptance"]
    splits = manifest["splits"]
    inventories: list[dict[str, Any]] = []
    language_counts: Counter[str] = Counter()
    domain_counts: dict[str, Counter[str]] = {language: Counter() for language in acceptance["required_languages"]}
    with documents_path.open("w", encoding="utf-8") as output_handle:
        for source in sources:
            inventory, counts = materialize_source(source, raw_dir, output_handle, splits["seed"], int(splits["train_percent"]))
            inventories.append(inventory)
            language_counts[source["language"]] += counts["documents"]
            domain_counts[source["language"]].update({key.removeprefix("domain:"): value for key, value in counts.items() if key.startswith("domain:")})

    failures: list[str] = []
    minimum = int(acceptance["minimum_documents_per_language"])
    for language in acceptance["required_languages"]:
        if language_counts[language] < minimum:
            failures.append(f"{language}: {language_counts[language]} < {minimum}")
        missing_domains = [domain for domain in acceptance["minimum_domain_coverage_per_language"] if domain_counts[language][domain] == 0]
        if missing_domains:
            failures.append(f"{language}: dominios sin cobertura {missing_domains}")
    total = sum(language_counts.values())
    if total < int(acceptance["minimum_total_documents"]):
        failures.append(f"total: {total} < {acceptance['minimum_total_documents']}")
    if failures:
        documents_path.unlink(missing_ok=True)
        raise RuntimeError("Puerta de calidad del Dataset no superada: " + "; ".join(failures))

    metadata = {
        "dataset_id": manifest["dataset_id"],
        "materialized_utc": datetime.now(UTC).isoformat(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "documents_path": documents_path.name,
        "documents_sha256": sha256_file(documents_path),
        "document_count": total,
        "language_counts": dict(language_counts),
        "domain_counts": {language: dict(counts) for language, counts in domain_counts.items()},
        "sources": inventories,
        "splits": splits,
        "acceptance": acceptance,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": manifest["dataset_id"], "documents": total, "languages": dict(language_counts), "metadata": str(metadata_path)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="training/aethel_knowledge_corpus_v1.manifest.json")
    parser.add_argument("--output-dir", default="/home/ubuntu/aethel-knowledge-corpus-v1")
    parser.add_argument("--force", action="store_true")
    run(parser.parse_args())
