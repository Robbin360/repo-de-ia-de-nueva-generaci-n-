"""Construye un corpus es-en trazable con mezcla por turnos y controles de calidad.

Las fuentes se activan únicamente mediante un manifiesto explícito y --allow-network.
El resultado contiene texto ya preparado, hashes y procedencia; no añade ejemplos
inventados ni usa conjuntos retenidos de evaluación.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import time
import unicodedata
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urlencode


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
REPEATED_RE = re.compile(r"(.)\1{24,}")


def normalize_text(value: object, remove_simple_pii: bool) -> str | None:
    if not isinstance(value, str):
        return None
    text = " ".join(unicodedata.normalize("NFKC", value).replace("\x00", " ").split())
    if remove_simple_pii:
        text = PHONE_RE.sub("[PHONE_REDACTED]", EMAIL_RE.sub("[EMAIL_REDACTED]", text))
    return text or None


def fingerprint(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def is_validation(digest: str, percent: float) -> bool:
    return int(digest[:16], 16) / float(0xFFFFFFFFFFFFFFFF) < percent


def hf_rows(source: dict) -> Iterator[dict]:
    from datasets import load_dataset

    yield from load_dataset(
        source["dataset"],
        source.get("config"),
        split=source.get("split", "train"),
        streaming=True,
        revision=source["revision"],
        trust_remote_code=False,
    )


def hf_rows_api(source: dict) -> Iterator[dict]:
    """Lee páginas pequeñas del servidor de datasets para no cargar un corpus completo en RAM."""
    offset = 0
    batch_size = int(source.get("batch_size", 100))
    endpoint = source.get("endpoint", "https://datasets-server.huggingface.co/rows")
    while True:
        query = urlencode({
            "dataset": source["dataset"],
            "config": source["config"],
            "split": source.get("split", "train"),
            "offset": offset,
            "length": batch_size,
            "revision": source["revision"],
        })
        request = urllib.request.Request(f"{endpoint}?{query}", headers={"User-Agent": "AethelNextGenDataPrep/1.0"})
        payload = None
        for attempt in range(6):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                if error.code not in (429, 500, 502, 503, 504) or attempt == 5:
                    raise
                retry_after = error.headers.get("Retry-After")
                delay = float(retry_after) if retry_after and retry_after.isdigit() else min(30.0, 2.0 ** attempt)
                print(f"Reintento de {source['id']} tras HTTP {error.code}: {delay:.1f}s", flush=True)
                time.sleep(delay)
        assert payload is not None
        rows = payload.get("rows", [])
        if not rows:
            return
        for item in rows:
            row = item.get("row")
            if isinstance(row, dict):
                yield row
        if len(rows) < batch_size:
            return
        offset += len(rows)
        time.sleep(float(source.get("request_delay_seconds", 0.75)))


def dictionary_text(row: dict, expected_language: str) -> str | None:
    if row.get("lang_code") != expected_language:
        return None
    word = normalize_text(row.get("word"), remove_simple_pii=False)
    if not word:
        return None
    glosses: list[str] = []
    examples: list[str] = []
    for sense in row.get("senses", []):
        if not isinstance(sense, dict):
            continue
        for gloss in sense.get("glosses", []):
            clean = normalize_text(gloss, remove_simple_pii=False)
            if clean:
                glosses.append(clean)
        for example in sense.get("examples", []):
            if isinstance(example, dict):
                clean = normalize_text(example.get("text"), remove_simple_pii=False)
                if clean:
                    examples.append(clean)
    if not glosses:
        return None
    pos = normalize_text(row.get("pos"), remove_simple_pii=False)
    label = "Entrada de diccionario" if expected_language == "es" else "Dictionary entry"
    definition = "; ".join(glosses[:3])
    result = f"{label}: {word}."
    if pos:
        result += f" {pos}."
    result += f" Definición: {definition}."
    if examples:
        result += f" Ejemplo: {examples[0]}."
    return result


def dictionary_rows(source: dict) -> Iterator[dict]:
    request = urllib.request.Request(source["url"], headers={"User-Agent": "AethelNextGenDataPrep/1.0"})
    with urllib.request.urlopen(request, timeout=90) as response:
        raw = gzip.GzipFile(fileobj=response) if source["url"].endswith(".gz") else response
        for raw_line in raw:
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            text = dictionary_text(row, source["language"])
            if text:
                yield {"text": text}


def source_rows(source: dict) -> Iterable[dict]:
    kind = source["kind"]
    if kind == "hf_text":
        return hf_rows(source)
    if kind == "hf_rows_api":
        return hf_rows_api(source)
    if kind == "wiktionary_jsonl":
        return dictionary_rows(source)
    raise ValueError(f"Tipo de fuente no admitido: {kind}")


def write_record(handle: gzip.GzipFile, text: str, source: dict, digest: str) -> None:
    handle.write(
        (json.dumps({"text": text, "source": source["id"], "language": source["language"], "sha256": digest}, ensure_ascii=False) + "\n").encode("utf-8")
    )


def run(args: argparse.Namespace) -> None:
    if not args.allow_network:
        raise RuntimeError("La preparación descargará datos reales: confirme con --allow-network.")
    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("approval_required") and not getattr(args, "approved_data_plan", False):
        raise RuntimeError("El manifiesto exige aprobación explícita (--approved-data-plan) antes de descargar fuentes.")
    sources = [item for item in manifest.get("sources", []) if item.get("enabled")]
    if not sources:
        raise RuntimeError("El manifiesto no tiene fuentes habilitadas.")
    for source in sources:
        for key in ("id", "kind", "language", "document_limit", "license", "provenance_url"):
            if not source.get(key):
                raise RuntimeError(f"La fuente no declara {key}: {source}")
        if manifest.get("approval_required") and source.get("approved") is not True:
            raise RuntimeError(f"La fuente no tiene aprobación explícita: {source['id']}")
        if source["kind"] == "hf_text" and not source.get("revision"):
            raise RuntimeError(f"La fuente HF debe fijar una revisión: {source['id']}")

    filters = manifest.get("filters", {})
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    validation_percent = float(filters.get("validation_percent", 0.005))
    min_default = int(filters.get("min_characters", 200))
    max_chars = int(filters.get("max_characters", 50_000))
    seen: set[str] = set()
    source_counts: dict[str, dict[str, int]] = {source["id"]: defaultdict(int) for source in sources}
    source_hashes = {source["id"]: hashlib.sha256() for source in sources}
    language_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    streams = {source["id"]: iter(source_rows(source)) for source in sources}
    active = list(sources)
    shard_number, records_in_shard = 0, 0
    shards: list[dict[str, str]] = []
    handle: gzip.GzipFile | None = None

    def open_shard() -> gzip.GzipFile:
        nonlocal shard_number, records_in_shard, handle
        if handle is not None:
            handle.close()
        path = output / f"train-{shard_number:05d}.jsonl.gz"
        shard_number += 1
        records_in_shard = 0
        shards.append({"path": path.name, "sha256": ""})
        return gzip.open(path, "wb")

    handle = open_shard()
    validation_path = output / "validation.jsonl.gz"
    with gzip.open(validation_path, "wb") as validation:
        while active:
            next_active: list[dict] = []
            for source in active:
                stats = source_counts[source["id"]]
                if stats["accepted"] + stats["validation"] >= int(source["document_limit"]):
                    continue
                try:
                    row = next(streams[source["id"]])
                except StopIteration:
                    stats["exhausted"] += 1
                    continue
                next_active.append(source)
                text = normalize_text(row.get(source.get("text_column", "text")), bool(filters.get("remove_simple_pii", True)))
                minimum = int(source.get("min_characters", min_default))
                if not text or len(text) < minimum or len(text) > max_chars or REPEATED_RE.search(text):
                    stats["rejected"] += 1
                    continue
                digest = fingerprint(text, args.seed)
                if filters.get("deduplicate_exact", True) and digest in seen:
                    stats["deduplicated"] += 1
                    continue
                seen.add(digest)
                if is_validation(digest, validation_percent):
                    write_record(validation, text, source, digest)
                    stats["validation"] += 1
                    language_counts[source["language"]]["validation"] += 1
                    source_hashes[source["id"]].update(digest.encode("ascii"))
                else:
                    write_record(handle, text, source, digest)
                    stats["accepted"] += 1
                    language_counts[source["language"]]["accepted"] += 1
                    source_hashes[source["id"]].update(digest.encode("ascii"))
                    records_in_shard += 1
                    if records_in_shard >= args.shard_documents:
                        handle = open_shard()
            active = next_active
    handle.close()
    for item in shards:
        path = output / item["path"]
        with gzip.open(path, "rt", encoding="utf-8") as candidate:
            has_records = bool(candidate.read(1))
        if not has_records:
            path.unlink()
            continue
        item["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    shards = [item for item in shards if item["sha256"]]
    required = manifest.get("minimum_documents_by_language", {})
    for language, minimum in required.items():
        total = language_counts[language]["accepted"] + language_counts[language]["validation"]
        if total < int(minimum):
            raise RuntimeError(f"Datos insuficientes para {language}: {total} < {minimum}")
    resolved_sources = []
    for source in sources:
        total = source_counts[source["id"]]["accepted"] + source_counts[source["id"]]["validation"]
        if total < int(source.get("minimum_documents", 1)):
            raise RuntimeError(f"Datos insuficientes para {source['id']}: {total} < {source.get('minimum_documents', 1)}")
        resolved = dict(source)
        if source["kind"] == "wiktionary_jsonl":
            resolved["revision"] = source_hashes[source["id"]].hexdigest()
            resolved["revision_kind"] = "sha256-del-contenido-filtrado"
        resolved_sources.append(resolved)
    result = {
        "schema_version": 1,
        "purpose": manifest.get("purpose"),
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "sources": resolved_sources,
        "source_counts": source_counts,
        "language_counts": language_counts,
        "filters": filters,
        "seed": args.seed,
        "shards": shards,
        "validation": {"path": validation_path.name, "sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest()},
    }
    (output / "prepared_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, default=dict), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, default=dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-documents", type=int, default=25_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--approved-data-plan", action="store_true")
    run(parser.parse_args())
