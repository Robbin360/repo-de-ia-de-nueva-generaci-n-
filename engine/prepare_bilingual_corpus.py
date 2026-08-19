"""Construye un corpus es-en trazable con mezcla por turnos y controles de calidad.

Las fuentes se activan únicamente mediante un manifiesto explícito y --allow-network.
El resultado contiene texto ya preparado, hashes y procedencia; no añade ejemplos
inventados ni usa conjuntos retenidos de evaluación.
"""
from __future__ import annotations

import argparse
import bz2
import gzip
import hashlib
import json
import re
import time
import unicodedata
import urllib.error
import urllib.request
import xml.etree.ElementTree as etree
from email.utils import parsedate_to_datetime
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


RETRYABLE_HTTP_CODES = frozenset({429, 500, 502, 503, 504})


def retry_delay(error: urllib.error.HTTPError, attempt: int, source: dict) -> float:
    """Calcula un backoff determinista y respeta Retry-After cuando es válido."""
    retry_after = error.headers.get("Retry-After") if error.headers else None
    if retry_after:
        try:
            return max(0.0, min(float(retry_after), float(source.get("max_retry_delay_seconds", 300))))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after).timestamp()
                return max(0.0, min(retry_at - time.time(), float(source.get("max_retry_delay_seconds", 300))))
            except (TypeError, ValueError, OverflowError):
                pass
    base = float(source.get("retry_backoff_seconds", 2.0))
    ceiling = float(source.get("max_retry_delay_seconds", 300))
    return min(ceiling, base * (2 ** attempt))


def cached_json_page(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) and isinstance(payload.get("rows"), list) else None


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def hf_rows_api(source: dict, cache_dir: Path | None = None) -> Iterator[dict]:
    """Lee páginas pequeñas y las conserva para reanudar tras un corte o un 429."""
    offset = 0
    batch_size = int(source.get("batch_size", 100))
    endpoint = source.get("endpoint", "https://datasets-server.huggingface.co/rows")
    page_dir = (cache_dir or Path(".aethel-source-cache")) / source["id"]
    while True:
        page_path = page_dir / f"page-{offset:012d}-{batch_size}.json"
        payload = cached_json_page(page_path)
        if payload is None:
            query = urlencode({
                "dataset": source["dataset"],
                "config": source["config"],
                "split": source.get("split", "train"),
                "offset": offset,
                "length": batch_size,
                "revision": source["revision"],
            })
            request = urllib.request.Request(
                f"{endpoint}?{query}",
                headers={"User-Agent": "AethelNextGenDataPrep/1.1"},
            )
            for attempt in range(int(source.get("max_retries", 8))):
                try:
                    with urllib.request.urlopen(request, timeout=90) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
                        raise RuntimeError(f"Respuesta inválida de {source['id']} en offset {offset}")
                    atomic_write_json(page_path, payload)
                    break
                except urllib.error.HTTPError as error:
                    if error.code not in RETRYABLE_HTTP_CODES or attempt + 1 >= int(source.get("max_retries", 8)):
                        raise
                    delay = retry_delay(error, attempt, source)
                    print(f"Reintento de {source['id']} tras HTTP {error.code}: {delay:.1f}s", flush=True)
                    time.sleep(delay)
            else:
                raise RuntimeError(f"Agotados los reintentos para {source['id']} en offset {offset}")
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


def download_resumable(source: dict, cache_dir: Path) -> Path:
    """Descarga a .part y reanuda con Range; sólo publica el archivo al completar."""
    destination = cache_dir / source["id"] / Path(source["url"]).name
    partial = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return destination
    for attempt in range(int(source.get("max_retries", 8))):
        start = partial.stat().st_size if partial.exists() else 0
        headers = {"User-Agent": "AethelNextGenDataPrep/1.1"}
        if start:
            headers["Range"] = f"bytes={start}-"
        request = urllib.request.Request(source["url"], headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                append = start > 0 and response.headers.get("Content-Range")
                if start and not append:
                    partial.unlink(missing_ok=True)
                    start = 0
                mode = "ab" if append else "wb"
                with partial.open(mode) as handle:
                    while chunk := response.read(1024 * 1024):
                        handle.write(chunk)
                partial.replace(destination)
                return destination
        except urllib.error.HTTPError as error:
            if error.code not in RETRYABLE_HTTP_CODES or attempt + 1 >= int(source.get("max_retries", 8)):
                raise
            delay = retry_delay(error, attempt, source)
            print(f"Reintento de descarga {source['id']} tras HTTP {error.code}: {delay:.1f}s", flush=True)
            time.sleep(delay)
    raise RuntimeError(f"Agotados los reintentos de descarga para {source['id']}")


def dictionary_rows(source: dict, cache_dir: Path | None = None) -> Iterator[dict]:
    if cache_dir is None:
        cache_dir = Path(".aethel-source-cache")
    local_path = download_resumable(source, cache_dir)
    with local_path.open("rb") as stream:
        raw = gzip.GzipFile(fileobj=stream) if source["url"].endswith(".gz") else stream
        for raw_line in raw:
            try:
                row = json.loads(raw_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue
            text = dictionary_text(row, source["language"])
            if text:
                yield {"text": text}


WIKITEXT_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}", re.DOTALL)
WIKITEXT_REF_RE = re.compile(r"<ref\b[^>/]*?(?:/>|>.*?</ref\s*>)", re.IGNORECASE | re.DOTALL)
WIKITEXT_TAG_RE = re.compile(r"<[^>]+>")
WIKITEXT_LINK_RE = re.compile(r"\[\[([^\]|]+)\|?([^\]]*)\]\]")


def plain_wikitext(value: str) -> str:
    """Reduce marcado de MediaWiki de forma conservadora, sin crear contenido."""
    text = WIKITEXT_REF_RE.sub(" ", value)
    previous = None
    while previous != text:
        previous = text
        text = WIKITEXT_TEMPLATE_RE.sub(" ", text)
    text = WIKITEXT_LINK_RE.sub(lambda match: match.group(2) or match.group(1), text)
    text = text.replace("'''", "").replace("''", "")
    return " ".join(WIKITEXT_TAG_RE.sub(" ", text).split())


def wikimedia_dump_rows(source: dict, cache_dir: Path | None = None) -> Iterator[dict]:
    """Extrae páginas de espacio principal desde un fragmento XML oficial de Wikimedia."""
    if cache_dir is None:
        cache_dir = Path(".aethel-source-cache")
    local_path = download_resumable(source, cache_dir)
    opener = bz2.open if local_path.suffix == ".bz2" else local_path.open
    with opener(local_path, "rb") as stream:
        for _, element in etree.iterparse(stream, events=("end",)):
            if element.tag.rsplit("}", 1)[-1] != "page":
                continue
            fields = {child.tag.rsplit("}", 1)[-1]: child for child in element}
            namespace = (fields.get("ns").text or "").strip() if fields.get("ns") is not None else ""
            revision = fields.get("revision")
            text_node = None
            if revision is not None:
                for child in revision:
                    if child.tag.rsplit("}", 1)[-1] == "text":
                        text_node = child
                        break
            raw = text_node.text if text_node is not None else None
            if namespace == "0" and "redirect" not in fields and raw:
                yield {"text": plain_wikitext(raw)}
            element.clear()


def source_rows(source: dict, cache_dir: Path | None = None) -> Iterable[dict]:
    kind = source["kind"]
    if kind == "hf_text":
        return hf_rows(source)
    if kind == "hf_rows_api":
        return hf_rows_api(source, cache_dir)
    if kind == "wiktionary_jsonl":
        return dictionary_rows(source, cache_dir)
    if kind == "wikimedia_xml_dump":
        return wikimedia_dump_rows(source, cache_dir)
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
    cache_dir = output / ".source_cache"
    streams: dict[str, Iterator[dict]] = {}
    required_sources = [source for source in sources if source.get("required", True)]
    optional_sources = [source for source in sources if not source.get("required", True)]
    active = list(required_sources)
    optional_started = False
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

    def stream_for(source: dict) -> Iterator[dict]:
        source_id = source["id"]
        if source_id not in streams:
            streams[source_id] = iter(source_rows(source, cache_dir))
        return streams[source_id]

    with gzip.open(validation_path, "wb") as validation:
        while active:
            next_active: list[dict] = []
            for source in active:
                stats = source_counts[source["id"]]
                if stats["accepted"] + stats["validation"] >= int(source["document_limit"]):
                    continue
                try:
                    row = next(stream_for(source))
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
            if next_active:
                active = next_active
                continue
            if optional_started:
                active = []
                continue
            required_by_language = manifest.get("minimum_documents_by_language", {})
            missing_languages = {
                language
                for language, minimum in required_by_language.items()
                if language_counts[language]["accepted"] + language_counts[language]["validation"] < int(minimum)
            }
            active = [source for source in optional_sources if source["language"] in missing_languages]
            optional_started = True
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
        if source.get("required", True) and total < int(source.get("minimum_documents", 1)):
            raise RuntimeError(f"Datos insuficientes para {source['id']}: {total} < {source.get('minimum_documents', 1)}")
        resolved = dict(source)
        if source["kind"] in {"wiktionary_jsonl", "wikimedia_xml_dump"}:
            if source.get("revision"):
                resolved["declared_revision"] = source["revision"]
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
