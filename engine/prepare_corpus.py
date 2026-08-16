"""Prepara shards auditables desde fuentes abiertas declaradas en un manifiesto.

El script no activa por sí mismo ninguna fuente: cada entrada requiere enabled=true
y --allow-network. Así se evita descargar o entrenar sobre corpus masivos sin revisión.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable

EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\w)(?:\+?\d[\d .()\-]{7,}\d)(?!\w)")
REPEATED_RE = re.compile(r"(.)\1{24,}")


def normalize_text(value: object, remove_simple_pii: bool) -> str | None:
    if not isinstance(value, str):
        return None
    text = unicodedata.normalize("NFKC", value).replace("\x00", " ")
    text = " ".join(text.split())
    if remove_simple_pii:
        text = EMAIL_RE.sub("[EMAIL_REDACTED]", text)
        text = PHONE_RE.sub("[PHONE_REDACTED]", text)
    return text


def iter_source(source: dict, allow_network: bool) -> Iterable[dict]:
    if not allow_network:
        raise RuntimeError("La descarga requiere --allow-network y una revisión explícita del manifiesto")
    from datasets import load_dataset

    return load_dataset(
        source["dataset"],
        source.get("config"),
        split=source.get("split", "train"),
        streaming=bool(source.get("streaming", True)),
        trust_remote_code=False,
    )


def write_record(handle: gzip.GzipFile, text: str, source_id: str, digest: str) -> None:
    handle.write((json.dumps({"text": text, "source": source_id, "sha256": digest}, ensure_ascii=False) + "\n").encode("utf-8"))


def belongs_to_validation(digest: str, validation_percent: float, seed: int) -> bool:
    """Asigna cada documento a un split estable, independiente del orden de descarga."""
    if validation_percent <= 0:
        return False
    if validation_percent >= 1:
        return True
    bucket = int(hashlib.sha256(f"{seed}:{digest}".encode("utf-8")).hexdigest()[:16], 16)
    return bucket / float(0xFFFFFFFFFFFFFFFF) < validation_percent


def run(args: argparse.Namespace) -> None:
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    filters = manifest.get("filters", {})
    min_chars = int(filters.get("min_characters", 200))
    max_chars = int(filters.get("max_characters", 50000))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    counts = {"accepted": 0, "deduplicated": 0, "rejected": 0, "validation": 0, "sources": {}}
    shard_index = 0
    in_shard = 0
    handle: gzip.GzipFile | None = None
    shard_hashes: list[dict] = []

    def open_shard() -> gzip.GzipFile:
        nonlocal shard_index, in_shard
        if handle is not None:
            handle.close()
        path = output / f"train-{shard_index:05d}.jsonl.gz"
        shard_index += 1
        in_shard = 0
        shard_hashes.append({"path": path.name, "sha256": None})
        return gzip.open(path, "wb")

    handle = open_shard()
    validation_path = output / "validation.jsonl.gz"
    with gzip.open(validation_path, "wb") as validation:
        for source in manifest.get("sources", []):
            if not source.get("enabled"):
                continue
            source_id = source["id"]
            counts["sources"][source_id] = {"accepted": 0, "validation": 0, "deduplicated": 0, "rejected": 0}
            for row in iter_source(source, args.allow_network):
                text = normalize_text(row.get(source.get("text_column", "text")), bool(filters.get("remove_simple_pii", True)))
                if not text or len(text) < min_chars or len(text) > max_chars or REPEATED_RE.search(text):
                    counts["rejected"] += 1
                    counts["sources"][source_id]["rejected"] += 1
                    continue
                digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
                if filters.get("deduplicate_exact", True) and digest in seen:
                    counts["deduplicated"] += 1
                    counts["sources"][source_id]["deduplicated"] += 1
                    continue
                seen.add(digest)
                if belongs_to_validation(digest, float(filters.get("validation_percent", 0.005)), args.seed):
                    write_record(validation, text, source_id, digest)
                    counts["validation"] += 1
                    counts["sources"][source_id]["validation"] += 1
                else:
                    write_record(handle, text, source_id, digest)
                    in_shard += 1
                    counts["accepted"] += 1
                    counts["sources"][source_id]["accepted"] += 1
                    if in_shard >= args.shard_documents:
                        handle = open_shard()
                if args.max_documents and counts["accepted"] + counts["validation"] >= args.max_documents:
                    break
            if args.max_documents and counts["accepted"] + counts["validation"] >= args.max_documents:
                break
    handle.close()
    for item in shard_hashes:
        item["sha256"] = hashlib.sha256((output / item["path"]).read_bytes()).hexdigest()
    result = {
        "input_manifest": str(Path(args.manifest).resolve()),
        "input_manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "counts": counts,
        "shards": shard_hashes,
        "validation": validation_path.name,
        "validation_sha256": hashlib.sha256(validation_path.read_bytes()).hexdigest(),
        "filters": filters,
        "seed": args.seed,
    }
    (output / "prepared_manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-documents", type=int, default=0, help="0 significa sin límite; úselo solo tras aprobación de presupuesto.")
    parser.add_argument("--shard-documents", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--allow-network", action="store_true")
    run(parser.parse_args())
