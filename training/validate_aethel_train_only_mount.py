"""Valida sólo los shards de entrenamiento para el preflight D1A.

No importa PyTorch, no consulta CUDA, no abre checkpoints ni evalúa filas JSONL.
Los shards holdout pueden coexistir en el montaje privado, pero este validador no
los abre ni los incluye en hashes, conteos o reportes.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Literal


PLAINTEXT_CONTRACT_FILENAME = "aethel_kaggle_decompressed_mount_contract.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plaintext_path(compressed_path: str) -> str:
    if not compressed_path.endswith(".jsonl.gz"):
        raise ValueError(f"Shard comprimido inválido en manifiesto: {compressed_path}")
    return compressed_path.removesuffix(".gz")


def train_entries(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    corpus_files = manifest.get("corpus_files")
    if not isinstance(corpus_files, list):
        raise ValueError("package_manifest.json no contiene corpus_files como lista.")
    entries: list[dict[str, Any]] = []
    for entry in corpus_files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            raise ValueError("Entrada inválida en corpus_files.")
        name = Path(entry["path"]).name
        if name.startswith("train-"):
            entries.append(entry)
    if not entries:
        raise ValueError("No hay shards train-*.jsonl.gz en el manifiesto.")
    return entries


def load_plaintext_contract(manifest_path: Path, contract_path: Path) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return None, [f"No se puede leer el contrato de montaje plaintext: {error}"]
    if contract.get("schema") != "aethel-kaggle-decompressed-mount/v1":
        errors.append("Esquema inválido del contrato de montaje plaintext.")
    if contract.get("package_manifest_sha256") != sha256_file(manifest_path):
        errors.append("El contrato plaintext no está ligado al package_manifest.json montado.")
    details = contract.get("plaintext_shards")
    if not isinstance(details, dict):
        errors.append("plaintext_shards no es un objeto en el contrato plaintext.")
    return (contract if not errors else None), errors


def detect_train_mount_format(package_dir: Path, entries: list[dict[str, Any]]) -> tuple[Literal["gzip", "plaintext", "invalid"], list[str]]:
    compressed = [entry["path"] for entry in entries]
    plaintext = [plaintext_path(path) for path in compressed]
    compressed_present = [path for path in compressed if (package_dir / path).is_file()]
    plaintext_present = [path for path in plaintext if (package_dir / path).is_file()]
    if len(compressed_present) == len(compressed) and not plaintext_present:
        return "gzip", []
    if len(plaintext_present) == len(plaintext) and not compressed_present:
        return "plaintext", []
    return "invalid", [
        "Los shards train están incompletos o mezclan formatos .jsonl.gz/.jsonl.",
        f"Shards train comprimidos presentes: {len(compressed_present)}/{len(compressed)}.",
        f"Shards train plaintext presentes: {len(plaintext_present)}/{len(plaintext)}.",
    ]


def validate_train_only(package_dir: Path, plaintext_contract_path: Path | None = None) -> dict[str, Any]:
    """Valida integridad de bytes y tokenizador sin abrir texto de training u holdout."""
    package_dir = package_dir.resolve()
    manifest_path = package_dir / "package_manifest.json"
    errors: list[str] = []
    if not manifest_path.is_file():
        return {
            "valid": False,
            "errors": ["Falta package_manifest.json."],
            "network_requests": 0,
            "holdout_content_read": False,
            "raw_train_text_parsed": False,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = train_entries(manifest)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        return {
            "valid": False,
            "errors": [str(error)],
            "network_requests": 0,
            "holdout_content_read": False,
            "raw_train_text_parsed": False,
        }

    mount_format, mount_errors = detect_train_mount_format(package_dir, entries)
    errors.extend(mount_errors)
    plaintext_contract: dict[str, Any] | None = None
    if mount_format == "plaintext":
        contract_path = plaintext_contract_path or Path(__file__).with_name(PLAINTEXT_CONTRACT_FILENAME)
        plaintext_contract, contract_errors = load_plaintext_contract(manifest_path, contract_path)
        errors.extend(contract_errors)

    verified: list[str] = []
    for entry in entries:
        compressed_path = entry["path"]
        candidate = package_dir / (compressed_path if mount_format == "gzip" else plaintext_path(compressed_path))
        if mount_format == "invalid" or not candidate.is_file():
            errors.append(f"Falta shard train: {compressed_path}.")
            continue
        if mount_format == "gzip":
            expected_bytes = entry.get("bytes")
            expected_sha256 = entry.get("sha256")
        else:
            assert plaintext_contract is not None
            details = plaintext_contract.get("plaintext_shards", {}).get(plaintext_path(compressed_path))
            if not isinstance(details, dict):
                errors.append(f"Falta contrato plaintext para shard train: {compressed_path}.")
                continue
            expected_bytes = details.get("bytes")
            expected_sha256 = details.get("sha256")
        if not isinstance(expected_bytes, int) or not isinstance(expected_sha256, str):
            errors.append(f"Metadatos de integridad inválidos para shard train: {compressed_path}.")
            continue
        if candidate.stat().st_size != expected_bytes:
            errors.append(f"Tamaño inválido de shard train: {candidate.relative_to(package_dir)}.")
            continue
        if sha256_file(candidate) != expected_sha256:
            errors.append(f"Hash inválido de shard train: {candidate.relative_to(package_dir)}.")
            continue
        verified.append(candidate.relative_to(package_dir).as_posix())

    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict):
        errors.append("Falta metadato de tokenizador versionado.")
    else:
        tokenizer_path = package_dir / str(tokenizer.get("path", ""))
        expected_tokenizer_hash = tokenizer.get("sha256")
        if not tokenizer_path.is_file() or not isinstance(expected_tokenizer_hash, str):
            errors.append("Tokenizador ausente o sin hash verificable.")
        elif sha256_file(tokenizer_path) != expected_tokenizer_hash:
            errors.append("Hash de tokenizador inválido.")
        elif tokenizer.get("derived_from") != "train split only":
            errors.append("La procedencia del tokenizador no declara train split only.")

    counts = manifest.get("counts")
    train_counts = {key: value for key, value in counts.items() if isinstance(key, str) and key.startswith("train:")} if isinstance(counts, dict) else {}
    if set(train_counts) != {"train:en", "train:es"} or not all(isinstance(value, int) and value > 0 for value in train_counts.values()):
        errors.append("Conteos train:en/train:es inválidos en package_manifest.json.")
    return {
        "schema": "aethel-d1a-train-only-preflight/v1",
        "valid": not errors,
        "dataset_id": manifest.get("dataset_id"),
        "package_manifest_sha256": sha256_file(manifest_path),
        "tokenizer_sha256": tokenizer.get("sha256") if isinstance(tokenizer, dict) else None,
        "mount_format": mount_format,
        "train_shards_verified": verified,
        "train_counts_declared": train_counts,
        "errors": errors,
        "network_requests": 0,
        "gpu_used": False,
        "checkpoint_loaded": False,
        "holdout_content_read": False,
        "raw_train_text_parsed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--plaintext-contract", type=Path, default=None)
    args = parser.parse_args()
    report = validate_train_only(args.package_dir, args.plaintext_contract)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
