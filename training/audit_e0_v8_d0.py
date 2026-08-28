#!/usr/bin/env python3
"""Auditor D0 de solo lectura para la evidencia estática de Seed E0 V8.

No importa PyTorch, no detecta/usa CUDA, no abre checkpoints y no recorre corpus
ni shards. Sólo enlaza: marcador exacto de fuente, manifiesto raíz del Dataset y
evidencia auditada versionada en el bundle de código.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_SOURCE_RELEASE = "d0-v1-e0-v8-static-audit"
E0_RELEASE = "e0-v8-canonical-cuda-device-check"
EVIDENCE_SCHEMA = "aethel-e0-v8-d0-evidence/v1"


class D0AuditError(RuntimeError):
    """Señala una ruptura de contrato sin iniciar GPU ni leer pesos."""


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise D0AuditError(f"JSON D0 no legible: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise D0AuditError(f"JSON D0 debe ser objeto: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_evidence(evidence: dict[str, Any]) -> None:
    if evidence.get("schema") != EVIDENCE_SCHEMA:
        raise D0AuditError("Schema de evidencia D0 no reconocido")
    if evidence.get("e0_release") != E0_RELEASE:
        raise D0AuditError("La evidencia no corresponde a E0 V8")

    training = evidence.get("training")
    checkpoint = evidence.get("checkpoint")
    dataset_contract = evidence.get("dataset_contract")
    holdout = evidence.get("holdout")
    router = evidence.get("router")
    limits = evidence.get("d0_limits")
    if not all(
        isinstance(section, dict)
        for section in (training, checkpoint, dataset_contract, holdout, router, limits)
    ):
        raise D0AuditError("Faltan secciones requeridas de evidencia D0")
    if training.get("completed_steps") != 4992 or checkpoint.get("final_step") != 4992:
        raise D0AuditError("La evidencia no acredita el paso final E0 V8")
    if checkpoint.get("latest_name") != "latest.pt" or checkpoint.get("tensor_count") != 150:
        raise D0AuditError("Contrato estructural del checkpoint V8 inesperado")
    if holdout.get("segments_per_language") != 256:
        raise D0AuditError("Cobertura holdout esperada no coincide")
    if router.get("final_healthy") is not False:
        raise D0AuditError("D0 no permite ocultar router final no saludable")
    expected_counts = {
        "holdout:en": 989,
        "holdout:es": 988,
        "train:en": 19011,
        "train:es": 19012,
    }
    if dataset_contract.get("dataset_id") != "aethel-knowledge-reasoning-bilingual-v1":
        raise D0AuditError("Contrato D0 apunta a Dataset distinto")
    if dataset_contract.get("counts") != expected_counts:
        raise D0AuditError("Conteos congelados D0 no coinciden")
    if dataset_contract.get("tokenizer_sha256") != "4a3608e4e45c9117415d1f4fa236aebe20771dc3a3ce85760d9fb9d218fa0815":
        raise D0AuditError("Hash de tokenizer D0 no coincide")
    if dataset_contract.get("holdout_excluded_from_tokenizer") is not True:
        raise D0AuditError("D0 requiere holdout excluido del tokenizer")
    manifest_hash = dataset_contract.get("package_manifest_sha256")
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise D0AuditError("Hash de manifiesto D0 no verificable")
    required_limits = {
        "gpu_used": False,
        "checkpoint_loaded": False,
        "raw_corpus_read": False,
        "holdout_content_read": False,
        "network_requests": 0,
        "promotion_authorized": False,
    }
    if {key: limits.get(key) for key in required_limits} != required_limits:
        raise D0AuditError("Límites de solo lectura D0 alterados")


def validate_package_manifest(
    package_manifest: dict[str, Any], expected_contract: dict[str, Any], actual_sha256: str
) -> None:
    if actual_sha256 != expected_contract["package_manifest_sha256"]:
        raise D0AuditError("Hash del manifiesto montado no coincide con el contrato congelado")
    if package_manifest.get("dataset_id") != expected_contract["dataset_id"]:
        raise D0AuditError("Dataset ID del manifiesto no coincide")
    if package_manifest.get("counts") != expected_contract["counts"]:
        raise D0AuditError("Conteos del manifiesto no coinciden")
    tokenizer = package_manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or tokenizer.get("sha256") != expected_contract["tokenizer_sha256"]:
        raise D0AuditError("Hash de tokenizer del manifiesto no coincide")
    if package_manifest.get("holdout_excluded_from_tokenizer") is not True:
        raise D0AuditError("El manifiesto no acredita holdout fuera del tokenizer")


def audit(source_root: Path, data_root: Path, output_dir: Path) -> dict[str, Any]:
    release_path = source_root / "training" / "aethel_kaggle_source_release.json"
    evidence_path = source_root / "training" / "e0_v8_d0_evidence.json"
    package_manifest_path = data_root / "package_manifest.json"
    if not package_manifest_path.is_file():
        raise D0AuditError(f"No existe manifiesto raíz del Dataset: {package_manifest_path}")

    release = read_json(release_path)
    if release.get("release") != EXPECTED_SOURCE_RELEASE:
        raise D0AuditError(
            f"Release D0 incorrecto: {release.get('release')!r}; se requiere {EXPECTED_SOURCE_RELEASE!r}"
        )
    evidence = read_json(evidence_path)
    validate_evidence(evidence)
    package_manifest = read_json(package_manifest_path)
    package_manifest_sha256 = sha256_file(package_manifest_path)
    dataset_contract = evidence["dataset_contract"]
    validate_package_manifest(package_manifest, dataset_contract, package_manifest_sha256)

    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": "aethel-e0-d0-audit-report/v1",
        "status": "D0_AUDIT_READY",
        "source_release": release["release"],
        "e0_evidence_release": evidence["e0_release"],
        "source_release_sha256": sha256_file(release_path),
        "evidence_sha256": sha256_file(evidence_path),
        "dataset": {
            "dataset_id": dataset_contract["dataset_id"],
            "package_manifest_sha256": package_manifest_sha256,
            "counts": dataset_contract["counts"],
            "tokenizer_sha256": dataset_contract["tokenizer_sha256"],
            "manifest_metadata_verified": True,
        },
        "completed_steps": evidence["training"]["completed_steps"],
        "checkpoint": {
            "latest_name": evidence["checkpoint"]["latest_name"],
            "final_step": evidence["checkpoint"]["final_step"],
            "tensor_count": evidence["checkpoint"]["tensor_count"],
            "checkpoint_loaded": False,
        },
        "holdout_scope": {
            "split": evidence["holdout"]["split"],
            "segments_per_language": evidence["holdout"]["segments_per_language"],
            "holdout_content_read": False,
        },
        "router_final_healthy": evidence["router"]["final_healthy"],
        "limits": evidence["d0_limits"],
        "notes": [
            "D0 enlaza evidencia estática auditada; no reevalúa el checkpoint.",
            "D0 no abre shards, textos de holdout ni pesos.",
            "D0 no autoriza promoción, serving ni reanudación.",
        ],
    }
    report_path = output_dir / "d0_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    report = audit(args.source_root, args.data_root, args.output_dir)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    print("D0_AUDIT_READY")


if __name__ == "__main__":
    main()
