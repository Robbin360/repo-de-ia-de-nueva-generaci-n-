"""Preflight de Sueño sin entrenamiento.

Valida que los manifiestos de La Roca, candidato LoRA, replay en cuarentena y
Dataset congelado sean compatibles. No abre shards, no crea optimizadores y no
autoriza ajuste: una salida correcta conserva el estado de cuarentena.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable


PREFLIGHT_SCHEMA_VERSION = 1


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _require_sha256(value: Any, field: str) -> str:
    result = str(value).strip().lower()
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{field} debe ser SHA-256 hexadecimal")
    return result


def _require(payload: dict[str, Any], key: str, expected: Any | None = None) -> Any:
    if key not in payload:
        raise ValueError(f"Falta campo obligatorio: {key}")
    value = payload[key]
    if expected is not None and value != expected:
        raise ValueError(f"{key} debe ser {expected!r}")
    return value


def _verify_replay_manifest(replay: dict[str, Any], known_holdout_hashes: set[str]) -> None:
    _require(replay, "kind", "aethel_sleep_replay_quarantine")
    unsigned = {key: value for key, value in replay.items() if key != "manifest_sha256"}
    if _require_sha256(replay.get("manifest_sha256"), "manifest_sha256") != _sha256(unsigned):
        raise ValueError("El hash del manifiesto de replay no coincide")
    for key in ("eligible_for_training", "eligible_for_promotion", "holdout_access_enabled", "external_action_enabled", "optimizer_creation_enabled"):
        _require(replay, key, False)
    _require(replay, "approval_required_before_training", True)
    records = _require(replay, "records")
    if _require(replay, "record_count") != len(records):
        raise ValueError("record_count no coincide con los registros de replay")
    record_hashes = [
        _require_sha256(_require(record, "source_sha256"), "record.source_sha256") for record in records
    ]
    if len(set(record_hashes)) != len(record_hashes):
        raise ValueError("El replay contiene procedencias duplicadas")
    if known_holdout_hashes.intersection(record_hashes):
        raise ValueError("El replay colisiona con un hash de holdout")


def _verify_dataset_manifest(dataset: dict[str, Any]) -> tuple[str, str]:
    dataset_id = str(_require(dataset, "dataset_id")).strip()
    if not dataset_id:
        raise ValueError("dataset_id no puede estar vacío")
    _require(dataset, "offline_training_ready", True)
    _require(dataset, "holdout_excluded_from_tokenizer", True)
    tokenizer = _require(dataset, "tokenizer")
    if _require(tokenizer, "derived_from") != "train split only":
        raise ValueError("El tokenizador debe derivarse sólo de train")
    tokenizer_hash = _require_sha256(_require(tokenizer, "sha256"), "tokenizer.sha256")
    files = _require(dataset, "corpus_files")
    paths = [str(_require(item, "path")) for item in files]
    if not any(path.startswith("corpus/train-") for path in paths):
        raise ValueError("El Dataset no contiene shards train")
    if not any(path.startswith("corpus/holdout-") for path in paths):
        raise ValueError("El Dataset no contiene shards holdout")
    for item in files:
        _require_sha256(_require(item, "sha256"), "corpus_files.sha256")
    counts = _require(dataset, "counts")
    for split in ("train:en", "train:es", "holdout:en", "holdout:es"):
        if int(_require(counts, split)) <= 0:
            raise ValueError(f"El Dataset no tiene registros para {split}")
    return dataset_id, _sha256(dataset)


def run_sleep_preflight(
    rock_manifest: dict[str, Any],
    candidate_manifest: dict[str, Any],
    replay_manifest: dict[str, Any],
    dataset_manifest: dict[str, Any],
    known_holdout_hashes: Iterable[str] = (),
) -> dict[str, Any]:
    """Verifica compatibilidad, pero conserva prohibido todo ajuste posterior."""
    _require(rock_manifest, "kind", "aethel_rock_reference")
    _require(rock_manifest, "lora_active", False)
    rock_hash = _require_sha256(_require(rock_manifest, "rock_state_sha256"), "rock_state_sha256")
    _require(candidate_manifest, "kind", "aethel_sleep_candidate")
    if _require_sha256(_require(candidate_manifest, "parent_rock_state_sha256"), "parent_rock_state_sha256") != rock_hash:
        raise ValueError("El candidato no pertenece a La Roca indicada")
    if _require_sha256(_require(candidate_manifest, "candidate_base_state_sha256"), "candidate_base_state_sha256") != rock_hash:
        raise ValueError("Los pesos base del candidato no coinciden con La Roca")
    for key, expected in {
        "training_started": False,
        "optimizer_created": False,
        "eligible_for_promotion": False,
        "holdout_access_enabled": False,
        "external_action_enabled": False,
    }.items():
        _require(candidate_manifest, key, expected)
    holdout = {_require_sha256(item, "known_holdout_hash") for item in known_holdout_hashes}
    _verify_replay_manifest(replay_manifest, holdout)
    if _require_sha256(_require(replay_manifest, "parent_rock_state_sha256"), "replay.parent_rock_state_sha256") != rock_hash:
        raise ValueError("El replay no pertenece a La Roca indicada")
    dataset_id, dataset_sha = _verify_dataset_manifest(dataset_manifest)
    report = {
        "schema_version": PREFLIGHT_SCHEMA_VERSION,
        "kind": "aethel_sleep_preflight_report",
        "preflight_status": "quarantined_preflight_pass",
        "parent_rock_state_sha256": rock_hash,
        "candidate_id": _require(candidate_manifest, "candidate_id"),
        "dataset_id": dataset_id,
        "dataset_manifest_sha256": dataset_sha,
        "replay_manifest_sha256": replay_manifest["manifest_sha256"],
        "replay_record_count": replay_manifest["record_count"],
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "optimizer_creation_enabled": False,
        "holdout_access_enabled": False,
        "requires_runtime_authorization": True,
    }
    report["report_sha256"] = _sha256(report)
    return report


def verify_sleep_preflight_report(report: dict[str, Any]) -> dict[str, Any]:
    """Verifica un reporte emitido por preflight sin ejecutar ninguna transición."""
    _require(report, "kind", "aethel_sleep_preflight_report")
    _require(report, "preflight_status", "quarantined_preflight_pass")
    unsigned = {key: value for key, value in report.items() if key != "report_sha256"}
    if _require_sha256(_require(report, "report_sha256"), "report_sha256") != _sha256(unsigned):
        raise ValueError("El hash del reporte de preflight no coincide")
    for key in (
        "eligible_for_training",
        "eligible_for_promotion",
        "optimizer_creation_enabled",
        "holdout_access_enabled",
    ):
        _require(report, key, False)
    _require(report, "requires_runtime_authorization", True)
    _require_sha256(_require(report, "parent_rock_state_sha256"), "parent_rock_state_sha256")
    if not str(_require(report, "candidate_id")).strip():
        raise ValueError("candidate_id no puede estar vacío")
    return dict(report)
