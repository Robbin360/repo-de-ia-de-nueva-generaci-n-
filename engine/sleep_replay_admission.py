"""Admisión de replay para Sueño sin entrenamiento ni acceso al holdout.

Este módulo transforma sólo metadatos de eventos líquidos ya curados en un
manifiesto de replay en cuarentena. No abre archivos de corpus, no retiene texto
de los eventos, no crea optimizadores y no cambia La Roca ni candidatos LoRA.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable


ADMISSION_SCHEMA_VERSION = 1
REQUIRED_EVENT_FIELDS = frozenset(
    {
        "event_id",
        "source",
        "source_sha256",
        "language",
        "domain",
        "priority",
        "ttl_observations",
        "eligible_for_sleep",
        "curation_status",
        "holdout_member",
    }
)
REQUIRED_APPROVAL_FIELDS = frozenset({"approval_id", "event_id", "source_sha256", "status", "approved_by"})


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bounded_number(value: Any, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} debe ser numérico") from error
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{field} debe estar entre cero y uno")
    return result


@dataclass(frozen=True)
class AdmissionRecord:
    event_id: str
    source: str
    source_sha256: str
    language: str
    domain: str
    priority: float
    ttl_observations: int
    approval_id: str
    approved_by: str

    def public_dict(self) -> dict[str, Any]:
        """Metadatos mínimos: el manifiesto nunca copia contenido del evento."""
        return {
            "event_id": self.event_id,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "language": self.language,
            "domain": self.domain,
            "priority": self.priority,
            "ttl_observations": self.ttl_observations,
            "approval_id": self.approval_id,
            "approved_by": self.approved_by,
        }


def _validated_approval_index(approvals: Iterable[dict[str, Any]]) -> dict[str, dict[str, str]]:
    """Indexa aprobaciones suministradas desde una autoridad separada del evento."""
    index: dict[str, dict[str, str]] = {}
    for approval in approvals:
        missing = sorted(REQUIRED_APPROVAL_FIELDS.difference(approval))
        if missing:
            raise ValueError(f"Aprobación sin campos obligatorios: {missing}")
        approval_id = str(approval["approval_id"]).strip()
        event_id = str(approval["event_id"]).strip()
        digest = str(approval["source_sha256"]).strip().lower()
        reviewer = str(approval["approved_by"]).strip()
        if not approval_id or not event_id or not reviewer:
            raise ValueError("approval_id, event_id y approved_by son obligatorios")
        if str(approval["status"]) != "approved":
            raise ValueError("La aprobación independiente no tiene estado approved")
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise ValueError("La aprobación debe estar vinculada a un SHA-256")
        if event_id in index:
            raise ValueError("Hay aprobaciones independientes duplicadas para un evento")
        index[event_id] = {"approval_id": approval_id, "source_sha256": digest, "approved_by": reviewer}
    return index


def review_event(
    event: dict[str, Any], known_holdout_hashes: Iterable[str], approvals_by_event: dict[str, dict[str, str]]
) -> AdmissionRecord:
    """Acepta sólo un evento curado, aprobado por registro separado y fuera de holdout."""
    missing = sorted(REQUIRED_EVENT_FIELDS.difference(event))
    if missing:
        raise ValueError(f"Evento sin campos obligatorios: {missing}")
    event_id = str(event["event_id"]).strip()
    source = str(event["source"]).strip()
    digest = str(event["source_sha256"]).strip().lower()
    language = str(event["language"]).strip().lower()
    domain = str(event["domain"]).strip().lower()
    if not event_id or not source:
        raise ValueError("event_id y source son obligatorios")
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError("source_sha256 debe ser un SHA-256 hexadecimal")
    if language not in {"en", "es"}:
        raise ValueError("Sólo se admite idioma en o es")
    if not domain:
        raise ValueError("domain no puede estar vacío")
    if bool(event["holdout_member"]) or digest in set(known_holdout_hashes):
        raise ValueError("El evento colisiona con holdout y no puede entrar a replay")
    if not bool(event["eligible_for_sleep"]):
        raise ValueError("El evento no es elegible para Sueño")
    if str(event["curation_status"]) != "curated":
        raise ValueError("El evento no fue curado")
    approval = approvals_by_event.get(event_id)
    if approval is None:
        raise ValueError("El evento no tiene una aprobación independiente")
    if approval["source_sha256"] != digest:
        raise ValueError("La aprobación independiente no coincide con la procedencia")
    ttl = int(event["ttl_observations"])
    if ttl <= 0:
        raise ValueError("El TTL debe seguir vigente")
    return AdmissionRecord(
        event_id=event_id,
        source=source,
        source_sha256=digest,
        language=language,
        domain=domain,
        priority=_bounded_number(event["priority"], "priority"),
        ttl_observations=ttl,
        approval_id=approval["approval_id"],
        approved_by=approval["approved_by"],
    )


def build_quarantined_replay_manifest(
    events: Iterable[dict[str, Any]],
    approvals: Iterable[dict[str, Any]],
    known_holdout_hashes: Iterable[str],
    parent_rock_state_sha256: str,
    max_records: int = 256,
) -> dict[str, Any]:
    """Construye una selección de metadatos apta sólo para revisión posterior.

    `eligible_for_training` queda en falso incluso cuando todos los eventos están
    aprobados: crear el manifiesto no abre la puerta de ajuste LoRA.
    """
    if not parent_rock_state_sha256 or len(parent_rock_state_sha256) != 64:
        raise ValueError("El hash de La Roca debe ser SHA-256")
    if max_records < 1:
        raise ValueError("max_records debe ser positivo")
    holdout = frozenset(str(item).lower() for item in known_holdout_hashes)
    approval_index = _validated_approval_index(approvals)
    admitted: list[AdmissionRecord] = []
    seen_ids: set[str] = set()
    seen_hashes: set[str] = set()
    for event in events:
        record = review_event(event, holdout, approval_index)
        if record.event_id in seen_ids or record.source_sha256 in seen_hashes:
            raise ValueError("Replay duplicado por event_id o source_sha256")
        seen_ids.add(record.event_id)
        seen_hashes.add(record.source_sha256)
        admitted.append(record)
    admitted.sort(key=lambda item: (-item.priority, item.event_id))
    selected = admitted[:max_records]
    records = [item.public_dict() for item in selected]
    manifest_without_hash = {
        "schema_version": ADMISSION_SCHEMA_VERSION,
        "kind": "aethel_sleep_replay_quarantine",
        "parent_rock_state_sha256": parent_rock_state_sha256.lower(),
        "records": records,
        "record_count": len(records),
        "eligible_for_training": False,
        "eligible_for_promotion": False,
        "holdout_access_enabled": False,
        "external_action_enabled": False,
        "optimizer_creation_enabled": False,
        "approval_required_before_training": True,
    }
    manifest = dict(manifest_without_hash)
    manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest_without_hash)).hexdigest()
    return manifest
