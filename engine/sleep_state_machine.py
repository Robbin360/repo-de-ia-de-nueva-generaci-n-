"""Máquina de estados local para gobernar candidatos de Sueño.

No ejecuta entrenamiento, no crea optimizadores y no promueve pesos. Sólo hace
cumplir transiciones explícitas y deja un ledger hash-encadenado para auditoría.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


QUARANTINED = "quarantined"
PREFLIGHT_PASS = "preflight_pass"
AUTHORIZED = "authorized"
RUNNING = "running"
EVALUATED = "evaluated"
REJECTED = "rejected"
PROMOTABLE = "promotable"
PROMOTED = "promoted"
ROLLED_BACK = "rolled_back"

TRANSITIONS: dict[str, set[str]] = {
    QUARANTINED: {PREFLIGHT_PASS, REJECTED},
    PREFLIGHT_PASS: {AUTHORIZED, REJECTED},
    AUTHORIZED: {RUNNING, REJECTED},
    RUNNING: {EVALUATED, REJECTED},
    EVALUATED: {PROMOTABLE, REJECTED},
    PROMOTABLE: {PROMOTED, ROLLED_BACK, REJECTED},
    PROMOTED: {ROLLED_BACK},
    REJECTED: {ROLLED_BACK},
    ROLLED_BACK: set(),
}
AUTHORITIES: dict[tuple[str, str], str] = {
    (QUARANTINED, PREFLIGHT_PASS): "preflight-verifier",
    (QUARANTINED, REJECTED): "system-or-reviewer",
    (PREFLIGHT_PASS, AUTHORIZED): "human-execution-approver",
    (PREFLIGHT_PASS, REJECTED): "system-or-reviewer",
    (AUTHORIZED, RUNNING): "runtime-executor",
    (AUTHORIZED, REJECTED): "system-or-reviewer",
    (RUNNING, EVALUATED): "evaluation-runner",
    (RUNNING, REJECTED): "system-or-reviewer",
    (EVALUATED, PROMOTABLE): "evaluation-reviewer",
    (EVALUATED, REJECTED): "evaluation-reviewer",
    (PROMOTABLE, PROMOTED): "human-promotion-approver",
    (PROMOTABLE, ROLLED_BACK): "rollback-operator",
    (PROMOTABLE, REJECTED): "evaluation-reviewer",
    (PROMOTED, ROLLED_BACK): "rollback-operator",
    (REJECTED, ROLLED_BACK): "rollback-operator",
}


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


@dataclass
class SleepLifecycle:
    """Registro de estado para un solo candidato, sin autoridad de cómputo."""

    candidate_id: str
    parent_rock_state_sha256: str
    state: str = QUARANTINED
    ledger: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.candidate_id:
            raise ValueError("candidate_id es obligatorio")
        if len(self.parent_rock_state_sha256) != 64:
            raise ValueError("parent_rock_state_sha256 debe ser SHA-256")
        if not self.ledger:
            self._append_initial_event()

    def _append_initial_event(self) -> None:
        event = {
            "sequence": 0,
            "from_state": None,
            "to_state": QUARANTINED,
            "authority": "system-bootstrap",
            "evidence_sha256": None,
            "previous_event_sha256": None,
            "candidate_id": self.candidate_id,
            "parent_rock_state_sha256": self.parent_rock_state_sha256,
        }
        event["event_sha256"] = _hash(event)
        self.ledger.append(event)

    def transition(self, to_state: str, authority: str, evidence_sha256: str) -> dict[str, Any]:
        """Efectúa una transición válida sin realizar la operación que representa."""
        if to_state not in TRANSITIONS.get(self.state, set()):
            raise ValueError(f"Transición prohibida: {self.state} -> {to_state}")
        required_authority = AUTHORITIES[(self.state, to_state)]
        if authority != required_authority:
            raise PermissionError(f"Autoridad requerida: {required_authority}")
        if len(evidence_sha256) != 64 or any(char not in "0123456789abcdef" for char in evidence_sha256.lower()):
            raise ValueError("evidence_sha256 debe ser SHA-256 hexadecimal")
        previous = self.ledger[-1]
        event = {
            "sequence": len(self.ledger),
            "from_state": self.state,
            "to_state": to_state,
            "authority": authority,
            "evidence_sha256": evidence_sha256.lower(),
            "previous_event_sha256": previous["event_sha256"],
            "candidate_id": self.candidate_id,
            "parent_rock_state_sha256": self.parent_rock_state_sha256,
        }
        event["event_sha256"] = _hash(event)
        self.ledger.append(event)
        self.state = to_state
        return dict(event)

    def verify_ledger(self) -> dict[str, Any]:
        """Verifica el hash-encadenamiento y la coherencia de todas las transiciones."""
        if not self.ledger or self.ledger[0]["to_state"] != QUARANTINED:
            raise ValueError("El ledger debe comenzar en cuarentena")
        previous_hash: str | None = None
        current = QUARANTINED
        for index, event in enumerate(self.ledger):
            stored = event.get("event_sha256")
            unsigned = {key: value for key, value in event.items() if key != "event_sha256"}
            if stored != _hash(unsigned):
                raise ValueError("El ledger contiene un evento alterado")
            if event["sequence"] != index or event["previous_event_sha256"] != previous_hash:
                raise ValueError("El ledger perdió su encadenamiento")
            if index:
                if event["from_state"] != current:
                    raise ValueError("El ledger tiene estado previo inconsistente")
                if event["to_state"] not in TRANSITIONS[current]:
                    raise ValueError("El ledger contiene transición prohibida")
                if event["authority"] != AUTHORITIES[(current, event["to_state"])]:
                    raise ValueError("El ledger tiene autoridad inconsistente")
                current = event["to_state"]
            previous_hash = stored
        if current != self.state:
            raise ValueError("El estado actual no coincide con el ledger")
        return {"candidate_id": self.candidate_id, "state": self.state, "events": len(self.ledger), "ledger_valid": True}
