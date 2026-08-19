"""Enlace mínimo entre preflight y el ledger de estados de Sueño.

El módulo no inicia el runtime: sólo permite registrar la evidencia que mueve un
candidato desde cuarentena a preflight aprobado.
"""
from __future__ import annotations

from typing import Any

from sleep_preflight import verify_sleep_preflight_report
from sleep_state_machine import PREFLIGHT_PASS, QUARANTINED, SleepLifecycle


def apply_verified_preflight(lifecycle: SleepLifecycle, report: dict[str, Any]) -> dict[str, Any]:
    """Acepta un reporte íntegro que pertenece exactamente al candidato en cuarentena."""
    if lifecycle.state != QUARANTINED:
        raise ValueError("El candidato debe permanecer en cuarentena antes del preflight")
    verified = verify_sleep_preflight_report(report)
    if verified["candidate_id"] != lifecycle.candidate_id:
        raise ValueError("El reporte de preflight pertenece a otro candidato")
    if verified["parent_rock_state_sha256"] != lifecycle.parent_rock_state_sha256:
        raise ValueError("El reporte de preflight pertenece a otra Roca")
    return lifecycle.transition(
        PREFLIGHT_PASS,
        authority="preflight-verifier",
        evidence_sha256=verified["report_sha256"],
    )
