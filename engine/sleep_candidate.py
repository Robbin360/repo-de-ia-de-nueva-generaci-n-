"""Contratos locales para crear candidatos LoRA de Sueño sin entrenarlos.

Este módulo no abre red, no crea optimizadores y no actualiza pesos. Su única
autoridad es clonar una referencia de La Roca, añadir adaptadores LoRA aislados
y emitir manifiestos hashables para una evaluación posterior.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


MANIFEST_SCHEMA_VERSION = 1


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def state_sha256(state: dict[str, torch.Tensor]) -> str:
    """Calcula una huella determinista de tensores, incluyendo nombres, forma y tipo."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _candidate_base_state(model: AethelNextGen) -> dict[str, torch.Tensor]:
    """Recupera el estado base de un candidato LoRA con los nombres de La Roca."""
    result: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name.endswith(".lora_a") or name.endswith(".lora_b"):
            continue
        result[name.replace(".base.", ".")] = value
    return result


def create_rock_manifest(model: AethelNextGen) -> dict[str, Any]:
    """Describe La Roca sólo cuando no contiene adaptadores activos."""
    if model.lora_config is not None:
        raise ValueError("La Roca de referencia no puede contener LoRA activo")
    config = asdict(model.config)
    state_hash = state_sha256(model.state_dict())
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "aethel_rock_reference",
        "config": config,
        "config_sha256": hashlib.sha256(_canonical_json(config)).hexdigest(),
        "rock_state_sha256": state_hash,
        "lora_active": False,
        "promotion_state": "active_reference",
        "mutable_by_observation": False,
    }


def write_manifest_atomic(payload: dict[str, Any], destination: str | Path) -> Path:
    """Escribe un manifiesto JSON sin dejar archivos parciales."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_canonical_json(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return path


@dataclass
class SleepCandidate:
    """Rama candidata sin autoridad para promoción ni modificación de La Roca."""

    model: AethelNextGen
    manifest: dict[str, Any]
    reference_manifest: dict[str, Any]


def create_sleep_candidate(
    rock: AethelNextGen,
    candidate_memory_path: str | Path,
    rank: int = 8,
    alpha: float = 16.0,
    candidate_id: str = "sleep-candidate-0001",
) -> SleepCandidate:
    """Clona La Roca y añade LoRA, sin crear optimizador ni efectuar ajuste."""
    reference = create_rock_manifest(rock)
    candidate_config = NextGenConfig(**asdict(rock.config))
    candidate = AethelNextGen(candidate_config, candidate_memory_path)
    candidate.load_state_dict(rock.state_dict(), strict=True)
    lora = candidate.enable_lora(rank=rank, alpha=alpha, freeze_base=True)
    candidate_base_hash = state_sha256(_candidate_base_state(candidate))
    if candidate_base_hash != reference["rock_state_sha256"]:
        raise RuntimeError("La rama candidata no conserva exactamente La Roca de referencia")
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": "aethel_sleep_candidate",
        "candidate_id": candidate_id,
        "parent_rock_state_sha256": reference["rock_state_sha256"],
        "candidate_base_state_sha256": candidate_base_hash,
        "config_sha256": reference["config_sha256"],
        "lora": lora,
        "training_started": False,
        "optimizer_created": False,
        "eligible_for_promotion": False,
        "promotion_state": "quarantined_candidate",
        "external_action_enabled": False,
        "holdout_access_enabled": False,
        "rollback": {"operation": "discard_candidate", "changes_rock": False},
    }
    return SleepCandidate(model=candidate, manifest=manifest, reference_manifest=reference)


def verify_candidate_isolation(candidate: SleepCandidate, rock: AethelNextGen) -> dict[str, Any]:
    """Falla ante mutación de La Roca o de la copia base dentro del candidato."""
    current_reference = create_rock_manifest(rock)
    if current_reference["rock_state_sha256"] != candidate.reference_manifest["rock_state_sha256"]:
        raise ValueError("La Roca cambió después de crear el candidato")
    candidate_base_hash = state_sha256(_candidate_base_state(candidate.model))
    if candidate_base_hash != candidate.reference_manifest["rock_state_sha256"]:
        raise ValueError("El candidato alteró pesos base de La Roca")
    base_trainable = [name for name, parameter in candidate.model.named_parameters() if ".base." in name and parameter.requires_grad]
    if base_trainable:
        raise ValueError(f"Pesos base entrenables dentro del candidato: {base_trainable}")
    lora_trainable = [name for name, parameter in candidate.model.named_parameters() if "lora_" in name and parameter.requires_grad]
    if not lora_trainable:
        raise ValueError("El candidato no contiene parámetros LoRA entrenables")
    if candidate.manifest["eligible_for_promotion"] or candidate.manifest["training_started"]:
        raise ValueError("Un candidato nuevo no puede declararse promovible ni entrenado")
    return {
        "rock_state_sha256": current_reference["rock_state_sha256"],
        "candidate_base_state_sha256": candidate_base_hash,
        "lora_trainable_parameters": len(lora_trainable),
        "rollback_ready": True,
        "promotion_state": candidate.manifest["promotion_state"],
    }


def rollback_candidate(candidate: SleepCandidate, rock: AethelNextGen) -> dict[str, Any]:
    """Valida que basta descartar el candidato: La Roca nunca es reescrita."""
    verified = verify_candidate_isolation(candidate, rock)
    return {
        "operation": "discard_candidate",
        "candidate_id": candidate.manifest["candidate_id"],
        "rock_state_sha256": verified["rock_state_sha256"],
        "rock_changed": False,
        "restored_reference": True,
    }
