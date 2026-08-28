"""Contrato verificable de reanudación Aethel entre sesiones.

El módulo no inicia entrenamiento ni accede a Kaggle. Agrupa el estado que debe
acompañar un checkpoint para continuar en una sesión posterior sin cambiar la
identidad del experimento.
"""
from __future__ import annotations

import hashlib
import random
from pathlib import Path
from typing import Any

import torch


RESUME_SCHEMA = "aethel-training-resume/v2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_layout_manifest(corpus_dir: Path) -> dict[str, Any]:
    paths = sorted(list(corpus_dir.glob("train-*.jsonl")) + list(corpus_dir.glob("train-*.jsonl.gz")))
    if not paths:
        raise ValueError("No hay shards train para construir el manifiesto de reanudación.")
    return {
        "files": [{"name": path.name, "bytes": path.stat().st_size} for path in paths],
        "layout_sha256": hashlib.sha256("\n".join(f"{path.name}:{path.stat().st_size}" for path in paths).encode("utf-8")).hexdigest(),
    }


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state(device)
    return state


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    if not isinstance(state, dict) or "python" not in state or "torch_cpu" not in state:
        raise ValueError("El checkpoint no contiene estado RNG completo para reanudar.")
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    if device.type == "cuda":
        cuda_state = state.get("torch_cuda")
        if not isinstance(cuda_state, torch.Tensor):
            raise ValueError("El checkpoint no contiene RNG CUDA para reanudar en GPU.")
        torch.cuda.set_rng_state(cuda_state.cpu(), device)


def build_resume_contract(
    *, config: dict[str, Any], tokenizer_sha256: str, corpus_dir: Path, data_manifest_sha256: str | None, strategy: str, world_size: int, trainer_profile: dict[str, Any]
) -> dict[str, Any]:
    return {
        "schema": RESUME_SCHEMA,
        "config": config,
        "tokenizer_sha256": tokenizer_sha256,
        "corpus_layout": corpus_layout_manifest(corpus_dir),
        "data_manifest_sha256": data_manifest_sha256,
        "strategy": strategy,
        "world_size": world_size,
        "trainer_profile": trainer_profile,
    }


def validate_resume_contract(saved: dict[str, Any], active: dict[str, Any]) -> None:
    if not isinstance(saved, dict) or saved.get("schema") != RESUME_SCHEMA:
        raise ValueError("El checkpoint no declara el contrato de reanudación Aethel v1.")
    required = ("config", "tokenizer_sha256", "corpus_layout", "data_manifest_sha256", "strategy", "world_size", "trainer_profile")
    if any(saved.get(field) != active.get(field) for field in required):
        raise ValueError("El contrato de datos, topología o entrenamiento no coincide; la reanudación queda bloqueada.")


def require_full_resume_payload(checkpoint: dict[str, Any]) -> None:
    required = (
        "model",
        "reference_state",
        "optimizer",
        "scaler",
        "rng_state",
        "runtime_state",
        "resume_contract",
        "step",
    )
    missing = [field for field in required if field not in checkpoint]
    if missing:
        raise ValueError(f"El checkpoint no permite reanudación fiel; faltan: {', '.join(missing)}.")
