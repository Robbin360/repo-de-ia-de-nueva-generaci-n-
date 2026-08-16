"""Selecciona una precisión segura según capacidad CUDA; permite sobreescritura explícita."""
from __future__ import annotations


def select_precision(cuda_major: int, requested: str | None = None) -> str:
    if requested:
        if requested not in {"fp16", "bf16", "fp32"}:
            raise ValueError("La precisión debe ser fp16, bf16 o fp32")
        return requested
    return "bf16" if cuda_major >= 8 else "fp16"
