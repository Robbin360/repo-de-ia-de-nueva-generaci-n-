"""Sonda CPU pura para estudiar balanceo top-k mediante carga histórica.

No cambia el router de Aethel ni constituye una validación de entrenamiento. La
función sólo transforma logits con una penalización determinista de carga.
"""
from __future__ import annotations

import math
from typing import Iterable


def load_aware_logits(logits: Iterable[float], load: Iterable[float], *, strength: float) -> list[float]:
    """Resta una penalización centrada por carga a cada logit.

    ``load`` representa una EMA de asignaciones previas. Centrar la carga evita
    desplazar todos los logits a la vez. No se permite una fuerza negativa.
    """
    scores = [float(value) for value in logits]
    loads = [float(value) for value in load]
    if len(scores) < 2 or len(scores) != len(loads):
        raise ValueError("logits y load deben tener la misma longitud y al menos dos expertos.")
    if not all(math.isfinite(value) for value in scores + loads):
        raise ValueError("logits y load deben ser finitos.")
    if not math.isfinite(float(strength)) or float(strength) < 0.0:
        raise ValueError("strength debe ser finito y no negativo.")
    mean_load = sum(loads) / len(loads)
    return [score - float(strength) * (expert_load - mean_load) for score, expert_load in zip(scores, loads)]


def top_k_indices(scores: Iterable[float], k: int = 2) -> tuple[int, ...]:
    """Selección top-k estable: empates se resuelven por índice ascendente."""
    values = list(float(value) for value in scores)
    if not 1 <= k <= len(values):
        raise ValueError("k debe estar entre uno y el número de expertos.")
    return tuple(sorted(range(len(values)), key=lambda index: (-values[index], index))[:k])


if __name__ == "__main__":
    base = [1.0, 0.99, 0.2, 0.1]
    load = [10.0, 9.0, 0.0, 0.0]
    before = top_k_indices(base)
    after = top_k_indices(load_aware_logits(base, load, strength=0.2))
    print({"before": before, "after": after})
    assert before == (0, 1)
    assert after == (2, 3)
    print("ROUTER_LOAD_AWARE_PROBE_OK")
