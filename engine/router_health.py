"""Contrato puro y determinista para resumir la salud del router MoE.

No importa PyTorch, no abre archivos ni conoce Dataset, pesos, GPU o red. El
entrenador le entrega únicamente la telemetría ya calculada por capa.
"""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class RouterHealth:
    """Resumen serializable de los extremos que gobiernan la puerta MoE."""

    healthy: bool
    max_imbalance: float
    min_entropy: float

    def as_dict(self) -> dict[str, bool | float]:
        return {
            "healthy": self.healthy,
            "max_imbalance": self.max_imbalance,
            "min_entropy": self.min_entropy,
        }


def _finite_float(value: object, *, field: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{field} debe ser numérico, no booleano.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field} debe ser numérico.") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{field} debe ser finito.")
    return numeric


def _normalized_metric(value: object, *, field: str) -> float:
    numeric = _finite_float(value, field=field)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{field} debe pertenecer al intervalo normalizado [0, 1].")
    return numeric


def classify_router_health(
    routing: Iterable[Mapping[str, object]],
    *,
    max_router_imbalance: float,
    min_router_entropy: float,
) -> RouterHealth:
    """Aplica la puerta MoE sobre todas las capas y falla de forma cerrada.

    Un router sólo se considera saludable cuando la mayor desviación de carga
    no supera el umbral y la menor entropía lo alcanza. La ausencia de capas o
    telemetría malformada es un error explícito: no puede convertirse en un
    ``healthy=True`` implícito.
    """

    imbalance_limit = _normalized_metric(max_router_imbalance, field="max_router_imbalance")
    entropy_floor = _normalized_metric(min_router_entropy, field="min_router_entropy")
    layers = list(routing)
    if not layers:
        raise ValueError("La puerta del router exige telemetría de al menos una capa.")

    imbalances: list[float] = []
    entropies: list[float] = []
    for index, layer in enumerate(layers):
        if not isinstance(layer, Mapping):
            raise TypeError(f"La telemetría de la capa {index} debe ser un mapeo.")
        try:
            imbalance = layer["imbalance"]
            entropy = layer["entropy"]
        except KeyError as exc:
            raise KeyError(f"Falta {exc.args[0]!r} en la telemetría de la capa {index}.") from exc
        imbalances.append(_normalized_metric(imbalance, field=f"routing[{index}].imbalance"))
        entropies.append(_normalized_metric(entropy, field=f"routing[{index}].entropy"))

    max_imbalance = max(imbalances)
    min_entropy = min(entropies)
    return RouterHealth(
        healthy=max_imbalance <= imbalance_limit and min_entropy >= entropy_floor,
        max_imbalance=max_imbalance,
        min_entropy=min_entropy,
    )
