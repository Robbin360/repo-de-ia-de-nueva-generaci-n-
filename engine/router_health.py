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
    max_topk_density: float | None = None
    min_topk_coverage: float | None = None
    max_overflow_rate: float | None = None

    def as_dict(self) -> dict[str, bool | float]:
        result: dict[str, bool | float] = {
            "healthy": self.healthy,
            "max_imbalance": self.max_imbalance,
            "min_entropy": self.min_entropy,
        }
        if self.max_topk_density is not None:
            result["max_topk_density"] = self.max_topk_density
        if self.min_topk_coverage is not None:
            result["min_topk_coverage"] = self.min_topk_coverage
        if self.max_overflow_rate is not None:
            result["max_overflow_rate"] = self.max_overflow_rate
        return result


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
    max_topk_density: float | None = None,
    min_topk_coverage: float | None = None,
    max_overflow_rate: float | None = None,
) -> RouterHealth:
    """Aplica la puerta MoE sobre todas las capas y falla de forma cerrada.

    Un router sólo se considera saludable cuando la mayor desviación de carga
    no supera el umbral y la menor entropía lo alcanza. La ausencia de capas o
    telemetría malformada es un error explícito: no puede convertirse en un
    ``healthy=True`` implícito.
    """

    imbalance_limit = _normalized_metric(max_router_imbalance, field="max_router_imbalance")
    entropy_floor = _normalized_metric(min_router_entropy, field="min_router_entropy")
    density_limit = (
        _normalized_metric(max_topk_density, field="max_topk_density")
        if max_topk_density is not None
        else None
    )
    coverage_floor = (
        _normalized_metric(min_topk_coverage, field="min_topk_coverage")
        if min_topk_coverage is not None
        else None
    )
    overflow_limit = (
        _normalized_metric(max_overflow_rate, field="max_overflow_rate")
        if max_overflow_rate is not None
        else None
    )
    layers = list(routing)
    if not layers:
        raise ValueError("La puerta del router exige telemetría de al menos una capa.")

    imbalances: list[float] = []
    entropies: list[float] = []
    topk_densities: list[float] = []
    topk_coverages: list[float] = []
    overflow_rates: list[float] = []
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
        if density_limit is not None or coverage_floor is not None or overflow_limit is not None:
            for key in ("max_density", "coverage", "overflow_rate"):
                if key not in layer:
                    raise KeyError(f"Falta {key!r} en la telemetría top-k de la capa {index}.")
            topk_densities.append(
                _normalized_metric(layer["max_density"], field=f"routing[{index}].max_density")
            )
            topk_coverages.append(
                _normalized_metric(layer["coverage"], field=f"routing[{index}].coverage")
            )
            overflow_rates.append(
                _normalized_metric(layer["overflow_rate"], field=f"routing[{index}].overflow_rate")
            )

    max_imbalance = max(imbalances)
    min_entropy = min(entropies)
    observed_max_density = max(topk_densities) if topk_densities else None
    observed_min_coverage = min(topk_coverages) if topk_coverages else None
    observed_max_overflow = max(overflow_rates) if overflow_rates else None
    topk_healthy = (
        density_limit is None or observed_max_density is not None and observed_max_density <= density_limit
    ) and (
        coverage_floor is None or observed_min_coverage is not None and observed_min_coverage >= coverage_floor
    ) and (
        overflow_limit is None or observed_max_overflow is not None and observed_max_overflow <= overflow_limit
    )
    return RouterHealth(
        healthy=max_imbalance <= imbalance_limit and min_entropy >= entropy_floor and topk_healthy,
        max_imbalance=max_imbalance,
        min_entropy=min_entropy,
        max_topk_density=observed_max_density,
        min_topk_coverage=observed_min_coverage,
        max_overflow_rate=observed_max_overflow,
    )
