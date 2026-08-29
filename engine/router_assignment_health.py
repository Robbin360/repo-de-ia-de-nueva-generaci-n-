"""Diagnóstico determinista de asignaciones duras de un router MoE.

Este módulo no cambia la política de routing. Resume la asignación top-k que
realmente llegó a los expertos, separándola de la probabilidad suave usada por
las pérdidas auxiliares.
"""
from __future__ import annotations

import math
from collections.abc import Iterable


def analyze_hard_assignments(
    assignments: Iterable[int],
    *,
    expert_count: int,
    capacity: int | None = None,
) -> dict[str, float | int | list[int]]:
    """Resume cobertura, densidad, concentración y overflow de top-k.

    ``assignments`` puede contener una entrada por selección (por ejemplo,
    tokens multiplicados por ``k``). El resultado es una métrica diagnóstica,
    no una pérdida diferenciable.
    """

    if isinstance(expert_count, bool) or not isinstance(expert_count, int) or expert_count < 1:
        raise ValueError("expert_count debe ser un entero positivo.")
    if capacity is not None and (
        isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1
    ):
        raise ValueError("capacity debe ser None o un entero positivo.")

    counts = [0] * expert_count
    total = 0
    for assignment in assignments:
        if isinstance(assignment, bool) or not isinstance(assignment, int):
            raise TypeError("Cada asignación debe ser un índice entero de experto.")
        if assignment < 0 or assignment >= expert_count:
            raise ValueError("Una asignación está fuera del rango de expertos.")
        counts[assignment] += 1
        total += 1

    if total == 0:
        raise ValueError("Se requiere al menos una asignación top-k.")

    densities = [count / total for count in counts]
    uniform = 1.0 / expert_count
    overflow = sum(max(0, count - capacity) for count in counts) if capacity is not None else 0
    return {
        "assignment_count": total,
        "counts": counts,
        "coverage": sum(count > 0 for count in counts) / expert_count,
        "max_density": max(densities),
        "min_density": min(densities),
        "max_imbalance": max(abs(density - uniform) for density in densities),
        "overflow_count": overflow,
        "overflow_rate": overflow / total,
        "entropy": -sum(
            density * math.log(density, expert_count)
            for density in densities
            if density > 0.0
        ),
    }
