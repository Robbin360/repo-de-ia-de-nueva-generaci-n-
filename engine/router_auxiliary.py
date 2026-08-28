"""Contrato puro para la contribución auxiliar de balanceo del router MoE.

No importa PyTorch ni accede a archivos, datos, GPU, red o artefactos. El
operador de multiplicación se deja genérico para preservar el gradiente cuando
el núcleo le entrega tensores.
"""
from __future__ import annotations

import math
from typing import TypeVar


LossLike = TypeVar("LossLike")


def validate_router_aux_loss_weight(value: object) -> float:
    """Acepta únicamente pesos auxiliares finitos y no negativos."""

    if isinstance(value, bool):
        raise TypeError("router_aux_loss_weight debe ser numérico, no booleano.")
    try:
        weight = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("router_aux_loss_weight debe ser numérico.") from exc
    if not math.isfinite(weight):
        raise ValueError("router_aux_loss_weight debe ser finito.")
    if weight < 0.0:
        raise ValueError("router_aux_loss_weight no puede ser negativo.")
    return weight


def validate_router_jitter_noise(value: object) -> float:
    """Acepta ruido gaussiano finito en ``[0, 1)`` sólo para selección en entrenamiento."""

    if isinstance(value, bool):
        raise TypeError("router_jitter_noise debe ser numérico, no booleano.")
    try:
        noise = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("router_jitter_noise debe ser numérico.") from exc
    if not math.isfinite(noise):
        raise ValueError("router_jitter_noise debe ser finito.")
    if not 0.0 <= noise < 1.0:
        raise ValueError("router_jitter_noise debe estar en el intervalo [0, 1).")
    return noise


def add_router_auxiliary_loss(base_loss: LossLike, aux_loss: LossLike, *, weight: object) -> LossLike:
    """Suma la contribución auxiliar con un peso validado y determinista."""

    return base_loss + validate_router_aux_loss_weight(weight) * aux_loss  # type: ignore[operator, return-value]


def router_balance_auxiliary_loss(token_density: LossLike, router_probability: LossLike) -> LossLike:
    """Devuelve ``n_experts * sum(density * probability)`` de forma verificable.

    Ambos argumentos deben ser vectores unidimensionales con igual número de
    expertos. La implementación permanece agnóstica a PyTorch, pero conserva
    sus operaciones tensoriales y por tanto el gradiente de ``router_probability``.
    """

    density_shape = getattr(token_density, "shape", None)
    probability_shape = getattr(router_probability, "shape", None)
    if density_shape is None or probability_shape is None:
        raise TypeError("La pérdida auxiliar exige vectores tensoriales con atributo shape.")
    if len(density_shape) != 1 or density_shape != probability_shape:
        raise ValueError("density y probability deben ser vectores con la misma forma.")
    experts = getattr(token_density, "numel", lambda: 0)()
    if experts < 1:
        raise ValueError("La pérdida auxiliar exige al menos un experto.")
    return experts * (token_density * router_probability).sum()  # type: ignore[operator, return-value]


def router_entropy_regularization_loss(router_probability: LossLike) -> LossLike:
    """Devuelve la entropía negativa media normalizada del router.

    Minimizar esta cantidad equivale a maximizar la entropía de la distribución
    suave, proporcionando una señal densa contra el colapso de probabilidades.
    El helper permanece agnóstico a PyTorch y conserva operaciones tensoriales.
    ``router_probability`` debe ser una matriz [tokens, expertos] no vacía.
    """

    probability_shape = getattr(router_probability, "shape", None)
    if probability_shape is None or len(probability_shape) != 2:
        raise ValueError("router_probability debe ser una matriz [tokens, expertos].")
    tokens, experts = probability_shape
    if tokens < 1 or experts < 2:
        raise ValueError("La regularización exige tokens y al menos dos expertos.")
    return (router_probability.clamp_min(1e-9) * router_probability.clamp_min(1e-9).log()).sum(dim=-1).mean() / math.log(experts)  # type: ignore[operator, return-value]
