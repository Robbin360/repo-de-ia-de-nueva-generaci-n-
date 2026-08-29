"""Contrato CPU de la puerta opcional de salud top-k."""
from __future__ import annotations

from router_health import classify_router_health


BASE = {"imbalance": 0.1, "entropy": 0.8}


def test_legacy_contract_remains_compatible() -> None:
    health = classify_router_health([BASE], max_router_imbalance=0.2, min_router_entropy=0.7)
    assert health.healthy is True
    assert health.as_dict() == {
        "healthy": True,
        "max_imbalance": 0.1,
        "min_entropy": 0.8,
    }


def test_topk_gate_rejects_concentration_and_overflow() -> None:
    layer = {
        **BASE,
        "max_density": 0.9,
        "coverage": 0.25,
        "overflow_rate": 0.2,
    }
    health = classify_router_health(
        [layer],
        max_router_imbalance=0.2,
        min_router_entropy=0.7,
        max_topk_density=0.5,
        min_topk_coverage=0.5,
        max_overflow_rate=0.05,
    )
    assert health.healthy is False
    assert health.max_topk_density == 0.9
    assert health.min_topk_coverage == 0.25
    assert health.max_overflow_rate == 0.2


def test_topk_gate_accepts_balanced_layer() -> None:
    layer = {
        **BASE,
        "max_density": 0.3,
        "coverage": 0.75,
        "overflow_rate": 0.0,
    }
    health = classify_router_health(
        [layer],
        max_router_imbalance=0.2,
        min_router_entropy=0.7,
        max_topk_density=0.5,
        min_topk_coverage=0.5,
        max_overflow_rate=0.05,
    )
    assert health.healthy is True


def test_topk_threshold_requires_topk_fields() -> None:
    try:
        classify_router_health(
            [BASE],
            max_router_imbalance=0.2,
            min_router_entropy=0.7,
            max_topk_density=0.5,
        )
    except KeyError as error:
        assert "max_density" in str(error)
    else:
        raise AssertionError("La puerta top-k debe exigir sus métricas explícitas.")


if __name__ == "__main__":
    test_legacy_contract_remains_compatible()
    test_topk_gate_rejects_concentration_and_overflow()
    test_topk_gate_accepts_balanced_layer()
    test_topk_threshold_requires_topk_fields()
    print("test_router_health_topk_contract: OK")
