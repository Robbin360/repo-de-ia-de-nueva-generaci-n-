"""Pruebas CPU puras del contrato de salud del router MoE.

No usa pesos, corpus, shards, holdout, GPU, red ni artefactos de entrenamiento.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from router_health import classify_router_health  # noqa: E402


class RouterHealthContractTests(unittest.TestCase):
    def test_accepts_exact_threshold_boundaries(self) -> None:
        health = classify_router_health(
            [{"entropy": 0.50, "imbalance": 0.30}],
            max_router_imbalance=0.30,
            min_router_entropy=0.50,
        )
        self.assertEqual(
            health.as_dict(),
            {"healthy": True, "max_imbalance": 0.30, "min_entropy": 0.50},
        )

    def test_uses_worst_layer_extremes(self) -> None:
        health = classify_router_health(
            [
                {"entropy": 0.88, "imbalance": 0.10},
                {"entropy": 0.49, "imbalance": 0.18},
                {"entropy": 0.73, "imbalance": 0.31},
            ],
            max_router_imbalance=0.30,
            min_router_entropy=0.50,
        )
        self.assertFalse(health.healthy)
        self.assertEqual(health.min_entropy, 0.49)
        self.assertEqual(health.max_imbalance, 0.31)

    def test_rejects_missing_empty_or_nonfinite_telemetry(self) -> None:
        with self.assertRaises(ValueError):
            classify_router_health([], max_router_imbalance=0.30, min_router_entropy=0.50)
        with self.assertRaises(KeyError):
            classify_router_health([{"entropy": 0.70}], max_router_imbalance=0.30, min_router_entropy=0.50)
        with self.assertRaises(ValueError):
            classify_router_health(
                [{"entropy": math.nan, "imbalance": 0.10}],
                max_router_imbalance=0.30,
                min_router_entropy=0.50,
            )

    def test_rejects_invalid_thresholds(self) -> None:
        routing = [{"entropy": 0.70, "imbalance": 0.10}]
        with self.assertRaises(ValueError):
            classify_router_health(routing, max_router_imbalance=-0.01, min_router_entropy=0.50)
        with self.assertRaises(ValueError):
            classify_router_health(routing, max_router_imbalance=0.30, min_router_entropy=1.01)


if __name__ == "__main__":
    unittest.main()
