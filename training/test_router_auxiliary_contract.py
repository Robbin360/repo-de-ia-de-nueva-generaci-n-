"""Pruebas puras del contrato de ponderación auxiliar del router MoE.

No usa PyTorch, corpus, shards, holdout, pesos, GPU, Kaggle ni red.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engine"))

from router_auxiliary import add_router_auxiliary_loss, router_balance_auxiliary_loss, validate_router_aux_loss_weight  # noqa: E402


class RouterAuxiliaryContractTests(unittest.TestCase):
    def test_historical_default_preserves_the_existing_formula(self) -> None:
        self.assertEqual(validate_router_aux_loss_weight(0.01), 0.01)
        self.assertAlmostEqual(add_router_auxiliary_loss(7.5, 2.0, weight=0.01), 7.52)

    def test_zero_weight_is_explicit_and_deterministic(self) -> None:
        self.assertEqual(add_router_auxiliary_loss(7.5, 2.0, weight=0.0), 7.5)

    def test_rejects_invalid_weights(self) -> None:
        for invalid in (-0.001, math.nan, math.inf):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    validate_router_aux_loss_weight(invalid)
        with self.assertRaises(TypeError):
            validate_router_aux_loss_weight(True)
        with self.assertRaises(TypeError):
            validate_router_aux_loss_weight("no-numérico")

    def test_collapsed_density_pushes_probability_away_from_the_dominant_expert(self) -> None:
        logits = torch.tensor([2.0, 0.0], dtype=torch.float64, requires_grad=True)
        probabilities = torch.softmax(logits, dim=0)
        loss = router_balance_auxiliary_loss(torch.tensor([1.0, 0.0], dtype=torch.float64), probabilities)
        loss.backward()
        self.assertGreater(float(logits.grad[0]), 0.0)
        self.assertLess(float(logits.grad[1]), 0.0)

    def test_uniform_density_has_no_preference_between_experts(self) -> None:
        logits = torch.tensor([2.0, 0.0], dtype=torch.float64, requires_grad=True)
        probabilities = torch.softmax(logits, dim=0)
        loss = router_balance_auxiliary_loss(torch.tensor([0.5, 0.5], dtype=torch.float64), probabilities)
        loss.backward()
        self.assertTrue(torch.allclose(logits.grad, torch.zeros_like(logits.grad), atol=1e-12))

    def test_rejects_mismatched_or_empty_vectors(self) -> None:
        with self.assertRaises(ValueError):
            router_balance_auxiliary_loss(torch.empty(0), torch.empty(0))
        with self.assertRaises(ValueError):
            router_balance_auxiliary_loss(torch.tensor([1.0]), torch.tensor([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
