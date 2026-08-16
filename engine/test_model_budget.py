"""Comprueba el contador analítico contra una configuración Aethel pequeña real."""
from __future__ import annotations

from aethel_nextgen import AethelNextGen, NextGenConfig
from report_model_budget import report


def test_analytical_budget_matches_tiny_model() -> None:
    values = {
        "vocab_size": 64,
        "dim": 64,
        "layers": 1,
        "heads": 8,
        "kv_heads": 2,
        "experts": 2,
        "active_experts": 1,
        "max_seq_len": 32,
    }
    model = AethelNextGen(NextGenConfig(**values), memory_path="/tmp/aethel-budget-test.jsonl")
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert report("tiny", values)["parameters_total"] == actual


if __name__ == "__main__":
    test_analytical_budget_matches_tiny_model()
    print("OK")
