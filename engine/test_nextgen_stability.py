"""Pruebas deterministas para los mecanismos de estabilidad de Aethel NextGen."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from aethel_model import AethelConfig, SparseMoE
from aethel_nextgen import CicloDeSueno


def test_router_bias_opposes_overused_expert() -> None:
    moe = SparseMoE(AethelConfig(vocab_size=32, dim=32, n_layers=1, n_heads=4, n_kv_heads=1, n_experts=4, active_experts=1))
    with torch.no_grad():
        moe.gate.weight.zero_()
        moe.gate.weight[0].fill_(1.0)
    moe.train()
    _ = moe(torch.ones(2, 8, 32))
    assert moe.last_routing_stats["max_load"] > 0.9
    assert moe.router_bias[0].item() < 0.0


def test_sleep_keeps_diverse_replay() -> None:
    sleep = CicloDeSueno(capacity=3)
    state = torch.zeros(1, 8)
    sleep.consolidate(state, [1, 2, 3], priority=0.5)
    sleep.consolidate(state, [2, 2, 2], priority=0.9)
    sleep.consolidate(state, [3, 3, 3], priority=0.8)
    manifest = sleep.manifest()
    assert manifest["replay_records"] == 3
    assert manifest["unique_signatures"] == 3
    pairs = sleep.sample_pairs(seq_len=2, batch_size=2, device=torch.device("cpu"))
    assert pairs is not None
    assert pairs[0].shape == (2, 2)


if __name__ == "__main__":
    test_router_bias_opposes_overused_expert()
    test_sleep_keeps_diverse_replay()
    print("OK")
