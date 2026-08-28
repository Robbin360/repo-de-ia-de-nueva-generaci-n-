"""Contrato CPU para el jitter de selección del router MoE."""
from __future__ import annotations

import torch

from aethel_model import AethelConfig, SparseMoE
from router_auxiliary import validate_router_jitter_noise


def main() -> None:
    assert validate_router_jitter_noise(0.0) == 0.0
    assert validate_router_jitter_noise("0.01") == 0.01
    for invalid in (-0.01, 1.0, float("inf")):
        try:
            validate_router_jitter_noise(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Se aceptó jitter inválido: {invalid!r}")

    config = AethelConfig(vocab_size=32, dim=8, n_experts=4, active_experts=2, router_jitter_noise=0.05)
    moe = SparseMoE(config)
    with torch.no_grad():
        moe.gate.weight.zero_()
    x = torch.zeros(1, 128, 8)
    torch.manual_seed(17)
    moe.train()
    _, _ = moe(x)
    assert moe.last_routing_stats["selection_jitter_noise"] == 0.05
    assert sum(value > 0.0 for value in moe.last_load) >= 3

    moe.eval()
    _, _ = moe(x)
    assert moe.last_routing_stats["selection_jitter_noise"] == 0.0
    print("ROUTER_JITTER_SELECTION_CONTRACT_OK")


if __name__ == "__main__":
    main()
