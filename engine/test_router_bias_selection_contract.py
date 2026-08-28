"""Contrato CPU: el sesgo de balanceo elige expertos sin contaminar probabilidades ni entropía."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.aethel_model import AethelConfig, SparseMoE


def main() -> None:
    torch.manual_seed(0)
    config = AethelConfig(
        dim=8,
        n_heads=2,
        n_kv_heads=1,
        n_experts=4,
        active_experts=2,
        require_triton=False,
    )
    router = SparseMoE(config).eval()
    with torch.no_grad():
        router.gate.weight.zero_()
        router.router_bias.copy_(torch.tensor([-2.0, 3.0, 2.0, -1.0]))

    router(torch.ones((1, 3, 8), dtype=torch.float32))

    # La selección debe obedecer al sesgo: expertos 1 y 2 reciben todas las rutas.
    assert router.last_load == [0.0, 50.0, 50.0, 0.0]
    # Con logits crudos uniformes la entropía densa se mantiene máxima (-1).
    assert torch.isclose(router.last_entropy_loss, torch.tensor(-1.0), atol=1e-6)
    print("ROUTER_BIAS_SELECTION_CPU_CONTRACT_OK")


if __name__ == "__main__":
    main()
