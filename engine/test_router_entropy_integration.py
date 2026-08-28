"""Prueba CPU de integración D1D; no usa Dataset, GPU, red ni checkpoints."""

import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.aethel_nextgen import AethelNextGen, NextGenConfig


def main() -> None:
    torch.manual_seed(17)
    with tempfile.TemporaryDirectory() as tmp:
        config = NextGenConfig(
            vocab_size=32,
            dim=32,
            layers=1,
            heads=4,
            kv_heads=2,
            experts=4,
            active_experts=2,
            max_seq_len=16,
            memory_slots=8,
            replay_capacity=16,
            router_aux_loss_weight=0.05,
            router_entropy_loss_weight=0.01,
        )
        model = AethelNextGen(config, memory_path=Path(tmp) / "memory.jsonl")
        model.train()
        tokens = torch.randint(0, config.vocab_size, (2, 8))
        targets = torch.randint(0, config.vocab_size, (2, 8))

        logits, loss, metrics = model(tokens, targets=targets)
        assert loss is not None
        assert model.core.last_router_entropy_loss.requires_grad
        assert metrics["router_entropy_loss_weight"] == 0.01
        assert "router_entropy_loss" in metrics

        ce_loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        expected = ce_loss + torch.tensor(
            config.router_aux_loss_weight * metrics["aux_loss"]
            + config.router_entropy_loss_weight * metrics["router_entropy_loss"]
        )
        assert torch.isclose(loss.detach(), expected, atol=1e-5)

    disabled = NextGenConfig(router_entropy_loss_weight=0.0)
    assert disabled.router_entropy_loss_weight == 0.0
    print("ROUTER_ENTROPY_INTEGRATION_CPU_CONTRACT_OK")


if __name__ == "__main__":
    main()
