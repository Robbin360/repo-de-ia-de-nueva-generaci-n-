"""Valida LoRA sin modificar pesos base y mide parámetros entrenables."""
from __future__ import annotations

import tempfile

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def main() -> None:
    config = NextGenConfig(vocab_size=64, dim=32, layers=1, heads=4, kv_heads=1, experts=2, active_experts=1, max_seq_len=16)
    with tempfile.TemporaryDirectory() as temporary:
        model = AethelNextGen(config, f"{temporary}/episodic.jsonl")
        baseline = sum(parameter.numel() for parameter in model.parameters())
        stats = model.enable_lora(rank=4, alpha=8.0, freeze_base=True)
        assert 0 < stats["parameters_trainable"] < baseline
        x = torch.randint(0, config.vocab_size, (1, 8))
        _, loss, _ = model(x, x)
        assert loss is not None
        loss.backward()
        assert all(parameter.grad is None for name, parameter in model.named_parameters() if ".base." in name)
        assert any(parameter.grad is not None for name, parameter in model.named_parameters() if "lora_" in name)
        print({"lora_adapters": "OK", **stats})


if __name__ == "__main__":
    main()
