"""Regresión del refinamiento adaptativo: estados seleccionados, presupuesto y compatibilidad base."""
from pathlib import Path
import tempfile

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def make_config(steps: int) -> NextGenConfig:
    return NextGenConfig(vocab_size=64, dim=32, layers=1, heads=4, kv_heads=1, experts=2, active_experts=1, max_seq_len=16, adaptive_refinement_steps=steps, adaptive_refinement_threshold=0.5, adaptive_compute_penalty=0.01)


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        baseline = AethelNextGen(make_config(0), Path(directory) / "base.jsonl")
        assert baseline.adaptive_refinement is None
        model = AethelNextGen(make_config(2), Path(directory) / "adaptive.jsonl")
        assert model.adaptive_refinement is not None
        with torch.no_grad():
            model.adaptive_refinement.difficulty.weight.zero_()
            model.adaptive_refinement.difficulty.bias.fill_(10.0)
        tokens = torch.randint(0, 64, (3, 8))
        targets = torch.randint(0, 64, (3, 8))
        _, loss, metrics = model(tokens, targets)
        assert loss is not None and torch.isfinite(loss)
        adaptive = metrics["adaptive_compute"]
        assert adaptive["selected"] == 3
        assert adaptive["effective_token_steps"] == 6
        assert metrics["reasoning_trace"]["protocol"][-2] == "refinamiento presupuestado"
        loss.backward()
        assert any(parameter.grad is not None for parameter in model.adaptive_refinement.parameters())
    print("OK: refinamiento adaptativo presupuestado")


if __name__ == "__main__":
    main()
