"""Comprueba guardado, reanudación y carga cruzada controlada entre baseline y ARC."""
from pathlib import Path
import tempfile

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def config(steps: int) -> NextGenConfig:
    return NextGenConfig(vocab_size=64, dim=32, layers=1, heads=4, kv_heads=1, experts=2, active_experts=1, max_seq_len=16, adaptive_refinement_steps=steps, adaptive_refinement_threshold=0.35)


def train_step(model: AethelNextGen, optimizer: torch.optim.Optimizer) -> None:
    tokens = torch.randint(0, 64, (2, 8))
    targets = torch.randint(0, 64, (2, 8))
    optimizer.zero_grad(set_to_none=True)
    _, loss, _ = model(tokens, targets)
    assert loss is not None and torch.isfinite(loss)
    loss.backward()
    optimizer.step()


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for name, steps in (("baseline", 0), ("arc", 2)):
            model = AethelNextGen(config(steps), root / f"{name}.jsonl")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            train_step(model, optimizer)
            checkpoint = root / f"{name}.pt"
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": vars(config(steps)), "step": 1}, checkpoint)
            restored = AethelNextGen(config(steps), root / f"{name}-restored.jsonl")
            restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            restored.load_state_dict(payload["model"], strict=True)
            restored_optimizer.load_state_dict(payload["optimizer"])
            train_step(restored, restored_optimizer)
        baseline_payload = torch.load(root / "baseline.pt", map_location="cpu", weights_only=False)
        arc_payload = torch.load(root / "arc.pt", map_location="cpu", weights_only=False)
        arc_from_base = AethelNextGen(config(2), root / "arc-from-base.jsonl")
        base_to_arc = arc_from_base.load_state_dict(baseline_payload["model"], strict=False)
        assert not base_to_arc.unexpected_keys
        assert base_to_arc.missing_keys and all(key.startswith("adaptive_refinement.") for key in base_to_arc.missing_keys)
        base_from_arc = AethelNextGen(config(0), root / "base-from-arc.jsonl")
        arc_to_base = base_from_arc.load_state_dict(arc_payload["model"], strict=False)
        assert not arc_to_base.missing_keys
        assert arc_to_base.unexpected_keys and all(key.startswith("adaptive_refinement.") for key in arc_to_base.unexpected_keys)
    print("OK: checkpoints ARC y baseline reanudables y compatibles de forma explícita")


if __name__ == "__main__":
    main()
