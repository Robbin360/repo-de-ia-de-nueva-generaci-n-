"""Prueba de memoria episódica/semántica y trazabilidad de razonamiento sin cadena de pensamiento."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def main() -> None:
    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="aethel-memory-") as directory:
        model = AethelNextGen(NextGenConfig(vocab_size=64, dim=32, layers=1, heads=4, kv_heads=1, experts=2, active_experts=1, max_seq_len=16, memory_slots=4, replay_capacity=8), Path(directory) / "episodic.jsonl")
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        _, loss, _ = model(tokens, targets)
        assert loss is not None and torch.isfinite(loss)
        model.observe(tokens, salience=0.8)
        _, _, metrics = model(tokens, targets)
        manifest = model.export_memory_manifest()
        assert manifest["episodic_records"] == 1
        assert manifest["semantic"]["semantic_records"] == 1
        trace = metrics["reasoning_trace"]
        assert trace["protocol"] == ["recuperación", "integración", "refinamiento presupuestado", "predicción"]
        assert trace["internal_chain_of_thought_exposed"] is False
        assert trace["episodic"]["selected"] == 1 and trace["semantic"]["selected"] == 1
        assert Path(manifest["semantic"]["path"]).is_file()
        print("memory_reasoning_trace OK")


if __name__ == "__main__":
    main()
