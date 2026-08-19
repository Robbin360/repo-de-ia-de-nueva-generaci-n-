"""Prueba de memoria episódica/semántica y trazabilidad de razonamiento sin cadena de pensamiento."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, CuriosityController, CuriositySignals, NextGenConfig


def main() -> None:
    torch.manual_seed(7)
    controller = CuriosityController()
    blocked = controller.assess(CuriositySignals(uncertainty=0.8, novelty=0.8, contradiction=0.0, expected_progress=0.8, permitted=False))
    assert blocked.action == "blocked" and blocked.blocked and blocked.to_dict()["external_action_enabled"] is False
    noisy = controller.assess(CuriositySignals(uncertainty=0.95, novelty=0.95, contradiction=0.0, expected_progress=0.0, risk=0.1, cost=0.1))
    assert noisy.action != "propose_replay" and "incertidumbre_sin_progreso_demostrado" in noisy.reasons
    useful = controller.assess(CuriositySignals(uncertainty=0.9, novelty=1.0, contradiction=0.8, expected_progress=1.0, risk=0.0, cost=0.0))
    assert useful.action == "propose_replay" and useful.requires_approval
    assert controller.manifest()["external_action_enabled"] is False
    with tempfile.TemporaryDirectory(prefix="aethel-memory-") as directory:
        model = AethelNextGen(NextGenConfig(vocab_size=64, dim=32, layers=1, heads=4, kv_heads=1, experts=2, active_experts=1, max_seq_len=16, memory_slots=4, replay_capacity=8), Path(directory) / "episodic.jsonl")
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
        targets = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long)
        rock_before = model.rock.stable_projection.weight.detach().clone()
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
        assert torch.equal(rock_before, model.rock.stable_projection.weight.detach())
        assert model.liquid.manifest()["version"] == 1
        assert Path(model.liquid.manifest()["snapshot_path"]).is_file()
        assert metrics["curiosity"]["action"] != "propose_replay"
        assert metrics["curiosity"]["external_action_enabled"] is False
        assert manifest["curiosity"]["external_action_enabled"] is False
        print("memory_reasoning_trace OK")


if __name__ == "__main__":
    main()
