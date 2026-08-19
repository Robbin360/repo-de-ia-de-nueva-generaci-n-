"""Prueba CPU de manifiesto de La Roca, candidato LoRA aislado y rollback."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig
from sleep_candidate import create_rock_manifest, create_sleep_candidate, rollback_candidate, verify_candidate_isolation, write_manifest_atomic


def config() -> NextGenConfig:
    return NextGenConfig(vocab_size=64, dim=32, layers=1, heads=4, kv_heads=2, experts=2, active_experts=1, max_seq_len=64, memory_slots=8, replay_capacity=8)


def run() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        rock = AethelNextGen(config(), root / "rock-episodic.jsonl")
        before = create_rock_manifest(rock)
        destination = write_manifest_atomic(before, root / "rock_manifest.json")
        stored = json.loads(destination.read_text(encoding="utf-8"))
        assert stored == before

        candidate = create_sleep_candidate(rock, root / "candidate-episodic.jsonl", rank=4, alpha=8.0, candidate_id="cpu-sleep-001")
        verified = verify_candidate_isolation(candidate, rock)
        assert verified["rock_state_sha256"] == before["rock_state_sha256"]
        assert candidate.manifest["training_started"] is False
        assert candidate.manifest["eligible_for_promotion"] is False
        assert candidate.manifest["holdout_access_enabled"] is False

        with torch.no_grad():
            next(parameter for name, parameter in candidate.model.named_parameters() if "lora_b" in name).add_(0.25)
        assert verify_candidate_isolation(candidate, rock)["rollback_ready"] is True
        rollback = rollback_candidate(candidate, rock)
        assert rollback["restored_reference"] is True and rollback["rock_changed"] is False
        assert create_rock_manifest(rock)["rock_state_sha256"] == before["rock_state_sha256"]

        tampered = create_sleep_candidate(rock, root / "tampered-episodic.jsonl", rank=4, alpha=8.0, candidate_id="cpu-sleep-tamper")
        with torch.no_grad():
            next(parameter for name, parameter in tampered.model.named_parameters() if ".base." in name).add_(0.25)
        try:
            verify_candidate_isolation(tampered, rock)
        except ValueError as error:
            assert "pesos base" in str(error)
        else:
            raise AssertionError("La verificación no detectó la mutación base del candidato")
    print("sleep_candidate_isolation OK")


if __name__ == "__main__":
    run()
