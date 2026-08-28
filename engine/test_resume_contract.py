from __future__ import annotations

import random
import tempfile
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig
from aethel_resume import (
    build_resume_contract,
    capture_rng_state,
    require_full_resume_payload,
    restore_rng_state,
    validate_resume_contract,
)


def _model() -> AethelNextGen:
    return AethelNextGen(
        NextGenConfig(
            vocab_size=32,
            dim=16,
            layers=1,
            heads=4,
            kv_heads=2,
            experts=2,
            active_experts=1,
            max_seq_len=16,
            memory_slots=4,
            replay_capacity=4,
        )
    )


def test_runtime_state_roundtrip() -> None:
    source = _model()
    source.memory_state.fill_(0.25)
    source.liquid.hebbian_trace.fill_(0.5)
    source.liquid.version = 7
    source.liquid.curiosity_events = 3
    source.sleep.consolidation_step = 2
    source.curiosity.uncertainty_by_context = {"unit": 0.4}
    payload = source.export_resume_runtime_state()

    restored = _model()
    restored.restore_resume_runtime_state(payload)
    assert torch.equal(restored.memory_state, source.memory_state)
    assert torch.equal(restored.liquid.hebbian_trace, source.liquid.hebbian_trace)
    assert restored.liquid.version == 7
    assert restored.liquid.curiosity_events == 3
    assert restored.sleep.consolidation_step == 2
    assert restored.curiosity.uncertainty_by_context == {"unit": 0.4}


def test_memory_transition_commits_after_backward_boundary() -> None:
    model = _model()
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    _, loss, _ = model(tokens, targets)
    loss.backward()
    before = model.memory_state.clone()
    model.commit_memory_state()
    assert not torch.equal(model.memory_state, before)


def test_resume_contract_and_rng_roundtrip() -> None:
    with tempfile.TemporaryDirectory() as directory:
        corpus = Path(directory)
        (corpus / "train-es.jsonl").write_text('{"text":"prueba"}\n', encoding="utf-8")
        payload = build_resume_contract(
            config={"dim": 16},
            tokenizer_sha256="tokenizer",
            corpus_dir=corpus,
            data_manifest_sha256="manifest",
            strategy="single",
            world_size=1,
            trainer_profile={"precision": "fp32", "schedule_total_steps": 10},
        )
        validate_resume_contract(payload, payload)
        altered = {**payload, "trainer_profile": {"precision": "bf16", "schedule_total_steps": 10}}
        try:
            validate_resume_contract(payload, altered)
            raise AssertionError("El cambio de precisión debe bloquear la reanudación fiel.")
        except ValueError:
            pass

        altered_horizon = {**payload, "trainer_profile": {"precision": "fp32", "schedule_total_steps": 12}}
        try:
            validate_resume_contract(payload, altered_horizon)
            raise AssertionError("El cambio de horizonte global debe bloquear la reanudación fiel.")
        except ValueError:
            pass

    random.seed(17)
    torch.manual_seed(17)
    rng_state = capture_rng_state(torch.device("cpu"))
    expected_python = random.random()
    expected_torch = torch.rand(1)
    random.random()
    torch.rand(1)
    restore_rng_state(rng_state, torch.device("cpu"))
    assert random.random() == expected_python
    assert torch.equal(torch.rand(1), expected_torch)


def test_full_payload_rejects_legacy_checkpoint() -> None:
    try:
        require_full_resume_payload({"model": {}, "step": 1})
        raise AssertionError("Un checkpoint de pesos sin optimizador/RNG debe rechazarse para reanudación fiel.")
    except ValueError as error:
        assert "optimizer" in str(error)

    try:
        require_full_resume_payload(
            {
                "model": {},
                "optimizer": {},
                "scaler": {},
                "rng_state": {},
                "runtime_state": {},
                "resume_contract": {},
                "step": 1,
            }
        )
        raise AssertionError("La reanudación debe exigir el estado de referencia de regularización.")
    except ValueError as error:
        assert "reference_state" in str(error)


if __name__ == "__main__":
    test_runtime_state_roundtrip()
    test_memory_transition_commits_after_backward_boundary()
    test_resume_contract_and_rng_roundtrip()
    test_full_payload_rejects_legacy_checkpoint()
    print("AETHEL_RESUME_CONTRACT_VERIFIED")
