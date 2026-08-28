"""Prueba ejecutable del inspector de checkpoints de Kaggle."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from inspect_checkpoint import inspect_checkpoint


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    packaged = root / "packaged.pt"
    raw = root / "legacy_raw.pth"
    state = {"core.embedding.weight": torch.zeros(4, 3)}
    torch.save(
        {
            "model": state,
            "config": {"dim": 3},
            "step": 7,
            "tokenizer": "/data/tokenizer.json",
            "optimizer": {},
            "scaler": {},
            "rng_state": {},
            "runtime_state": {},
            "resume_contract": {},
        },
        packaged,
    )
    torch.save(state, raw)

    report = inspect_checkpoint(packaged, require_reproducible=True)
    assert report["reproducible_resume"] is True
    assert report["faithful_resume_missing"] == []
    assert report["parameter_count"] == 12
    assert report["step"] == 7

    raw_report = inspect_checkpoint(raw)
    assert raw_report["origin"] == "raw_state_dict"
    try:
        inspect_checkpoint(raw, require_reproducible=True)
    except ValueError as error:
        assert "No se permite reanudar" in str(error)
    else:
        raise AssertionError("Un state_dict histórico crudo no debe ser reanudable.")

print("checkpoint inspection passed")
