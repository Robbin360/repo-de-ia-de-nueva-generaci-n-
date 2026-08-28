"""Regresión CUDA de la actualización de memoria líquida durante E0.

Se puede ejecutar directamente. En un entorno sin CUDA se declara explícitamente
como omitida; el lanzador Kaggle ya exige CUDA y por tanto no permite omitirla.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def main() -> None:
    if not torch.cuda.is_available():
        print("SKIPPED_NO_CUDA")
        return

    torch.manual_seed(17)
    cuda_index = torch.cuda.current_device()
    device = torch.device("cuda", cuda_index)
    config = NextGenConfig(
        vocab_size=64,
        dim=32,
        layers=1,
        heads=4,
        kv_heads=1,
        experts=2,
        active_experts=1,
        max_seq_len=16,
        memory_slots=4,
        replay_capacity=8,
    )
    with tempfile.TemporaryDirectory(prefix="aethel-liquid-cuda-") as directory:
        memory_path = Path(directory) / "episodic.jsonl"
        model = AethelNextGen(config, memory_path).to(device)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long, device=device)
        targets = torch.tensor([[2, 3, 4, 5, 6, 7]], dtype=torch.long, device=device)
        memory_state_pointer = model.memory_state.data_ptr()
        # PyTorch materializa el buffer como `cuda:<índice>`; `torch.device("cuda")`
        # no es igual a ese objeto aunque ambos apunten a la GPU actual.
        assert model.memory_state.device.type == "cuda"
        assert model.memory_state.device.index == cuda_index
        _, loss, _ = model(tokens, targets)
        assert loss is not None and torch.isfinite(loss)
        model.observe(tokens, salience=0.8)

        manifest = model.export_memory_manifest()
        assert model.memory_state.device.type == "cuda"
        assert model.memory_state.device.index == cuda_index
        assert model.memory_state.data_ptr() == memory_state_pointer
        assert model.liquid.hebbian_trace.device == device
        assert manifest["episodic_records"] == 1
        assert manifest["semantic"]["semantic_records"] == 1
        assert Path(manifest["semantic"]["path"]).is_file()
        assert model.liquid.manifest()["version"] == 1
        print("VERIFIED_LIQUID_CUDA_ALIGNMENT")


if __name__ == "__main__":
    main()
