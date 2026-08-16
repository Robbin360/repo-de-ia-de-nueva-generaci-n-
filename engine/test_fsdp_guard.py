"""Comprueba que FSDP no se active accidentalmente en CPU o en un solo proceso."""
from __future__ import annotations

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig
from train_aethel_gpu import build_model


class Args:
    strategy = "fsdp"


def main() -> None:
    core = AethelNextGen(
        NextGenConfig(vocab_size=64, dim=64, layers=1, heads=8, kv_heads=2, experts=2, active_experts=1, max_seq_len=32),
        "/tmp/aethel-fsdp-guard.jsonl",
    )
    try:
        build_model(core, Args(), 1, torch.device("cpu"))
    except RuntimeError as error:
        assert "FSDP exige" in str(error)
        print("FSDP guard OK")
        return
    raise AssertionError("FSDP se activó sin varias GPU CUDA")


if __name__ == "__main__":
    main()
