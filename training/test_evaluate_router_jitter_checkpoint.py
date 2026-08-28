"""Contrato CPU del evaluador aislado de checkpoint jitter."""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = ROOT / "engine"
if str(ROOT / "training") not in sys.path:
    sys.path.insert(0, str(ROOT / "training"))
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from aethel_nextgen import AethelNextGen, NextGenConfig
from evaluate_router_jitter_checkpoint import RECEIPT_NAME, STATUS_READY, evaluate_checkpoint


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tokenizer(path: Path) -> None:
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Aethel": 1, "responde": 2, "replies": 3, ":": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(path))


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tokenizer_path = root / "tokenizer.json"
        _tokenizer(tokenizer_path)
        config = NextGenConfig(
            vocab_size=5,
            dim=16,
            layers=1,
            heads=4,
            kv_heads=1,
            experts=2,
            active_experts=1,
            max_seq_len=64,
            memory_slots=8,
            replay_capacity=8,
            router_aux_loss_weight=0.05,
            router_entropy_loss_weight=0.03,
            router_jitter_noise=0.01,
        )
        model = AethelNextGen(config, memory_path=root / "build_memory" / "episodic.jsonl")
        checkpoint_path = root / "latest.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": {},
                "step": 768,
                "config": asdict(config),
                "tokenizer": str(tokenizer_path),
                "tokenizer_sha256": _sha256(tokenizer_path),
                "strategy": "single",
            },
            checkpoint_path,
        )
        checkpoint_before = _sha256(checkpoint_path)
        output = root / "inference-check"
        report = evaluate_checkpoint(checkpoint_path, output, requested_device="cpu", max_new_tokens=2)
        assert report["status"] == STATUS_READY
        assert report["checkpoint_sha256_before"] == checkpoint_before == report["checkpoint_sha256_after"]
        assert report["parameter_fingerprint_before"] == report["parameter_fingerprint_after"]
        assert report["model_training"] is False
        assert len(report["generations"]) == 2
        assert all(item["finite_logits"] for item in report["generations"])
        assert all(value == 0.0 for item in report["generations"] for value in item["observed_router_selection_jitter_noise"])
        assert (output / RECEIPT_NAME).is_file()
        saved = json.loads((output / RECEIPT_NAME).read_text(encoding="utf-8"))
        assert saved["limits"]["training_started"] is False
        assert saved["limits"]["persistent_memory_written"] is False
        assert not (output / "volatile_memory" / "episodic_memory.jsonl").exists()
        try:
            evaluate_checkpoint(checkpoint_path, output, requested_device="cpu", max_new_tokens=2)
        except FileExistsError:
            pass
        else:
            raise AssertionError("El evaluador permitió reutilizar una salida existente.")
    print("AETHEL_ROUTER_JITTER_CHECKPOINT_EVALUATOR_CPU_VALIDATED")


if __name__ == "__main__":
    main()
