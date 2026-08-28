"""Contrato CPU de evaluación aislada para checkpoints Edge completos."""
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
for candidate in (ROOT / "training", ROOT / "engine"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from aethel_nextgen import AethelNextGen, NextGenConfig
from aethel_resume import RESUME_SCHEMA
from evaluate_edge_checkpoint import RECEIPT_NAME, STATUS_READY, evaluate_checkpoint


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        tokenizer_path = root / "tokenizer.json"
        tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "Aethel": 1, "responde": 2, "replies": 3, ":": 4, "uno": 5, "dos": 6}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.save(str(tokenizer_path))
        config = NextGenConfig(vocab_size=7, dim=16, layers=1, heads=4, kv_heads=1, experts=2, active_experts=1, max_seq_len=4, memory_slots=8, replay_capacity=8, router_aux_loss_weight=0.05, router_entropy_loss_weight=0.03, router_jitter_noise=0.01)
        model = AethelNextGen(config, memory_path=root / "build-memory" / "episodic.jsonl")
        manifest_path = root / "prepared_manifest.json"
        manifest_path.write_text('{"schema":"prepared"}\n', encoding="utf-8")
        validation_path = root / "validation.jsonl"
        validation_path.write_text(
            '{"text":"Aethel uno dos Aethel uno dos", "language":"en"}\n'
            '{"text":"Aethel uno dos Aethel uno dos", "language":"es"}\n', encoding="utf-8"
        )
        checkpoint_path = root / "latest.pt"
        config_dict = asdict(config)
        torch.save({
            "model": model.state_dict(), "reference_state": {}, "optimizer": {}, "scaler": {}, "rng_state": {}, "runtime_state": {}, "step": 12,
            "config": config_dict, "tokenizer": "/old/session/tokenizer.json", "tokenizer_sha256": digest(tokenizer_path),
            "resume_contract": {"schema": RESUME_SCHEMA, "config": config_dict, "tokenizer_sha256": digest(tokenizer_path), "corpus_layout": {}, "data_manifest_sha256": digest(manifest_path), "strategy": "single", "world_size": 1, "trainer_profile": {}},
        }, checkpoint_path)
        before = digest(checkpoint_path)
        output = root / "evaluation"
        report = evaluate_checkpoint(checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, data_manifest_path=manifest_path, validation_path=validation_path, output=output, requested_device="cpu", seq_len=4, max_segments_per_language=1, max_new_tokens=2)
        assert report["status"] == STATUS_READY
        assert report["step"] == 12 and report["segments"] == 2
        assert set(report["by_language"]) == {"en", "es"}
        assert report["parameter_fingerprint_before"] == report["parameter_fingerprint_after"]
        assert digest(checkpoint_path) == before
        assert (output / RECEIPT_NAME).is_file()
        assert json.loads((output / RECEIPT_NAME).read_text(encoding="utf-8"))["limits"]["training_started"] is False
    print("AETHEL_EDGE_EVALUATOR_CPU_VALIDATED")


if __name__ == "__main__":
    main()
