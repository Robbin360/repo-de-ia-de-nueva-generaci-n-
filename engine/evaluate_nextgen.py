"""Mide pérdida y perplejidad reales en el holdout JSONL BPE de Aethel NextGen."""
from __future__ import annotations

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def records(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("text"):
                yield row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True, help="Holdout JSONL o JSONL.GZ, nunca un benchmark retenido.")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-segments", type=int, default=256)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    corpus_path = Path(args.corpus)
    if any(name in corpus_path.name.lower() for name in ("mgsm", "belebele", "flores")):
        raise RuntimeError("Este evaluador de pérdida usa solo validation.jsonl; los benchmarks retenidos requieren evaluación específica.")
    from tokenizers import Tokenizer

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = NextGenConfig(**{key: value for key, value in checkpoint["config"].items() if key in NextGenConfig.__dataclass_fields__})
    model = AethelNextGen(config, memory_path=Path(args.checkpoint).with_name("evaluation_memory.jsonl")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    by_language: dict[str, list[float]] = defaultdict(list)
    total_losses: list[float] = []
    with torch.no_grad():
        for row in records(corpus_path):
            ids = tokenizer.encode(row["text"]).ids
            language = row.get("language", "unknown")
            for start in range(0, len(ids) - args.seq_len, args.seq_len):
                x = torch.tensor(ids[start : start + args.seq_len], dtype=torch.long, device=device).unsqueeze(0)
                y = torch.tensor(ids[start + 1 : start + args.seq_len + 1], dtype=torch.long, device=device).unsqueeze(0)
                _, loss, _ = model(x, y)
                value = float(loss.float().cpu())
                total_losses.append(value)
                by_language[language].append(value)
                if len(total_losses) >= args.max_segments:
                    break
            if len(total_losses) >= args.max_segments:
                break
    if not total_losses:
        raise RuntimeError("El holdout no produjo segmentos: confirme corpus, tokenizador y --seq-len.")
    mean_loss = sum(total_losses) / len(total_losses)
    result = {
        "checkpoint": args.checkpoint,
        "device": str(device),
        "segments": len(total_losses),
        "loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 20.0)),
        "by_language": {language: {"segments": len(values), "loss": sum(values) / len(values), "perplexity": math.exp(min(sum(values) / len(values), 20.0))} for language, values in by_language.items()},
        "split": "prepared_validation_holdout",
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
