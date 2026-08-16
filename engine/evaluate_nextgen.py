"""Evaluación real de pérdida y perplexidad sobre el tramo final de un corpus."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig
from train_nextgen import byte_tokens


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = NextGenConfig(**{key: value for key, value in checkpoint["config"].items() if key in NextGenConfig.__dataclass_fields__})
    model = AethelNextGen(config, memory_path=Path(args.checkpoint).with_name("evaluation_memory.jsonl")).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    tokens = byte_tokens(Path(args.corpus).read_text(encoding="utf-8"))
    holdout = tokens[int(len(tokens) * 0.9) :]
    losses = []
    with torch.no_grad():
        for start in range(0, len(holdout) - args.seq_len - 1, args.seq_len):
            x = holdout[start : start + args.seq_len].unsqueeze(0).to(device)
            y = holdout[start + 1 : start + args.seq_len + 1].unsqueeze(0).to(device)
            _, loss, _ = model(x, y)
            losses.append(float(loss.cpu()))
    mean_loss = sum(losses) / len(losses)
    result = {"checkpoint": args.checkpoint, "device": str(device), "examples": len(losses), "loss": mean_loss, "perplexity": math.exp(min(mean_loss, 20.0)), "split": "last_10_percent_real_corpus"}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
