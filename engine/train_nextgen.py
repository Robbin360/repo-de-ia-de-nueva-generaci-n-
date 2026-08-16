"""Entrenamiento real de Aethel NextGen sobre un corpus local proporcionado."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def byte_tokens(text: str) -> torch.Tensor:
    data = text.encode("utf-8")
    if len(data) < 2:
        raise ValueError("El corpus debe contener al menos dos bytes")
    return torch.tensor(list(data), dtype=torch.long)


def batch_from_corpus(tokens: torch.Tensor, seq_len: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if len(tokens) <= seq_len + 1:
        raise ValueError("El corpus es menor que seq_len + 1")
    starts = [random.randrange(0, len(tokens) - seq_len - 1) for _ in range(batch_size)]
    x = torch.stack([tokens[start : start + seq_len] for start in starts]).to(device)
    y = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts]).to(device)
    return x, y


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    corpus_path = Path(args.corpus)
    text = corpus_path.read_text(encoding="utf-8")
    tokens = byte_tokens(text)
    config = NextGenConfig(dim=args.dim, layers=args.layers, heads=args.heads, kv_heads=args.kv_heads, experts=args.experts, max_seq_len=args.seq_len)
    model = AethelNextGen(config, args.memory_path).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    ema_reference = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = checkpoint_dir / "nextgen_metrics.jsonl"
    start = time.time()
    model.train()

    with metrics_path.open("a", encoding="utf-8") as metrics:
        for step in range(1, args.steps + 1):
            x, y = batch_from_corpus(tokens, args.seq_len, args.batch_size, device)
            optimizer.zero_grad(set_to_none=True)
            _, loss, runtime = model(x, y)
            total_loss = loss + model.regularization_loss(reference, args.replay_regularization)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    ema_reference[name].mul_(args.ema_decay).add_(parameter.detach(), alpha=1.0 - args.ema_decay)
            if step % args.observe_every == 0:
                model.observe(x[:1], salience=float(loss.detach().cpu()))
            ema_drift = sum((parameter.detach() - ema_reference[name]).float().pow(2).mean() for name, parameter in model.named_parameters()).sqrt()
            routing = [layer.feed_forward.last_routing_stats for layer in model.core.layers]
            event = {"step": step, "loss": float(loss.detach().cpu()), "total_loss": float(total_loss.detach().cpu()), "elapsed_s": time.time() - start, "device": str(device), "parameters": sum(p.numel() for p in model.parameters()), "memory": model.export_memory_manifest(), "runtime": runtime, "experts": list(model.core.last_expert_loads), "routing": routing, "ema_drift": float(ema_drift.detach().cpu())}
            metrics.write(json.dumps(event, ensure_ascii=False) + "\n")
            metrics.flush()
            if step % args.save_every == 0 or step == args.steps:
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "config": vars(config), "step": step, "event": event}, checkpoint_dir / f"nextgen_step_{step}.pt")
            print(json.dumps(event, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--checkpoint-dir", default="engine/artifacts/nextgen")
    parser.add_argument("--memory-path", default="engine/artifacts/nextgen/episodic_memory.jsonl")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--dim", type=int, default=384)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--replay-regularization", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--observe-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default=None)
    run(parser.parse_args())
