"""Entrenamiento GPU reanudable de Aethel con tokenizador BPE y soporte torchrun/DDP.

Ejemplo de una GPU:
  python engine/train_aethel_gpu.py --corpus-dir /data/aethel --tokenizer /data/tokenizer.json --output /data/runs/aethel-100m

Ejemplo de varias GPU:
  torchrun --standalone --nproc_per_node=8 engine/train_aethel_gpu.py --corpus-dir /data/aethel --tokenizer /data/tokenizer.json --output /data/runs/aethel-1b
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from aethel_nextgen import AethelNextGen, NextGenConfig


def distributed_setup() -> tuple[int, int, torch.device]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, torch.device("cuda", local_rank)
    return rank, world_size, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def corpus_records(corpus_dir: Path):
    for path in sorted(corpus_dir.glob("train-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line).get("text")
                if text:
                    yield text


def token_batches(corpus_dir: Path, tokenizer_path: Path, seq_len: int, batch_size: int, rank: int, world_size: int):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bucket: list[torch.Tensor] = []
    for index, text in enumerate(corpus_records(corpus_dir)):
        if index % world_size != rank:
            continue
        ids = tokenizer.encode(text).ids
        if len(ids) < seq_len + 1:
            continue
        for start in range(0, len(ids) - seq_len - 1, seq_len):
            segment = torch.tensor(ids[start : start + seq_len + 1], dtype=torch.long)
            bucket.append(segment)
            if len(bucket) == batch_size:
                stacked = torch.stack(bucket)
                bucket = []
                yield stacked[:, :-1], stacked[:, 1:]


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def run(args: argparse.Namespace) -> None:
    rank, world_size, device = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = NextGenConfig(vocab_size=tokenizer.get_vocab_size(), dim=args.dim, layers=args.layers, heads=args.heads, kv_heads=args.kv_heads, experts=args.experts, active_experts=args.active_experts, max_seq_len=args.seq_len, memory_slots=args.memory_slots, replay_capacity=args.replay_capacity)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    core = AethelNextGen(config, output / f"episodic_rank_{rank}.jsonl").to(device)
    model = DDP(core, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=True) if world_size > 1 else core
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    start_step = 0
    checkpoint_path = output / "latest.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        target = model.module if isinstance(model, DDP) else model
        target.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    target = model.module if isinstance(model, DDP) else model
    reference = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}
    batches = token_batches(Path(args.corpus_dir), Path(args.tokenizer), args.seq_len, args.batch_size, rank, world_size)
    metrics_path = output / f"metrics_rank_{rank}.jsonl"
    started = time.time()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for step in range(start_step + 1, args.max_steps + 1):
            try:
                x, y = next(batches)
            except StopIteration:
                batches = token_batches(Path(args.corpus_dir), Path(args.tokenizer), args.seq_len, args.batch_size, rank, world_size)
                x, y = next(batches)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            autocast_device = "cuda" if device.type == "cuda" else "cpu"
            dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
            with torch.autocast(device_type=autocast_device, dtype=dtype, enabled=device.type == "cuda" and args.precision != "fp32"):
                _, loss, runtime = model(x, y)
                total_loss = loss + target.regularization_loss(reference, args.replay_regularization)
                total_loss = total_loss / args.gradient_accumulation
            scaler.scale(total_loss).backward()
            if step % args.gradient_accumulation == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % args.observe_every == 0:
                target.observe(x[:1], salience=float(loss.detach().float().cpu()))
            if world_size > 1:
                dist.broadcast(target.liquid.hebbian_trace, src=0)
                dist.broadcast(target.memory_state, src=0)
            if rank == 0:
                elapsed = max(1e-6, time.time() - started)
                event = {"step": step, "loss": float(loss.detach().float().cpu()), "total_loss": float(total_loss.detach().float().cpu() * args.gradient_accumulation), "tokens_per_second": (step * args.batch_size * args.seq_len * world_size) / elapsed, "world_size": world_size, "device": str(device), "runtime": runtime, "memory": target.export_memory_manifest(), "experts": list(target.core.last_expert_loads), "config": asdict(config)}
                metrics.write(json.dumps(event, ensure_ascii=False) + "\n")
                metrics.flush()
                print(json.dumps(event, ensure_ascii=False), flush=True)
                if step % args.save_every == 0 or step == args.max_steps:
                    atomic_save({"model": target.state_dict(), "optimizer": optimizer.state_dict(), "step": step, "config": asdict(config), "event": event, "tokenizer": str(Path(args.tokenizer).resolve())}, checkpoint_path)
                    atomic_save({"model": target.state_dict(), "step": step, "config": asdict(config)}, output / f"step_{step:08d}.pt")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=16)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--experts", type=int, default=8)
    parser.add_argument("--active-experts", type=int, default=2)
    parser.add_argument("--memory-slots", type=int, default=512)
    parser.add_argument("--replay-capacity", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--replay-regularization", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--observe-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
