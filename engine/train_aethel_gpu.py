"""Entrenamiento GPU reanudable de Aethel con tokenizador BPE y soporte torchrun/DDP/FSDP.

Ejemplo de una GPU:
  python engine/train_aethel_gpu.py --corpus-dir /data/aethel --tokenizer /data/tokenizer.json --output /data/runs/aethel-100m

Ejemplo de varias GPU:
  torchrun --standalone --nproc_per_node=8 engine/train_aethel_gpu.py --strategy fsdp --corpus-dir /data/aethel --tokenizer /data/tokenizer.json --output /data/runs/aethel-1b
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import random
import shutil
import signal
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
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend)
        if torch.cuda.is_available():
            local_rank = int(os.environ["LOCAL_RANK"])
            torch.cuda.set_device(local_rank)
            return rank, world_size, torch.device("cuda", local_rank)
        return rank, world_size, torch.device("cpu")
    return rank, world_size, torch.device("cuda" if torch.cuda.is_available() else "cpu")


def corpus_records(corpus_dir: Path):
    for path in sorted(corpus_dir.glob("train-*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                text = json.loads(line).get("text")
                if text:
                    yield text


def token_batches(corpus_dir: Path, tokenizer_path: Path, seq_len: int, batch_size: int, rank: int, world_size: int, skip_batches: int = 0):
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    bucket: list[torch.Tensor] = []
    emitted = 0
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
                if emitted < skip_batches:
                    emitted += 1
                    continue
                emitted += 1
                yield stacked[:, :-1], stacked[:, 1:]


def atomic_save(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def prune_portable_snapshots(output: Path, keep_snapshots: int) -> list[str]:
    """Conserva los snapshots portátiles más recientes sin tocar ``latest.pt``.

    ``latest.pt`` es el punto de reanudación completo con optimizador. Los
    ``step_*.pt`` son copias de contingencia sin optimizador y se acotan para
    que una sesión larga de Kaggle no consuma el disco de salida.
    """
    if keep_snapshots < 1:
        raise ValueError("keep_snapshots debe ser al menos 1 para conservar una contingencia portable.")
    snapshots = sorted(output.glob("step_*.pt"))
    for obsolete in snapshots[:-keep_snapshots]:
        obsolete.unlink()
    return [path.name for path in snapshots[-keep_snapshots:]]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_tokenizer(tokenizer_path: Path, output: Path) -> tuple[Path, str]:
    """Copia el tokenizador junto al peso para que el checkpoint sea portátil."""
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "tokenizer.json"
    source_hash = sha256_file(tokenizer_path)
    if not destination.exists() or sha256_file(destination) != source_hash:
        shutil.copy2(tokenizer_path, destination)
    return destination, source_hash


def validate_resume_metadata(checkpoint: dict, config: dict, tokenizer_hash: str) -> None:
    """Impide reanudar con una topología o tokenizador distinto del checkpoint."""
    if checkpoint.get("config") != config:
        raise ValueError("El checkpoint no coincide con la configuración activa; la reanudación queda bloqueada.")
    if checkpoint.get("tokenizer_sha256") != tokenizer_hash:
        raise ValueError("El tokenizador activo no coincide con el hash del checkpoint; la reanudación queda bloqueada.")


def learning_rate_at_step(step: int, max_steps: int, peak: float, minimum: float, warmup_steps: int) -> float:
    """Warmup lineal y decaimiento coseno; el valor queda registrado por paso."""
    if warmup_steps and step <= warmup_steps:
        return peak * step / warmup_steps
    progress = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)))
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


def build_model(core: AethelNextGen, args: argparse.Namespace, world_size: int, device: torch.device):
    """Envuelve el núcleo sin imponer sharding a los pilotos de una GPU."""
    if args.strategy == "fsdp":
        if world_size < 2 or device.type != "cuda":
            raise RuntimeError("FSDP exige torchrun con al menos dos GPU CUDA; use --strategy single o ddp fuera de ese caso.")
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(core, device_id=device, use_orig_params=True, sync_module_states=True)
    if args.strategy == "ddp":
        if world_size < 2:
            raise RuntimeError("DDP exige torchrun con al menos dos procesos.")
        return DDP(core, device_ids=[device.index] if device.type == "cuda" else None, broadcast_buffers=True)
    if world_size > 1:
        raise RuntimeError("torchrun detectó varios procesos: seleccione --strategy ddp o --strategy fsdp.")
    return core


def unwrap_model(model: torch.nn.Module) -> AethelNextGen:
    return model.module if isinstance(model, DDP) or hasattr(model, "module") else model  # type: ignore[return-value]


def fsdp_checkpoint_state(model: torch.nn.Module, optimizer: torch.optim.Optimizer, strategy: str) -> tuple[dict, dict | None]:
    """Obtiene estado completo en rango 0, con participación de todos los rangos FSDP."""
    if strategy != "fsdp":
        return model.state_dict(), optimizer.state_dict()
    from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

    config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, config):
        model_state = model.state_dict()
    optimizer_state = FSDP.full_optim_state_dict(model, optimizer, rank0_only=True)
    return model_state, optimizer_state


def run(args: argparse.Namespace) -> None:
    rank, world_size, device = distributed_setup()
    random.seed(args.seed + rank)
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed + rank)
        torch.backends.cuda.matmul.allow_tf32 = True

    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = NextGenConfig(vocab_size=tokenizer.get_vocab_size(), dim=args.dim, layers=args.layers, heads=args.heads, kv_heads=args.kv_heads, experts=args.experts, active_experts=args.active_experts, max_seq_len=args.seq_len, memory_slots=args.memory_slots, replay_capacity=args.replay_capacity, router_bias_step=args.router_bias_step, router_bias_limit=args.router_bias_limit, lora_rank=args.lora_rank, lora_alpha=args.lora_alpha, lora_freeze_base=not args.lora_train_base, require_triton=device.type == "cuda" and not args.allow_pytorch_fallback)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer_artifact = output / "tokenizer.json"
    if rank == 0:
        tokenizer_artifact, tokenizer_hash = snapshot_tokenizer(Path(args.tokenizer), output)
    else:
        tokenizer_hash = ""
    if world_size > 1:
        dist.barrier()
    tokenizer_hash = sha256_file(tokenizer_artifact)
    core = AethelNextGen(config, output / f"episodic_rank_{rank}.jsonl").to(device)
    model = build_model(core, args, world_size, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.precision == "fp16")
    start_step = 0
    checkpoint_path = output / "latest.pt"
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        validate_resume_metadata(checkpoint, asdict(config), tokenizer_hash)
        target = unwrap_model(model)
        if args.strategy == "fsdp":
            from torch.distributed.fsdp import FullStateDictConfig, FullyShardedDataParallel as FSDP, StateDictType

            state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_config):
                model.load_state_dict(checkpoint["model"])
            optimizer_state = FSDP.scatter_full_optim_state_dict(checkpoint["optimizer"], model)
            optimizer.load_state_dict(optimizer_state)
        else:
            target.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])

    target = unwrap_model(model)
    reference = {name: parameter.detach().clone() for name, parameter in target.named_parameters()}
    batches = token_batches(Path(args.corpus_dir), Path(args.tokenizer), args.seq_len, args.batch_size, rank, world_size, skip_batches=start_step)
    metrics_path = output / f"metrics_rank_{rank}.jsonl"
    started = time.time()
    requested_signal: int | None = None

    def request_graceful_stop(signum: int, _frame) -> None:
        nonlocal requested_signal
        requested_signal = signum

    signal.signal(signal.SIGTERM, request_graceful_stop)
    signal.signal(signal.SIGINT, request_graceful_stop)

    def save_recoverable_checkpoint(step: int, event: dict | None, reason: str) -> None:
        """Sólo se llama después de un paso de optimizador, por lo que es reanudable."""
        model_state, optimizer_state = fsdp_checkpoint_state(model, optimizer, args.strategy)
        if rank == 0:
            payload = {
                "model": model_state,
                "optimizer": optimizer_state,
                "step": step,
                "config": asdict(config),
                "event": event,
                "tokenizer": str(tokenizer_artifact.resolve()),
                "tokenizer_sha256": tokenizer_hash,
                "strategy": args.strategy,
                "checkpoint_reason": reason,
            }
            atomic_save(payload, checkpoint_path)
            atomic_save({key: value for key, value in payload.items() if key != "optimizer"}, output / f"step_{step:08d}.pt")
            retained_snapshots = prune_portable_snapshots(output, args.keep_snapshots)
            receipt = {
                "latest": checkpoint_path.name,
                "step": step,
                "reason": reason,
                "tokenizer_sha256": tokenizer_hash,
                "retained_snapshots": retained_snapshots,
                "resume_contract": "Importe la salida comprometida como Dataset privado y fije AETHEL_RESUME_CHECKPOINT a latest.pt tras inspección.",
            }
            temporary = output / "recovery_receipt.json.tmp"
            temporary.write_text(json.dumps(receipt, ensure_ascii=False) + "\n", encoding="utf-8")
            temporary.replace(output / "recovery_receipt.json")
        if world_size > 1:
            dist.barrier()

    last_checkpoint_step = start_step
    model.train()
    optimizer.zero_grad(set_to_none=True)
    with metrics_path.open("a", encoding="utf-8") as metrics:
        for step in range(start_step + 1, args.max_steps + 1):
            current_lr = learning_rate_at_step(step, args.max_steps, args.learning_rate, args.min_learning_rate, args.warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
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
                replay_loss = None
                if step % args.replay_every == 0:
                    replay_pairs = target.sleep.sample_pairs(args.seq_len, min(args.batch_size, args.replay_batch_size), device)
                    if replay_pairs is not None:
                        replay_x, replay_y = replay_pairs
                        _, replay_loss, _ = model(replay_x, replay_y)
                        total_loss = total_loss + args.replay_loss_weight * replay_loss
                total_loss = total_loss / args.gradient_accumulation
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Pérdida no finita en el paso {step}; se preserva el último checkpoint atómico y se detiene la corrida.")
            scaler.scale(total_loss).backward()
            optimizer_updated = step % args.gradient_accumulation == 0
            if optimizer_updated:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if step % args.observe_every == 0:
                target.observe(torch.cat([x[:1], y[:1, -1:]], dim=1), salience=float(loss.detach().float().cpu()))
            if world_size > 1:
                dist.broadcast(target.liquid.hebbian_trace, src=0)
                dist.broadcast(target.memory_state, src=0)
            if rank == 0:
                elapsed = max(1e-6, time.time() - started)
                routing = [layer.feed_forward.last_routing_stats for layer in target.core.layers]
                max_imbalance = max((item["imbalance"] for item in routing), default=0.0)
                min_entropy = min((item["entropy"] for item in routing), default=1.0)
                healthy = max_imbalance <= args.max_router_imbalance and min_entropy >= args.min_router_entropy
                event = {"step": step, "loss": float(loss.detach().float().cpu()), "replay_loss": float(replay_loss.detach().float().cpu()) if replay_loss is not None else None, "total_loss": float(total_loss.detach().float().cpu() * args.gradient_accumulation), "learning_rate": current_lr, "tokens_per_second": (step * args.batch_size * args.seq_len * world_size) / elapsed, "tokens_seen": step * args.batch_size * args.seq_len * world_size, "world_size": world_size, "device": str(device), "runtime": runtime, "memory": target.export_memory_manifest(), "experts": list(target.core.last_expert_loads), "routing": routing, "router_health": {"healthy": healthy, "max_imbalance": max_imbalance, "min_entropy": min_entropy}, "adaptation": target.lora_config, "parameters_trainable": sum(parameter.numel() for parameter in target.parameters() if parameter.requires_grad), "config": asdict(config)}
                metrics.write(json.dumps(event, ensure_ascii=False) + "\n")
                metrics.flush()
                print(json.dumps(event, ensure_ascii=False), flush=True)
            due = optimizer_updated and (step - last_checkpoint_step >= args.save_every or step == args.max_steps or requested_signal is not None)
            if due:
                reason = f"signal-{requested_signal}" if requested_signal is not None else ("final" if step == args.max_steps else "periodic")
                save_recoverable_checkpoint(step, event if rank == 0 else None, reason)
                last_checkpoint_step = step
            if requested_signal is not None:
                if rank == 0 and not due:
                    marker = {"signal": requested_signal, "last_recoverable_step": last_checkpoint_step}
                    (output / "interrupted_before_safe_boundary.json").write_text(json.dumps(marker, ensure_ascii=False) + "\n", encoding="utf-8")
                break
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
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--replay-regularization", type=float, default=1e-5)
    parser.add_argument("--replay-loss-weight", type=float, default=0.10)
    parser.add_argument("--replay-every", type=int, default=200)
    parser.add_argument("--replay-batch-size", type=int, default=1)
    parser.add_argument("--router-bias-step", type=float, default=0.05)
    parser.add_argument("--router-bias-limit", type=float, default=0.5)
    parser.add_argument("--lora-rank", type=int, default=0, help="Rango LoRA opcional; 0 mantiene ajuste completo.")
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-train-base", action="store_true", help="Mantiene entrenables los pesos base además de LoRA.")
    parser.add_argument("--allow-pytorch-fallback", action="store_true", help="Sólo laboratorio: permite operadores PyTorch en GPU si Triton no está listo.")
    parser.add_argument("--max-router-imbalance", type=float, default=0.30)
    parser.add_argument("--min-router-entropy", type=float, default=0.50)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--observe-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-snapshots", type=int, default=3, help="Número de snapshots step_*.pt portátiles que se conservan junto a latest.pt.")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--strategy", choices=["single", "ddp", "fsdp"], default="single")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--resume", action="store_true")
    run(parser.parse_args())
