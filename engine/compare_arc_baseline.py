"""Comparación reproducible ARC vs baseline sobre un corpus real; no genera puntuaciones sintéticas."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import random
import resource
import tempfile
import time
from pathlib import Path

import torch

from aethel_nextgen import AethelNextGen, NextGenConfig


def byte_tokens(text: str) -> torch.Tensor:
    data = text.encode("utf-8")
    if len(data) < 3:
        raise ValueError("El corpus debe contener al menos tres bytes")
    return torch.tensor(list(data), dtype=torch.long)


def fixed_batches(tokens: torch.Tensor, steps: int, batch_size: int, seq_len: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    maximum = len(tokens) - seq_len - 1
    if maximum < 1:
        raise ValueError("El corpus es menor que seq_len + 1")
    batches = []
    for _ in range(steps):
        starts = torch.randint(0, maximum, (batch_size,), generator=generator)
        x = torch.stack([tokens[int(start) : int(start) + seq_len] for start in starts])
        y = torch.stack([tokens[int(start) + 1 : int(start) + seq_len + 1] for start in starts])
        batches.append((x, y))
    return batches


def train_variant(name: str, config: NextGenConfig, batches: list[tuple[torch.Tensor, torch.Tensor]], memory_path: Path, seed: int) -> dict:
    random.seed(seed)
    torch.manual_seed(seed)
    model = AethelNextGen(config, memory_path)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    losses: list[float] = []
    fractions: list[float] = []
    effective_steps: list[int] = []
    imbalances: list[float] = []
    started = time.perf_counter()
    model.train()
    for x, y in batches:
        optimizer.zero_grad(set_to_none=True)
        _, loss, runtime = model(x, y)
        assert loss is not None
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
        adaptive = runtime["adaptive_compute"]
        fractions.append(float(adaptive["fraction"]))
        effective_steps.append(int(adaptive["effective_token_steps"]))
        routing = [layer.feed_forward.last_routing_stats for layer in model.core.layers]
        imbalances.append(sum(float(layer["imbalance"]) for layer in routing) / max(1, len(routing)))
    elapsed = time.perf_counter() - started
    return {
        "name": name,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "loss_mean": sum(losses) / len(losses),
        "router_imbalance_mean": sum(imbalances) / len(imbalances),
        "adaptive_selected_fraction_mean": sum(fractions) / len(fractions),
        "adaptive_effective_steps_total": sum(effective_steps),
        "elapsed_seconds_cpu": elapsed,
        "tokens_processed": len(batches) * batches[0][0].numel(),
        "tokens_per_second_cpu": len(batches) * batches[0][0].numel() / elapsed,
    }


def peak_rss_bytes() -> int:
    """Pico RSS del proceso aislado en Linux; evita confundir RAM de una variante con la otra."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1]) * 1024
    # Linux expresa ru_maxrss en KiB; se usa sólo como fallback documentado.
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def isolated_worker(queue: mp.Queue, name: str, config_values: dict, corpus_path: str, steps: int, batch_size: int, seq_len: int, seed: int, memory_directory: str) -> None:
    try:
        tokens = byte_tokens(Path(corpus_path).read_text(encoding="utf-8"))
        batches = fixed_batches(tokens, steps, batch_size, seq_len, seed)
        result = train_variant(name, NextGenConfig(**config_values), batches, Path(memory_directory) / f"{name}_episodic.jsonl", seed)
        result["peak_rss_bytes"] = peak_rss_bytes()
        queue.put({"result": result})
    except Exception as error:  # pragma: no cover - surfaced to caller with context
        queue.put({"error": repr(error)})


def run_isolated_variant(name: str, config_values: dict, args: argparse.Namespace, memory_directory: Path) -> dict:
    context = mp.get_context("spawn")
    queue = context.Queue()
    process = context.Process(target=isolated_worker, args=(queue, name, config_values, args.corpus, args.steps, args.batch_size, args.seq_len, args.seed, str(memory_directory)))
    process.start()
    payload = queue.get()
    process.join(timeout=120)
    if process.exitcode != 0 or "error" in payload:
        raise RuntimeError(f"La variante {name} no completó la medición aislada: {payload.get('error', process.exitcode)}")
    return payload["result"]


def run(args: argparse.Namespace) -> dict:
    corpus = Path(args.corpus)
    text = corpus.read_text(encoding="utf-8")
    common = dict(vocab_size=256, dim=args.dim, layers=args.layers, heads=args.heads, kv_heads=args.kv_heads, experts=args.experts, active_experts=1, max_seq_len=args.seq_len)
    baseline_config = NextGenConfig(**common)
    arc_config = NextGenConfig(**common, adaptive_refinement_steps=args.arc_steps, adaptive_refinement_threshold=args.arc_threshold, adaptive_compute_penalty=args.arc_penalty)
    with tempfile.TemporaryDirectory(prefix="aethel-arc-") as directory:
        root = Path(directory)
        baseline = run_isolated_variant("baseline", vars(baseline_config), args, root)
        arc = run_isolated_variant("arc", vars(arc_config), args, root)
    result = {
        "protocol": "same-corpus-same-batches-same-seed",
        "corpus": str(corpus),
        "corpus_sha256": hashlib.sha256(corpus.read_bytes()).hexdigest(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "seed": args.seed,
        "baseline": baseline,
        "arc": arc,
        "delta": {
            "parameters": arc["parameters"] - baseline["parameters"],
            "loss_last": arc["loss_last"] - baseline["loss_last"],
            "loss_mean": arc["loss_mean"] - baseline["loss_mean"],
            "router_imbalance_mean": arc["router_imbalance_mean"] - baseline["router_imbalance_mean"],
            "cpu_tokens_per_second": arc["tokens_per_second_cpu"] - baseline["tokens_per_second_cpu"],
            "peak_rss_bytes": arc["peak_rss_bytes"] - baseline["peak_rss_bytes"],
        },
        "interpretation": "Experimento de humo en CPU: mide ejecución real sobre el corpus indicado, pero no autoriza una conclusión de calidad, FLOPs, VRAM o rendimiento GPU.",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="engine/corpora/aethel_repo_corpus.txt")
    parser.add_argument("--output", default="training/experiments/arc_baseline_comparison.json")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--arc-steps", type=int, default=2)
    parser.add_argument("--arc-threshold", type=float, default=0.35)
    parser.add_argument("--arc-penalty", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=17)
    print(json.dumps(run(parser.parse_args()), ensure_ascii=False, indent=2))
