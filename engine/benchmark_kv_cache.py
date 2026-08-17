"""Mide la decodificación autoregresiva con y sin KV-cache en la misma CPU/GPU."""
from __future__ import annotations

import json
import time

import torch

from aethel_model import AethelConfig, AethelModel


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def decode_without_cache(model: AethelModel, tokens: torch.Tensor, prompt_len: int) -> float:
    start = time.perf_counter()
    for position in range(prompt_len, tokens.shape[1]):
        model(tokens[:, : position + 1])
    synchronize(tokens.device)
    return time.perf_counter() - start


@torch.no_grad()
def decode_with_cache(model: AethelModel, tokens: torch.Tensor, prompt_len: int) -> float:
    _, _, caches = model(tokens[:, :prompt_len], kv_caches=[None] * len(model.layers))
    start = time.perf_counter()
    for position in range(prompt_len, tokens.shape[1]):
        _, _, caches = model(tokens[:, position : position + 1], start_pos=position, kv_caches=caches)
    synchronize(tokens.device)
    return time.perf_counter() - start


def main() -> None:
    torch.manual_seed(23)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = AethelConfig(vocab_size=256, dim=128, n_layers=2, n_heads=4, n_kv_heads=1, n_experts=2, active_experts=1, max_seq_len=128)
    model = AethelModel(config).to(device).eval()
    tokens = torch.randint(0, config.vocab_size, (1, 96), device=device)
    prompt_len = 48
    # Calentamiento para estabilizar carga de módulos y kernels.
    decode_without_cache(model, tokens[:, :64], 32)
    decode_with_cache(model, tokens[:, :64], 32)
    no_cache_seconds = decode_without_cache(model, tokens, prompt_len)
    cache_seconds = decode_with_cache(model, tokens, prompt_len)
    generated = tokens.shape[1] - prompt_len
    result = {
        "device": str(device),
        "prompt_tokens": prompt_len,
        "generated_tokens": generated,
        "without_cache_seconds": round(no_cache_seconds, 6),
        "with_cache_seconds": round(cache_seconds, 6),
        "without_cache_tokens_per_second": round(generated / no_cache_seconds, 2),
        "with_cache_tokens_per_second": round(generated / cache_seconds, 2),
        "speedup": round(no_cache_seconds / cache_seconds, 3),
        "note": "Microbenchmark local: comparar sólo en el mismo hardware, versión PyTorch y configuración.",
    }
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
