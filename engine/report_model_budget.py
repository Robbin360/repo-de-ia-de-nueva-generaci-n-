"""Informa presupuestos reales de parámetros Aethel antes de iniciar una corrida costosa."""
from __future__ import annotations

import argparse
import json
import math


PRESETS = {
    "pilot-100m": {"vocab_size": 32_000, "dim": 512, "layers": 4, "heads": 8, "kv_heads": 2, "experts": 8, "active_experts": 2, "max_seq_len": 1024},
    "research-300m": {"vocab_size": 32_000, "dim": 768, "layers": 8, "heads": 12, "kv_heads": 3, "experts": 8, "active_experts": 2, "max_seq_len": 2048},
    "scale-1b": {"vocab_size": 32_000, "dim": 1024, "layers": 16, "heads": 16, "kv_heads": 4, "experts": 8, "active_experts": 2, "max_seq_len": 2048},
}


def report(name: str, values: dict) -> dict:
    """Cuenta tensores a partir de sus formas, sin instanciar pesos ni agotar RAM."""
    dim = values["dim"]
    heads = values["heads"]
    kv_heads = values["kv_heads"]
    experts = values["experts"]
    layers = values["layers"]
    vocab_size = values["vocab_size"]
    hidden_dim = 256 * math.ceil((8 * dim / 3) / 256)

    tied_embedding = vocab_size * dim
    attention = (2 * dim * dim) + (2 * dim * dim * kv_heads // heads)
    moe = (dim * experts) + (3 * experts * dim * hidden_dim)
    norms = 2 * dim
    core = tied_embedding + layers * (attention + moe + norms) + dim
    # La Roca, El Líquido, neuromodulación, Espacio Global, GRU, norma y memory_to_core.
    # GRUCell contiene dos matrices de 3*dim x dim y dos sesgos de 3*dim.
    nextgen = (dim * dim + dim) + (dim * dim) + (dim + 1) + (9 * dim + 3) + (dim * dim) + (6 * dim * dim + 6 * dim) + (2 * dim) + (dim * dim)
    total = core + nextgen
    # Estimación transparente: pesos BF16, gradientes BF16 y dos estados Adam FP32 por parámetro.
    optimizer_bytes = total * (2 + 2 + 4 + 4)
    return {
        "preset": name,
        "parameters_total": total,
        "parameters_millions": round(total / 1_000_000, 2),
        "active_expert_fraction": f"{values['active_experts']}/{experts}",
        "estimated_optimizer_gib": round(optimizer_bytes / 1024**3, 2),
        "context": values["max_seq_len"],
        "experts": experts,
        "active_experts": values["active_experts"],
        "moe_hidden_dim": hidden_dim,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=[*PRESETS, "all"], default="all")
    args = parser.parse_args()
    selected = PRESETS.items() if args.preset == "all" else [(args.preset, PRESETS[args.preset])]
    print(json.dumps([report(name, values) for name, values in selected], ensure_ascii=False, indent=2))
