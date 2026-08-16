import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

from aethel_model import AethelConfig, AethelModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--experts", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", default="engine/artifacts/aethel_real.pt")
    args = parser.parse_args()
    torch.manual_seed(42)
    config = AethelConfig()
    config.vocab_size = 2048
    config.dim = args.dim
    config.n_layers = args.layers
    config.n_heads = max(1, args.dim // 64)
    config.n_kv_heads = max(1, config.n_heads // 2)
    config.n_experts = args.experts
    config.active_experts = min(2, args.experts)
    config.max_seq_len = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = "Aethel aprende representaciones mediante RoPE GQA y Sparse MoE. La Roca conserva memoria estable. El Liquido adapta señales. Ciclo de Sueno consolida conocimiento. Neuromodulacion prioriza sorpresa. Espacio de Trabajo Global integra hipotesis."
    values = torch.tensor([ord(char) % config.vocab_size for char in corpus], dtype=torch.long)
    model = AethelModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    model.train()
    start = time.time()
    for step in range(args.steps):
        positions = torch.randint(0, len(values) - 32, (4,))
        batch = torch.stack([values[pos:pos + 32] for pos in positions]).to(device)
        targets = torch.stack([values[pos + 1:pos + 33] for pos in positions]).to(device)
        logits, aux_loss, _ = model(batch)
        loss = F.cross_entropy(logits.reshape(-1, config.vocab_size), targets.reshape(-1)) + 0.01 * aux_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        payload = {"type": "metric", "step": step + 1, "steps": args.steps, "loss": round(float(loss.detach().cpu()), 6), "tokens": (step + 1) * batch.numel(), "device": str(device), "vram": round(torch.cuda.memory_allocated() / 1e9, 4) if torch.cuda.is_available() else None, "elapsed": round(time.time() - start, 3), "experts": getattr(model, "last_expert_loads", None), "kv_cache": None}
        print(json.dumps(payload), flush=True)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({"model": model.state_dict(), "config": vars(args)}, args.output)
    print(json.dumps({"type": "complete", "output": args.output, "device": str(device)}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"type": "error", "error": str(exc)}), flush=True)
        sys.exit(1)
