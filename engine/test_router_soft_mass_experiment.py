"""Experimento CPU: masa probabilística suave contra atractor top-2.

No modifica el modelo ni usa corpus/GPU. Compara una señal de carga discreta
(density top-2) con una penalización sobre la masa probabilística agregada.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def normalized_imbalance(values: torch.Tensor) -> float:
    target = 1.0 / values.numel()
    return float((values - target).abs().sum().item() / 2.0)


def run(*, soft_mass: bool, steps: int = 160) -> dict[str, float | int | bool]:
    torch.manual_seed(17)
    tokens, experts = 256, 8
    logits = torch.zeros(tokens, experts, requires_grad=True)
    with torch.no_grad():
        logits[:, 0] = 2.5
        logits[:, 1] = 2.0
        logits[:, 2:] = -0.5
    optimizer = torch.optim.SGD([logits], lr=0.25)
    for _ in range(steps):
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(2, dim=-1).indices
        density = torch.zeros(experts)
        density.scatter_add_(0, top.reshape(-1), torch.ones(tokens * 2) / (tokens * 2))
        if soft_mass:
            mass = probs.mean(dim=0)
            loss = ((mass - 1.0 / experts) ** 2).sum() * experts
        else:
            # La densidad top-2 dura es no diferenciable; esta rama representa
            # un baseline sin señal auxiliar de masa, manteniendo un grafo nulo.
            loss = probs.sum() * 0.0
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(2, dim=-1).indices
        density = torch.zeros(experts)
        density.scatter_add_(0, top.reshape(-1), torch.ones(tokens * 2) / (tokens * 2))
        mass = probs.mean(dim=0)
    return {
        "soft_mass_signal": soft_mass,
        "steps": steps,
        "top2_imbalance": normalized_imbalance(density),
        "probability_mass_imbalance": normalized_imbalance(mass),
        "top2_experts_used": int(torch.unique(top).numel()),
        "top2_counts": density.tolist(),
        "probability_mass": mass.tolist(),
    }


def main() -> None:
    result = {"hard_density": run(soft_mass=False), "soft_mass": run(soft_mass=True)}
    result["interpretation"] = (
        "CPU-only toy router experiment; not a model-quality, GPU-performance or training result. "
        "A signal is useful only if it improves hard top-2 load without unacceptable routing cost."
    )
    output = Path("training/experiments/router_soft_mass_cpu_2026-08-28.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
