"""CPU-only probe for reproducible pre-top-k logit noise.

This is an isolated diagnostic, not a training change. It compares deterministic
noise injection against clean logits and reports hard top-k concentration.
"""
from __future__ import annotations

import torch


def hard_assignment(logits: torch.Tensor, k: int = 2) -> torch.Tensor:
    if logits.ndim != 2 or k < 1 or k > logits.shape[1]:
        raise ValueError("logits must be [tokens, experts] and 1 <= k <= experts")
    return torch.topk(logits, k=k, dim=-1).indices


def coverage(assignments: torch.Tensor, experts: int) -> float:
    return float(torch.unique(assignments).numel() / experts)


def concentration(assignments: torch.Tensor, experts: int) -> float:
    counts = torch.bincount(assignments.reshape(-1), minlength=experts).float()
    return float(counts.max().item() / assignments.numel())


def probe(seed: int = 17, tokens: int = 128, experts: int = 8, k: int = 2, sigma: float = 0.05) -> dict[str, float | bool]:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.zeros(tokens, experts)
    logits[:, 0] = 1.0
    logits[:, 1] = 0.99
    clean = hard_assignment(logits, k)
    noisy = hard_assignment(logits + torch.randn(logits.shape, generator=generator) * sigma, k)
    return {
        "seed": float(seed),
        "sigma": float(sigma),
        "clean_coverage": coverage(clean, experts),
        "noisy_coverage": coverage(noisy, experts),
        "clean_concentration": concentration(clean, experts),
        "noisy_concentration": concentration(noisy, experts),
        "deterministic_repeat": bool(torch.equal(noisy, hard_assignment(logits + torch.randn(logits.shape, generator=torch.Generator().manual_seed(seed)) * sigma, k))),
    }


if __name__ == "__main__":
    for sigma in (0.0, 0.01, 0.05, 0.2):
        print(probe(sigma=sigma))
