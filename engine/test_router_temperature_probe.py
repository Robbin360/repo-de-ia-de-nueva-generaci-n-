"""Probe CPU: deterministic temperature scaling versus hard top-k assignment.

This is not a training run. It verifies an important routing invariant: for
positive temperatures, dividing logits by temperature changes soft entropy but
not deterministic argtopk indices. A temperature-only intervention therefore
cannot fix hard expert collapse unless the training rule also samples, adds
noise, or uses a temperature-sensitive auxiliary objective.
"""

from __future__ import annotations

import torch


def hard_assignments(logits: torch.Tensor, k: int, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.topk(logits / temperature, k=k, dim=-1).indices


def soft_entropy(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    probabilities = torch.softmax(logits / temperature, dim=-1)
    return -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=-1)


def main() -> None:
    torch.manual_seed(17)
    logits = torch.randn(4, 32, 8) * 0.2
    logits[..., 0] += 2.0
    logits[..., 1] += 1.5

    cold = hard_assignments(logits, k=2, temperature=0.5)
    warm = hard_assignments(logits, k=2, temperature=2.0)
    assert torch.equal(cold, warm), "positive temperature changed deterministic top-k order"

    cold_entropy = soft_entropy(logits, temperature=0.5).mean().item()
    warm_entropy = soft_entropy(logits, temperature=2.0).mean().item()
    assert warm_entropy > cold_entropy, (cold_entropy, warm_entropy)

    print("ROUTER_TEMPERATURE_PROBE_PASS")
    print(f"hard_assignments_equal=true")
    print(f"cold_soft_entropy={cold_entropy:.9f}")
    print(f"warm_soft_entropy={warm_entropy:.9f}")
    print("interpretation=temperature_changes_soft_entropy_not_deterministic_top_k")


if __name__ == "__main__":
    main()
