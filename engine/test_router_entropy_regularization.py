"""Prueba CPU pura de la hipótesis de entropía del router; no usa Dataset ni checkpoint."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from engine.router_auxiliary import router_entropy_regularization_loss


def main() -> None:
    concentrated_logits = torch.tensor(
        [[5.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    uniform_logits = torch.zeros((2, 4), dtype=torch.float32, requires_grad=True)
    concentrated = torch.softmax(concentrated_logits, dim=-1)
    uniform = torch.softmax(uniform_logits, dim=-1)

    concentrated_loss = router_entropy_regularization_loss(concentrated)
    uniform_loss = router_entropy_regularization_loss(uniform)
    # La salida es entropía negativa: uniforme = -1 y concentrada > -1.
    assert concentrated_loss.item() > uniform_loss.item()

    concentrated_loss.backward()
    assert concentrated_logits.grad is not None
    # Descenso del gradiente sobre el logit dominante debe aumentar entropía.
    assert concentrated_logits.grad[0, 0].item() > 0.0
    assert concentrated_logits.grad[0, 1].item() < 0.0

    uniform_loss.backward()
    assert uniform_logits.grad is not None
    assert torch.allclose(uniform_logits.grad, torch.zeros_like(uniform_logits.grad), atol=1e-6)
    print("ROUTER_ENTROPY_REGULARIZATION_CPU_CONTRACT_OK")


if __name__ == "__main__":
    main()
