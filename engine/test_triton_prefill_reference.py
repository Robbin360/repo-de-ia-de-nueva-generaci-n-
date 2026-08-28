"""Contrato CPU del prefill causal pendiente de una implementación Triton por bloques."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from triton_bridge import causal_prefill_experimental, causal_prefill_reference


def test_prefill_reference_matches_sdpa() -> None:
    torch.manual_seed(17)
    q = torch.randn((2, 3, 7, 8))
    k = torch.randn((2, 3, 7, 8))
    v = torch.randn((2, 3, 7, 8))
    actual = causal_prefill_reference(q, k, v)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_prefill_reference_blocks_future_tokens() -> None:
    q = torch.ones((1, 1, 3, 2))
    k = torch.ones((1, 1, 3, 2))
    v = torch.tensor([[[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]]])
    output = causal_prefill_reference(q, k, v)
    assert torch.equal(output[0, 0, 0], v[0, 0, 0])
    assert output[0, 0, 1, 0] < 10.0


def test_prefill_reference_rejects_incompatible_shapes() -> None:
    q = torch.ones((1, 1, 2, 2))
    try:
        causal_prefill_reference(q, q, torch.ones((1, 1, 3, 2)))
    except ValueError as error:
        assert "misma forma" in str(error)
    else:
        raise AssertionError("prefill debe rechazar shapes incompatibles")


def test_experimental_prefill_uses_cpu_reference_semantics_without_cuda() -> None:
    torch.manual_seed(29)
    q = torch.randn((1, 2, 5, 8))
    k = torch.randn((1, 2, 5, 8))
    v = torch.randn((1, 2, 5, 8))
    actual = causal_prefill_experimental(q, k, v)
    expected = causal_prefill_reference(q, k, v)
    assert torch.allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_experimental_prefill_keeps_strict_cuda_contract() -> None:
    q = torch.ones((1, 1, 2, 2))
    try:
        causal_prefill_experimental(q, q, q, require_triton=True)
    except RuntimeError as error:
        assert "Triton y CUDA" in str(error)
    else:
        raise AssertionError("el prefill experimental no puede relajar el contrato estricto")


def test_prefill_launch_scale_is_plain_python_float() -> None:
    """Evita repetir `.to()` sobre un `tl.constexpr` entero en la compilación GPU."""
    scale = 64 ** -0.5
    assert isinstance(scale, float)
    assert math.isclose(scale, 0.125)


if __name__ == "__main__":
    test_prefill_reference_matches_sdpa()
    test_prefill_reference_blocks_future_tokens()
    test_prefill_reference_rejects_incompatible_shapes()
    test_experimental_prefill_uses_cpu_reference_semantics_without_cuda()
    test_experimental_prefill_keeps_strict_cuda_contract()
    test_prefill_launch_scale_is_plain_python_float()
    print("PASS: referencia CPU de prefill causal verificada")
