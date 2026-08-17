import torch

from triton_bridge import HAS_TRITON, causal_decode_attention, fused_swiglu, top2_router


def main() -> None:
    x = torch.tensor([[-1.0, 0.0, 1.0]])
    y = torch.tensor([[2.0, 2.0, 2.0]])
    expected = torch.nn.functional.silu(x) * y
    assert torch.allclose(fused_swiglu(x, y), expected)
    try:
        fused_swiglu(x, y, require_triton=True)
    except RuntimeError:
        assert not x.is_cuda or not HAS_TRITON
    else:
        assert x.is_cuda and HAS_TRITON

    torch.manual_seed(7)
    q = torch.randn(2, 3, 1, 8)
    k = torch.randn(2, 3, 5, 8)
    v = torch.randn(2, 3, 5, 8)
    actual_attention = causal_decode_attention(q, k, v)
    expected_attention = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    assert torch.allclose(actual_attention, expected_attention, atol=1e-6)

    logits = torch.tensor([[1.0, 4.0, 2.0, 3.0], [5.0, -1.0, 0.0, 2.0]])
    weights, indices = top2_router(logits)
    assert indices.tolist() == [[1, 3], [0, 3]]
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2), atol=1e-6)
    print({"triton_bridge_cpu_verified": True, "triton_available": HAS_TRITON, "decode_fallback_verified": True, "router_fallback_verified": True})


if __name__ == "__main__":
    main()
