import torch

from triton_bridge import HAS_TRITON, fused_swiglu


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
    print({"triton_bridge_cpu_verified": True, "triton_available": HAS_TRITON})


if __name__ == "__main__":
    main()
