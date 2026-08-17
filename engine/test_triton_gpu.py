"""Validación GPU real del kernel Triton integrado; no produce métricas si falta CUDA."""

from __future__ import annotations

import time

import torch

from triton_bridge import HAS_TRITON, fused_swiglu


def main() -> None:
    if not torch.cuda.is_available() or not HAS_TRITON:
        print({"status": "SKIPPED", "reason": "CUDA y Triton son obligatorios para esta validación real", "cuda": torch.cuda.is_available(), "triton": HAS_TRITON})
        return

    torch.manual_seed(17)
    x = torch.randn((8, 128, 1024), device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    reference = torch.nn.functional.silu(x) * y
    torch.cuda.synchronize()
    start = time.perf_counter()
    actual = fused_swiglu(x, y, require_triton=True)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1_000
    max_error = (actual - reference).abs().max().item()
    assert torch.allclose(actual, reference, rtol=1e-4, atol=1e-5), f"Triton SwiGLU diverge: max_error={max_error}"
    print({"status": "VERIFIED", "kernel": "swiglu", "device": torch.cuda.get_device_name(), "shape": list(x.shape), "max_error": max_error, "first_call_ms": elapsed_ms})


if __name__ == "__main__":
    main()
