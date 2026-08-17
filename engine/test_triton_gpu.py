"""Validación GPU real del kernel Triton integrado; no produce métricas si falta CUDA."""

from __future__ import annotations

import time

import torch

from triton_bridge import HAS_TRITON, causal_decode_attention, fused_swiglu, top2_router


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
    q = torch.randn((2, 4, 1, 64), device="cuda", dtype=torch.float32)
    k = torch.randn((2, 4, 128, 64), device="cuda", dtype=torch.float32)
    v = torch.randn_like(k)
    attention_reference = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    attention_actual = causal_decode_attention(q, k, v, require_triton=True)
    attention_error = (attention_actual - attention_reference).abs().max().item()
    assert torch.allclose(attention_actual, attention_reference, rtol=2e-4, atol=2e-4), f"Triton decode diverge: max_error={attention_error}"

    logits = torch.randn((257, 8), device="cuda", dtype=torch.float32)
    router_reference_values, router_reference_indices = torch.topk(torch.softmax(logits, dim=-1, dtype=torch.float), 2, dim=-1)
    router_reference_values = router_reference_values / router_reference_values.sum(dim=-1, keepdim=True)
    router_actual_values, router_actual_indices = top2_router(logits, require_triton=True)
    assert torch.equal(router_actual_indices, router_reference_indices), "Triton router seleccionó expertos distintos"
    router_error = (router_actual_values - router_reference_values).abs().max().item()
    assert torch.allclose(router_actual_values, router_reference_values, rtol=2e-4, atol=2e-4), f"Triton router diverge: max_error={router_error}"
    print({"status": "VERIFIED", "kernels": ["swiglu", "causal_decode", "router_top2"], "device": torch.cuda.get_device_name(), "swiglu_shape": list(x.shape), "swiglu_max_error": max_error, "first_swiglu_call_ms": elapsed_ms, "decode_max_error": attention_error, "router_max_error": router_error})


if __name__ == "__main__":
    main()
