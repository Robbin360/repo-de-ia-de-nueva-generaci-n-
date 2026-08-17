"""Puente de kernels Aethel: Triton en CUDA, fallback sólo para laboratorio CPU."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _swiglu_kernel(x_ptr, y_ptr, out_ptr, n_elements, block_size: tl.constexpr):
        offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        tl.store(out_ptr + offsets, x * tl.sigmoid(x.to(tl.float32)) * y, mask=mask)

    @triton.jit
    def _causal_decode_kernel(q_ptr, k_ptr, v_ptr, out_ptr, seq_len, head_dim: tl.constexpr, max_seq: tl.constexpr):
        program = tl.program_id(0)
        offsets_s = tl.arange(0, max_seq)
        offsets_d = tl.arange(0, head_dim)
        q = tl.load(q_ptr + program * head_dim + offsets_d, mask=offsets_d < head_dim, other=0.0)
        base = program * seq_len * head_dim
        matrix_offsets = base + offsets_s[:, None] * head_dim + offsets_d[None, :]
        mask = (offsets_s[:, None] < seq_len) & (offsets_d[None, :] < head_dim)
        keys = tl.load(k_ptr + matrix_offsets, mask=mask, other=0.0)
        scores = tl.sum(keys * q[None, :], axis=1) * (1.0 / tl.sqrt(head_dim.to(tl.float32)))
        scores = tl.where(offsets_s < seq_len, scores, float("-inf"))
        weights = tl.softmax(scores)
        values = tl.load(v_ptr + matrix_offsets, mask=mask, other=0.0)
        output = tl.sum(values * weights[:, None], axis=0)
        tl.store(out_ptr + program * head_dim + offsets_d, output, mask=offsets_d < head_dim)

    @triton.jit
    def _top2_router_kernel(logits_ptr, indices_ptr, weights_ptr, n_tokens, n_experts: tl.constexpr, block_experts: tl.constexpr):
        token = tl.program_id(0)
        offsets = tl.arange(0, block_experts)
        logits = tl.load(logits_ptr + token * n_experts + offsets, mask=offsets < n_experts, other=float("-inf"))
        first_value = tl.max(logits, axis=0)
        first_index = tl.argmax(logits, axis=0)
        second_logits = tl.where(offsets == first_index, float("-inf"), logits)
        second_value = tl.max(second_logits, axis=0)
        second_index = tl.argmax(second_logits, axis=0)
        maximum = tl.maximum(first_value, second_value)
        first_weight = tl.exp(first_value - maximum)
        second_weight = tl.exp(second_value - maximum)
        normalizer = first_weight + second_weight
        tl.store(indices_ptr + token * 2, first_index)
        tl.store(indices_ptr + token * 2 + 1, second_index)
        tl.store(weights_ptr + token * 2, first_weight / normalizer)
        tl.store(weights_ptr + token * 2 + 1, second_weight / normalizer)


def fused_swiglu(x: torch.Tensor, y: torch.Tensor, *, require_triton: bool = False) -> torch.Tensor:
    """Ejecuta SwiGLU fusionado; GPU de producción puede exigir Triton explícitamente."""
    if x.shape != y.shape:
        raise ValueError("SwiGLU requiere tensores con la misma forma")
    if x.is_cuda and y.is_cuda and HAS_TRITON:
        output = torch.empty_like(x)
        _swiglu_kernel[(triton.cdiv(x.numel(), 1024),)](x, y, output, x.numel(), block_size=1024)
        return output
    if require_triton:
        raise RuntimeError("La ruta GPU de Aethel exige Triton y CUDA; no se permite fallback PyTorch en producción.")
    return F.silu(x) * y


def causal_decode_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, require_triton: bool = False) -> torch.Tensor:
    """Atención exacta para decodificar un único token contra una KV-cache.

    La ruta Triton está limitada deliberadamente a `seq_len <= 2048` y
    `head_dim` potencia de dos hasta 128. El prefill de entrenamiento sigue
    usando SDPA hasta contar con un kernel causal por bloques validado en GPU.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape != (*k.shape[:2], 1, k.shape[-1]) or k.shape != v.shape:
        raise ValueError("decode attention espera q=[B,H,1,D] y k/v=[B,H,S,D]")
    if q.is_cuda and k.is_cuda and v.is_cuda and HAS_TRITON:
        batch, heads, _, head_dim = q.shape
        seq_len = k.shape[2]
        if seq_len > 2048 or head_dim > 128 or head_dim & (head_dim - 1):
            raise RuntimeError("decode Triton requiere seq_len <= 2048 y head_dim potencia de dos <= 128")
        q_flat = q.contiguous().view(batch * heads, head_dim)
        k_flat = k.contiguous().view(batch * heads, seq_len, head_dim)
        v_flat = v.contiguous().view(batch * heads, seq_len, head_dim)
        output = torch.empty_like(q_flat)
        _causal_decode_kernel[(batch * heads,)](q_flat, k_flat, v_flat, output, seq_len, head_dim=head_dim, max_seq=2048)
        return output.view(batch, heads, 1, head_dim)
    if require_triton:
        raise RuntimeError("la decodificación GPU de Aethel exige Triton y CUDA; no se permite fallback PyTorch en producción")
    return F.scaled_dot_product_attention(q, k, v, is_causal=False)


def top2_router(logits: torch.Tensor, *, require_triton: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Selecciona y normaliza los dos expertos mejores durante inferencia."""
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("router top-2 espera logits [tokens, expertos>=2]")
    if logits.is_cuda and HAS_TRITON:
        tokens, experts = logits.shape
        if experts > 32:
            raise RuntimeError("router Triton preparado admite hasta 32 expertos; divida la ruta antes de activarla")
        indices = torch.empty((tokens, 2), device=logits.device, dtype=torch.long)
        weights = torch.empty((tokens, 2), device=logits.device, dtype=logits.dtype)
        block_experts = triton.next_power_of_2(experts)
        _top2_router_kernel[(tokens,)](logits, indices, weights, tokens, n_experts=experts, block_experts=block_experts)
        return weights, indices
    if require_triton:
        raise RuntimeError("el router GPU de Aethel exige Triton y CUDA; no se permite fallback PyTorch en producción")
    probabilities = F.softmax(logits, dim=-1, dtype=torch.float)
    weights, indices = torch.topk(probabilities, 2, dim=-1)
    return (weights / weights.sum(dim=-1, keepdim=True)).to(logits.dtype), indices
