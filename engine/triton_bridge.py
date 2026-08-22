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


def causal_prefill_reference(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Referencia explícita de prefill causal para el futuro kernel por bloques.

    El contrato es q/k/v con forma [B, H, S, D], máscara triangular inferior
    inclusiva y softmax estable en float. Está pensada para equivalencia CPU,
    no para rendimiento ni como sustituto de Triton en CUDA estricto.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or q.shape != k.shape or k.shape != v.shape:
        raise ValueError("prefill causal espera q/k/v con la misma forma [B,H,S,D]")
    _, _, sequence_length, head_dim = q.shape
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * (head_dim ** -0.5)
    causal_mask = torch.ones(
        (sequence_length, sequence_length), dtype=torch.bool, device=q.device
    ).tril()
    scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    return torch.matmul(weights, v.float()).to(q.dtype)


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


def moe_dispatch_combine_reference(
    tokens: torch.Tensor,
    selected_experts: torch.Tensor,
    gates: torch.Tensor,
    expert_functions: list,
) -> torch.Tensor:
    """Referencia semántica de dispatch/combina MoE para CPU y pruebas.

    Conserva exactamente el contrato que deberá implementar el kernel Triton:
    cada asignación `(token, slot)` se agrupa por experto, el experto procesa
    el token y la contribución se acumula ponderada por el gate normalizado.
    No es un kernel rápido ni autoriza fallback GPU estricto.
    """
    if tokens.ndim != 2:
        raise ValueError("dispatch MoE espera tokens [tokens, dim]")
    if selected_experts.ndim != 2 or gates.shape != selected_experts.shape:
        raise ValueError("selected_experts y gates deben tener forma [tokens, top_k]")
    if selected_experts.shape[0] != tokens.shape[0]:
        raise ValueError("cada token debe tener las mismas asignaciones MoE")
    if selected_experts.dtype != torch.long:
        raise ValueError("selected_experts debe contener índices torch.long")
    if not expert_functions:
        raise ValueError("dispatch MoE requiere al menos un experto")
    if selected_experts.numel() and (
        selected_experts.min().item() < 0 or selected_experts.max().item() >= len(expert_functions)
    ):
        raise ValueError("índice de experto fuera del catálogo de expertos")

    output = torch.zeros_like(tokens)
    for expert_index, expert_function in enumerate(expert_functions):
        token_indices, slots = torch.where(selected_experts == expert_index)
        if token_indices.numel() == 0:
            continue
        expert_output = expert_function(tokens[token_indices])
        if expert_output.shape != (token_indices.numel(), tokens.shape[1]):
            raise ValueError("un experto MoE devolvió una forma incompatible")
        contribution = expert_output * gates[token_indices, slots].unsqueeze(-1).to(expert_output.dtype)
        output = output.index_add(0, token_indices, contribution.to(output.dtype))
    return output


def moe_capacity_reference(
    selected_experts: torch.Tensor,
    *,
    n_experts: int,
    capacity: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Define el packing determinista de capacidad para un dispatch MoE futuro.

    Devuelve `positions`, `accepted` y `loads`. El orden de admisión es
    token-major y, dentro del token, slot-major; cualquier asignación que
    exceda la capacidad del experto queda con posición -1 y `accepted=False`.
    Es una referencia CPU de metadatos, no un kernel de producción.
    """
    if selected_experts.ndim != 2 or selected_experts.dtype != torch.long:
        raise ValueError("capacity MoE espera índices [tokens, top_k] torch.long")
    if n_experts < 1 or capacity < 1:
        raise ValueError("n_experts y capacity deben ser positivos")
    if selected_experts.numel() and (
        selected_experts.min().item() < 0 or selected_experts.max().item() >= n_experts
    ):
        raise ValueError("índice de experto fuera del rango de capacidad")

    positions = torch.full_like(selected_experts, -1)
    accepted = torch.zeros_like(selected_experts, dtype=torch.bool)
    loads = torch.zeros((n_experts,), dtype=torch.long, device=selected_experts.device)
    for flat_index, expert_index in enumerate(selected_experts.reshape(-1).tolist()):
        current_load = int(loads[expert_index])
        if current_load >= capacity:
            continue
        token_index = flat_index // selected_experts.shape[1]
        slot_index = flat_index % selected_experts.shape[1]
        positions[token_index, slot_index] = current_load
        accepted[token_index, slot_index] = True
        loads[expert_index] += 1
    return positions, accepted, loads
