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
