import torch

# Intentar importar Triton si está disponible en la GPU NVIDIA
try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:
    @triton.jit
    def swiglu_fwd_kernel(
        x_ptr,      # Puntero al tensor X
        y_ptr,      # Puntero al tensor Y
        out_ptr,    # Puntero a la salida
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """
        Kernel Triton fusionado para SwiGLU: SiLU(X) * Y.
        Mantiene los datos en SRAM y elimina escrituras intermedias a la VRAM (HBM).
        """
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        
        # SwiGLU: x * sigmoid(x) * y
        sigmoid_x = tl.sigmoid(x.to(tl.float32))
        silu_x = x * sigmoid_x
        out = silu_x * y
        
        tl.store(out_ptr + offsets, out, mask=mask)

def fused_swiglu(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Función de SwiGLU fusionado.
    Utiliza el kernel en Triton si CUDA está disponible,
    o utiliza el operador optimizado en PyTorch como fallback.
    """
    assert x.shape == y.shape, "Las dimensiones de X e Y deben coincidir"
    
    if HAS_TRITON and x.is_cuda and y.is_cuda:
        out = torch.empty_like(x)
        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
        
        swiglu_fwd_kernel[grid](
            x, y, out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        return out
    else:
        # Fallback de PyTorch (Eficaz en CPU o dispositivos sin Triton)
        return torch.nn.functional.silu(x) * y

if __name__ == "__main__":
    print("Testing Fused SwiGLU Kernel...")
    x = torch.randn(128, 4096)
    y = torch.randn(128, 4096)
    res = fused_swiglu(x, y)
    print(f"✅ Executed SwiGLU. Output shape: {res.shape}")
