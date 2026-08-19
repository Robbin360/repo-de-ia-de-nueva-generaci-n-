"""Comprueba el contador analítico contra una configuración Aethel pequeña real."""
from __future__ import annotations

from aethel_model import enforce_triton_moe_dispatch_contract, enforce_triton_prefill_contract
from aethel_nextgen import AethelNextGen, NextGenConfig
from report_model_budget import report


def test_analytical_budget_matches_tiny_model() -> None:
    values = {
        "vocab_size": 64,
        "dim": 64,
        "layers": 1,
        "heads": 8,
        "kv_heads": 2,
        "experts": 2,
        "active_experts": 1,
        "max_seq_len": 32,
    }
    model = AethelNextGen(NextGenConfig(**values), memory_path="/tmp/aethel-budget-test.jsonl")
    actual = sum(parameter.numel() for parameter in model.parameters())
    assert report("tiny", values)["parameters_total"] == actual


def test_triton_required_prefill_contract_blocks_gpu_sdpa_without_cuda() -> None:
    """La política se prueba como función pura: no finge disponer de una GPU."""
    enforce_triton_prefill_contract(require_triton=False, is_cuda=True, is_decode=False)
    enforce_triton_prefill_contract(require_triton=True, is_cuda=False, is_decode=False)
    enforce_triton_prefill_contract(require_triton=True, is_cuda=True, is_decode=True)
    try:
        enforce_triton_prefill_contract(require_triton=True, is_cuda=True, is_decode=False)
    except RuntimeError as error:
        assert "kernel Triton causal validado" in str(error)
    else:
        raise AssertionError("El prefill CUDA sin kernel Triton validado debe quedar bloqueado")


def test_triton_required_moe_dispatch_contract_blocks_gpu_python_loop_without_cuda() -> None:
    """El router top-2 no se hace pasar por dispatch MoE completo."""
    enforce_triton_moe_dispatch_contract(require_triton=False, is_cuda=True)
    enforce_triton_moe_dispatch_contract(require_triton=True, is_cuda=False)
    try:
        enforce_triton_moe_dispatch_contract(require_triton=True, is_cuda=True)
    except RuntimeError as error:
        assert "dispatch/combina MoE" in str(error)
    else:
        raise AssertionError("El dispatch MoE CUDA sin kernel Triton debe quedar bloqueado")


if __name__ == "__main__":
    test_analytical_budget_matches_tiny_model()
    test_triton_required_prefill_contract_blocks_gpu_sdpa_without_cuda()
    test_triton_required_moe_dispatch_contract_blocks_gpu_python_loop_without_cuda()
    print("OK")
