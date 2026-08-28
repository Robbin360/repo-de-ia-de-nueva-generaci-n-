"""Ejecutor de evidencia CUDA para la matriz de aceptación Triton de Aethel.

No habilita `require_triton`, no entrena pesos y no evalúa el Dataset. Registra
paridad y entorno para una configuración autorizada; el informe debe revisarse
antes de modificar cualquier contrato de producción.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent.parent
ENGINE = ROOT / "engine"
sys.path.insert(0, str(ENGINE))

from triton_bridge import (  # noqa: E402
    HAS_TRITON,
    causal_prefill_experimental,
    moe_capacity_reference,
    top2_router,
)


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def source_release() -> dict:
    """Lee la identidad inmutable del bundle cuando Kaggle ha extraído código sin `.git`."""
    marker = ROOT / "training" / "aethel_kaggle_source_release.json"
    if not marker.is_file():
        return {"status": "unavailable", "reason": "source release marker is absent"}
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "invalid", "reason": f"{type(error).__name__}: {error}"}
    return {
        "status": "available",
        "schema": payload.get("schema"),
        "release": payload.get("release"),
        "training_authorized": payload.get("training_authorized"),
    }


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    delta = (actual.float() - expected.float()).abs()
    relative = delta / expected.float().abs().clamp_min(1e-6)
    return float(delta.max().item()), float(relative.max().item())


def prefill_case(batch: int, heads: int, sequence: int, head_dim: int, dtype: torch.dtype) -> dict:
    torch.manual_seed(1729 + sequence + head_dim)
    q = torch.randn((batch, heads, sequence, head_dim), device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    actual = causal_prefill_experimental(q, k, v, require_triton=True)
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    abs_error, relative_error = max_errors(actual, expected)
    return {
        "shape": [batch, heads, sequence, head_dim],
        "dtype": str(dtype).replace("torch.", ""),
        "max_abs_error": abs_error,
        "max_relative_error": relative_error,
        "elapsed_ms": elapsed_ms,
        "peak_bytes": int(torch.cuda.max_memory_allocated()),
        "finite": bool(torch.isfinite(actual).all().item()),
    }


def routing_case() -> dict:
    torch.manual_seed(1731)
    logits = torch.randn((65, 8), device="cuda", dtype=torch.float32)
    gates, indices = top2_router(logits, require_triton=True)
    expected_gates, expected_indices = torch.topk(torch.softmax(logits, dim=-1), 2, dim=-1)
    expected_gates = expected_gates / expected_gates.sum(dim=-1, keepdim=True)
    positions, accepted, loads = moe_capacity_reference(indices, n_experts=8, capacity=16)
    return {
        "indices_equal": bool(torch.equal(indices, expected_indices)),
        "gates_max_abs_error": float((gates - expected_gates).abs().max().item()),
        "gates_normalized": bool(torch.allclose(gates.sum(dim=-1), torch.ones(65, device="cuda"))),
        "capacity_accepted": int(accepted.sum().item()),
        "capacity_dropped": int((~accepted).sum().item()),
        "capacity_positions_valid": bool(((positions >= -1) & (positions < 16)).all().item()),
        "loads": [int(value) for value in loads.cpu().tolist()],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--full-matrix", action="store_true")
    args = parser.parse_args()

    report: dict = {
        "purpose": "Aethel Triton CUDA acceptance evidence",
        "commit": git_commit(),
        "source_release": source_release(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton_importable": HAS_TRITON,
        "cuda_available": torch.cuda.is_available(),
        "status": "NOT_RUN",
    }
    if not torch.cuda.is_available() or not HAS_TRITON:
        report["blocker"] = "CUDA y Triton deben estar disponibles; no se habilitó ningún contrato."
        write_report(args.output, report)
        print(json.dumps(report, sort_keys=True))
        return 2

    report.update(
        {
            "device": torch.cuda.get_device_name(0),
            "cuda": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "driver_note": "registrar driver mediante nvidia-smi junto al informe",
        }
    )
    cases = [(1, 1, 31, 32), (1, 2, 64, 64)]
    if args.full_matrix:
        cases.extend([(1, 1, 127, 64), (1, 8, 256, 128), (2, 8, 1024, 128)])
    try:
        report["prefill"] = [prefill_case(*case, torch.float16) for case in cases]
        report["router_capacity"] = routing_case()
        report["status"] = "PASSED_EXPERIMENTAL"
        report["notice"] = (
            "Este estado no habilita prefill/dispatch estrictos: faltan gradientes, "
            "dispatch-combine Triton y revisión de matriz completa."
        )
    except Exception as error:  # pylint: disable=broad-except
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
    write_report(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASSED_EXPERIMENTAL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
