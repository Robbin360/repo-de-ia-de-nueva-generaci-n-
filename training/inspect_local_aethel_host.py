#!/usr/bin/env python3
"""Genera un informe local de preparación Aethel sin entrenar ni reservar GPU."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "training"))
from validate_aethel_knowledge_package import validate  # noqa: E402


def writable_directory(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".aethel-write-probe"
        probe.write_text("probe\n", encoding="utf-8")
        probe.unlink()
        return True, None
    except OSError as error:
        return False, str(error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT)
    parser.add_argument("--data-dir", type=Path, default=Path("/home/ubuntu/aethel-knowledge-corpus-v1-package"))
    parser.add_argument("--output-dir", type=Path, default=Path("./aethel-host-inspection"))
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    report_path = args.report.resolve() if args.report else output_dir / "host_inspection.json"
    blockers: list[str] = []
    required_source = (
        "training/run_kaggle_seed_offline.sh",
        "training/run_gpu_preflight.sh",
        "training/validate_aethel_knowledge_package.py",
        "engine/train_aethel_gpu.py",
        "engine/evaluate_nextgen.py",
        "engine/triton_bridge.py",
    )
    missing_source = [relative for relative in required_source if not (source_dir / relative).is_file()]
    if missing_source:
        blockers.append("source_contract_missing")

    dataset = {"valid": False, "errors": ["No se ejecutó la validación."]}
    if not data_dir.is_dir():
        blockers.append("dataset_directory_missing")
    else:
        dataset = validate(data_dir)
        if not dataset.get("valid"):
            blockers.append("dataset_validation_failed")

    writable, write_error = writable_directory(output_dir)
    if not writable:
        blockers.append("output_directory_not_writable")
    disk_path = output_dir if output_dir.exists() else output_dir.parent
    disk = shutil.disk_usage(disk_path)

    cuda_available = torch.cuda.is_available()
    devices = []
    if cuda_available:
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "vram_bytes": int(properties.total_memory),
                    "compute_capability": f"{properties.major}.{properties.minor}",
                }
            )
    else:
        blockers.append("cuda_unavailable")

    try:
        import triton  # type: ignore

        triton_version: str | None = getattr(triton, "__version__", "unknown")
    except ImportError:
        triton_version = None
        blockers.append("triton_import_unavailable")

    report = {
        "schema_version": 1,
        "purpose": "Aethel local host readiness; does not authorize training",
        "source_dir": str(source_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "source": {"required_files": list(required_source), "missing": missing_source},
        "dataset": dataset,
        "storage": {"writable": writable, "write_error": write_error, "free_bytes": int(disk.free), "total_bytes": int(disk.total)},
        "environment": {"python": sys.version.split()[0], "torch": torch.__version__, "cuda_available": cuda_available, "cuda_version": torch.version.cuda, "triton_version": triton_version, "devices": devices},
        "seed_status": "READY_FOR_AUTHORIZATION" if not blockers else "BLOCKED",
        "edge_status": "REQUIRES_SEPARATE_CAPACITY_PLAN",
        "blockers": blockers,
        "next": "No iniciar entrenamiento: revisar el informe, ejecutar preflight y obtener autorización explícita.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["seed_status"] == "READY_FOR_AUTHORIZATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
