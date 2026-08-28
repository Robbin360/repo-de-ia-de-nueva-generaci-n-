#!/usr/bin/env python3
"""Valida las entradas de Aethel Seed E0 sin importar CUDA ni iniciar entrenamiento."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def writable_directory(path: Path) -> tuple[bool, str | None]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".aethel-offline-preflight-write-probe"
        probe.write_text("probe\n", encoding="utf-8")
        probe.unlink()
        return True, None
    except OSError as error:
        return False, str(error)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    sys.path.insert(0, str(source_dir / "training"))
    from validate_aethel_knowledge_package import validate  # noqa: PLC0415

    required_source = (
        "training/validate_aethel_knowledge_package.py",
        "training/run_kaggle_seed_offline.sh",
        "training/run_triton_cuda_acceptance.py",
        "engine/aethel_model.py",
        "engine/train_aethel_gpu.py",
        "engine/evaluate_nextgen.py",
    )
    missing_source = [relative for relative in required_source if not (source_dir / relative).is_file()]
    dataset: dict[str, Any]
    if data_dir.is_dir():
        dataset = validate(data_dir)
    else:
        dataset = {"valid": False, "network_requests": 0, "errors": ["dataset_directory_missing"], "warnings": []}

    writable, write_error = writable_directory(output_dir)
    disk_path = output_dir if output_dir.exists() else output_dir.parent
    disk = shutil.disk_usage(disk_path)
    run_authorized = os.environ.get("AETHEL_RUN_AUTHORIZED", "") == "YES"
    fallback_authorized = os.environ.get("AETHEL_LAB_FALLBACK_AUTHORIZED", "NO") == "YES"
    blockers: list[str] = []
    if missing_source:
        blockers.append("source_contract_missing")
    if not dataset.get("valid"):
        blockers.append("dataset_validation_failed")
    if not writable:
        blockers.append("output_directory_not_writable")
    if run_authorized or fallback_authorized:
        blockers.append("training_authorization_must_remain_disabled_for_offline_preflight")

    report = {
        "schema_version": 1,
        "purpose": "Aethel Seed E0 offline preflight; no CUDA query and no training",
        "source_dir": str(source_dir),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "source": {"required_files": list(required_source), "missing": missing_source},
        "dataset": dataset,
        "storage": {"writable": writable, "write_error": write_error, "free_bytes": int(disk.free)},
        "authorization": {
            "run_authorized": run_authorized,
            "pytorch_fallback_authorized": fallback_authorized,
            "training_started": False,
        },
        "seed_status": "READY_FOR_GPU_AUTHORIZATION" if not blockers else "BLOCKED",
        "blockers": blockers,
        "network_requests": 0,
    }
    report_path = output_dir / "offline_preflight.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0 if report["seed_status"] == "READY_FOR_GPU_AUTHORIZATION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
