"""Empaqueta una sesión Edge terminada sin cargar ni modificar los pesos."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


REQUIRED_FILES = (
    "latest.pt",
    "tokenizer.json",
    "metrics_rank_0.jsonl",
    "router_diagnostic.json",
    "recovery_receipt.json",
    "aethel_direct_validation.json",
)
GATE = "SAVE_KAGGLE_VERSION_NOW.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--phase-id", required=True)
    parser.add_argument("--session-target-step", type=int, required=True)
    parser.add_argument("--schedule-total-steps", type=int, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    package = args.package.resolve()
    data_manifest = args.data_manifest.resolve()
    if not output.is_dir() or package.exists():
        raise SystemExit("La salida debe existir y el paquete debe ser inédito.")
    if not data_manifest.is_file() or data_manifest.name != "prepared_manifest.json":
        raise SystemExit("--data-manifest debe ser un prepared_manifest.json existente.")
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"Faltan artefactos obligatorios: {', '.join(missing)}")
    gate = output / GATE
    if gate.exists():
        raise SystemExit("La compuerta Save Version ya existe; no se reutiliza la salida.")
    gate.write_text(
        "AETHEL EDGE SESSION PRESERVATION READY\n"
        "Acción manual obligatoria: use Save Version en Kaggle inmediatamente, en esta misma sesión.\n"
        "No cierre, reinicie ni cambie de sesión hasta preservar el paquete y el checkpoint.\n",
        encoding="utf-8",
    )
    package.parent.mkdir(parents=True, exist_ok=True)
    archived = (*REQUIRED_FILES, "prepared_manifest.json", GATE)
    with tarfile.open(package, "x:gz") as archive:
        for name in archived:
            source = data_manifest if name == "prepared_manifest.json" else output / name
            archive.add(source, arcname=f"{output.name}/{name}", recursive=False)
    receipt = {
        "schema": "aethel-edge-session-preservation-receipt/v1",
        "status": "AETHEL_EDGE_SESSION_PRESERVATION_READY",
        "phase_id": args.phase_id,
        "session_target_step": args.session_target_step,
        "schedule_total_steps": args.schedule_total_steps,
        "checkpoint": {"path": str(output / "latest.pt"), "sha256": sha256_file(output / "latest.pt")},
        "tokenizer": {"path": str(output / "tokenizer.json"), "sha256": sha256_file(output / "tokenizer.json")},
        "data_manifest": {"path": str(data_manifest), "sha256": sha256_file(data_manifest)},
        "metrics": {"path": str(output / "metrics_rank_0.jsonl"), "sha256": sha256_file(output / "metrics_rank_0.jsonl")},
        "package": {"path": str(package), "sha256": sha256_file(package), "contents": list(archived)},
        "required_manual_action": "SAVE_KAGGLE_VERSION_IMMEDIATELY",
        "limits": {"checkpoint_loaded": False, "checkpoint_uploaded": False, "holdout_content_read": False, "promotion_authorized": False},
    }
    (output / "edge_session_preservation_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("AETHEL_EDGE_SESSION_PRESERVATION_READY")
    print(f"PACKAGE={package}")
    print("SAVE_VERSION_NOW=Use Save Version inmediatamente en Kaggle.")


if __name__ == "__main__":
    main()
