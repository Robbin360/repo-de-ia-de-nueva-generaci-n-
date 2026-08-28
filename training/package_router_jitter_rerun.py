"""Empaqueta evidencia de una repetición jitter ya validada sin cargar pesos."""
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
SAVE_VERSION_GATE = "SAVE_KAGGLE_VERSION_NOW.txt"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    package = args.package.resolve()
    if not output.is_dir():
        raise SystemExit(f"Falta la salida de entrenamiento validada: {output}")
    if package.exists():
        raise SystemExit(f"El paquete ya existe y no se sobrescribe: {package}")
    missing = [name for name in REQUIRED_FILES if not (output / name).is_file()]
    if missing:
        raise SystemExit(f"Faltan artefactos obligatorios: {', '.join(missing)}")

    gate = output / SAVE_VERSION_GATE
    if gate.exists():
        raise SystemExit(f"La compuerta de preservación ya existe y no se reutiliza: {gate}")
    gate.write_text(
        "AETHEL ROUTER JITTER RERUN PRESERVATION READY\n"
        "Acción manual obligatoria: use Save Version en Kaggle antes de cerrar, reiniciar, cambiar de sesión o solicitar otra acción.\n"
        "El checkpoint se conserva sólo si esta salida y su paquete se incluyen en la versión guardada.\n",
        encoding="utf-8",
    )

    package.parent.mkdir(parents=True, exist_ok=True)
    prefix = output.name
    archived_files = (*REQUIRED_FILES, SAVE_VERSION_GATE)
    with tarfile.open(package, "x:gz") as archive:
        for name in archived_files:
            archive.add(output / name, arcname=f"{prefix}/{name}", recursive=False)

    receipt = output / "checkpoint_preservation_receipt.json"
    payload = {
        "schema": "aethel-router-jitter-rerun-preservation-receipt/v1",
        "status": "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_READY",
        "checkpoint": {"path": str(output / "latest.pt"), "sha256": sha256_file(output / "latest.pt")},
        "tokenizer": {"path": str(output / "tokenizer.json"), "sha256": sha256_file(output / "tokenizer.json")},
        "metrics": {"path": str(output / "metrics_rank_0.jsonl"), "sha256": sha256_file(output / "metrics_rank_0.jsonl")},
        "package": {"path": str(package), "sha256": sha256_file(package), "contents": list(archived_files)},
        "PERSISTENCE_ACTION_REQUIRED": "SAVE_KAGGLE_VERSION",
        "limits": {
            "checkpoint_loaded": False,
            "checkpoint_uploaded": False,
            "holdout_content_read": False,
            "promotion_authorized": False,
        },
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_READY")
    print(f"RECEIPT={receipt}")
    print(f"PACKAGE={package}")
    print("PERSISTENCE_ACTION_REQUIRED=SAVE_KAGGLE_VERSION")


if __name__ == "__main__":
    main()
