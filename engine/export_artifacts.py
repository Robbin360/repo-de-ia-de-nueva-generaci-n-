"""Empaqueta y exporta artefactos de entrenamiento a un destino persistente verificable.

No permite usar directorios efímeros de Kaggle como destino. Para Kaggle, use
el modo `kaggle-dataset`, que versiona un Dataset privado mediante el CLI del
usuario dentro del Notebook. La autenticación nunca se guarda en el repositorio.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path


EPHEMERAL_PREFIXES = ("/kaggle/working", "/content", "/tmp")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_persistent(destination: Path) -> None:
    resolved = destination.expanduser().resolve()
    if any(str(resolved).startswith(prefix) for prefix in EPHEMERAL_PREFIXES):
        raise ValueError(f"Destino efímero no permitido: {resolved}")


def package_artifacts(source: Path, staging: Path) -> tuple[Path, Path]:
    if not source.is_dir():
        raise FileNotFoundError(f"No existe el directorio de artefactos: {source}")
    staging.mkdir(parents=True, exist_ok=True)
    archive = staging / "aethel-artifacts.tar.gz"
    manifest = staging / "aethel-artifacts.manifest.json"
    with tarfile.open(archive, "w:gz") as bundle:
        bundle.add(source, arcname=source.name)
    contents = []
    for path in sorted(source.rglob("*")):
        if path.is_file():
            contents.append({"path": str(path.relative_to(source)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": str(source.resolve()),
        "archive": archive.name,
        "archive_sha256": sha256(archive),
        "files": contents,
    }, indent=2), encoding="utf-8")
    return archive, manifest


def export_filesystem(source: Path, destination: Path) -> dict:
    assert_persistent(destination)
    archive, manifest = package_artifacts(source, destination)
    return {"mode": "filesystem", "archive": str(archive), "manifest": str(manifest), "archive_sha256": sha256(archive)}


def export_kaggle_dataset(source: Path, staging: Path, dataset: str) -> dict:
    simulated = os.environ.get("AETHEL_KAGGLE_SIMULATION") == "1"
    if "/kaggle/working" not in str(staging.resolve()) and not simulated:
        raise ValueError("El staging de Kaggle debe estar en /kaggle/working para que se publique como Dataset.")
    archive, manifest = package_artifacts(source, staging)
    metadata = staging / "dataset-metadata.json"
    metadata.write_text(json.dumps({"id": dataset, "title": dataset.split("/")[-1], "isPrivate": True, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    command = ["kaggle", "datasets", "version", "-p", str(staging), "-m", f"Aethel artifacts {datetime.now(timezone.utc).isoformat()}"]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return {"mode": "kaggle-dataset", "dataset": dataset, "archive": str(archive), "manifest": str(manifest), "archive_sha256": sha256(archive), "cli_output": result.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--mode", choices=("filesystem", "kaggle-dataset"), required=True)
    parser.add_argument("--destination", type=Path, help="Directorio persistente montado para modo filesystem")
    parser.add_argument("--staging", type=Path, default=Path("/kaggle/working/aethel-persist-staging"))
    parser.add_argument("--dataset", help="Referencia privada owner/dataset para modo kaggle-dataset")
    args = parser.parse_args()
    if args.mode == "filesystem":
        if args.destination is None:
            parser.error("--destination es obligatorio para modo filesystem")
        result = export_filesystem(args.source, args.destination)
    else:
        if not args.dataset:
            parser.error("--dataset es obligatorio para modo kaggle-dataset")
        result = export_kaggle_dataset(args.source, args.staging, args.dataset)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
