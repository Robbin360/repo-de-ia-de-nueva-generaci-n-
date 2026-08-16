"""Prueba de integración local de exportación: el destino debe estar fuera de rutas efímeras."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from export_artifacts import EPHEMERAL_PREFIXES, export_filesystem


def main() -> None:
    with tempfile.TemporaryDirectory() as root:
        root_path = Path(root)
        source = root_path / "run"
        source.mkdir()
        (source / "latest.pt").write_bytes(b"verified-checkpoint")
        (source / "metrics.jsonl").write_text('{"step":1,"loss":1.0}\n', encoding="utf-8")
        destination = Path.home() / "aethel-persistent-export-test"
        if any(str(destination.resolve()).startswith(prefix) for prefix in EPHEMERAL_PREFIXES):
            raise RuntimeError("El destino de la prueba no puede ser efímero")
        result = export_filesystem(source, destination)
        manifest = json.loads(Path(result["manifest"]).read_text(encoding="utf-8"))
        assert Path(result["archive"]).is_file()
        assert len(manifest["files"]) == 2
        print(json.dumps({"persistent_export_verified": True, "archive_sha256": result["archive_sha256"], "files": len(manifest["files"])}))


if __name__ == "__main__":
    main()

