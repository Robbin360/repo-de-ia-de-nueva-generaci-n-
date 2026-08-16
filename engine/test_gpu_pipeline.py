"""Prueba mínima de corpus preparado, tokenizador BPE y runner GPU en fallback CPU."""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from export_artifacts import export_filesystem


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="aethel-gpu-pipeline-") as temporary:
        temporary_path = Path(temporary)
        corpus = temporary_path / "corpus"
        corpus.mkdir()
        shard = corpus / "train-00000.jsonl.gz"
        records = [
            {"text": "Aethel integra La Roca, El Líquido y expertos dispersos para aprender con eficiencia."},
            {"text": "El Ciclo de Sueño consolida replay y el Espacio de Trabajo Global integra contextos."},
            {"text": "RoPE y GQA reducen el coste de atención mientras Neuromodulación prioriza observaciones."},
            {"text": "La memoria episódica recupera estados útiles sin modificar pesos de forma silenciosa."},
        ]
        with gzip.open(shard, "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tokenizer = temporary_path / "tokenizer.json"
        subprocess.run([sys.executable, "engine/train_tokenizer.py", "--corpus-dir", str(corpus), "--output", str(tokenizer), "--vocab-size", "128"], cwd=root, check=True)
        run_dir = temporary_path / "run"
        subprocess.run([
            sys.executable,
            "engine/train_aethel_gpu.py",
            "--corpus-dir", str(corpus),
            "--tokenizer", str(tokenizer),
            "--output", str(run_dir),
            "--max-steps", "2",
            "--seq-len", "8",
            "--batch-size", "1",
            "--gradient-accumulation", "1",
            "--dim", "64",
            "--layers", "1",
            "--heads", "4",
            "--kv-heads", "1",
            "--experts", "2",
            "--active-experts", "1",
            "--precision", "fp32",
            "--save-every", "1",
        ], cwd=root, check=True)
        assert (run_dir / "latest.pt").exists(), "No se creó el checkpoint reanudable"
        assert (run_dir / "metrics_rank_0.jsonl").read_text(encoding="utf-8").count("\n") == 2, "No se registraron métricas reales de ambos pasos"
        destination = Path.home() / "aethel-persistent-e2e-artifacts"
        shutil.rmtree(destination, ignore_errors=True)
        exported = export_filesystem(run_dir, destination)
        manifest = json.loads(Path(exported["manifest"]).read_text(encoding="utf-8"))
        recorded = {item["path"] for item in manifest["files"]}
        assert Path(exported["archive"]).is_file(), "No se creó el archivo persistente"
        assert "latest.pt" in recorded and "metrics_rank_0.jsonl" in recorded, "Faltan checkpoint o métricas en el manifiesto persistente"
        print(json.dumps({"runner_persistence_verified": True, "archive": exported["archive"], "files": len(recorded)}))


if __name__ == "__main__":
    main()
