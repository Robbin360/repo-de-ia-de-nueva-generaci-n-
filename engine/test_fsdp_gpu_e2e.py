"""Validación FSDP real: dos GPU CUDA, checkpoint rango 0 y reanudación.

La prueba no emula FSDP en CPU. Si el host no dispone de dos GPU CUDA, informa
el requisito y termina sin declarar una validación exitosa.
"""
from __future__ import annotations

import gzip
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def run_torchrun(root: Path, corpus: Path, tokenizer: Path, output: Path, max_steps: int, resume: bool = False) -> None:
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc_per_node=2",
        "engine/train_aethel_gpu.py",
        "--strategy",
        "fsdp",
        "--corpus-dir",
        str(corpus),
        "--tokenizer",
        str(tokenizer),
        "--output",
        str(output),
        "--max-steps",
        str(max_steps),
        "--seq-len",
        "8",
        "--batch-size",
        "1",
        "--gradient-accumulation",
        "1",
        "--dim",
        "64",
        "--layers",
        "1",
        "--heads",
        "4",
        "--kv-heads",
        "1",
        "--experts",
        "2",
        "--active-experts",
        "1",
        "--precision",
        "fp32",
        "--save-every",
        "1",
        "--observe-every",
        "1",
        "--replay-every",
        "1",
    ]
    if resume:
        command.append("--resume")
    subprocess.run(command, cwd=root, check=True)


def main() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print(json.dumps({"fsdp_e2e": "SKIPPED", "reason": "requires_at_least_two_cuda_gpus", "cuda_devices": torch.cuda.device_count()}))
        return

    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="aethel-fsdp-e2e-") as temporary:
        temporary_path = Path(temporary)
        corpus = temporary_path / "corpus"
        corpus.mkdir()
        records = [
            {"text": "Aethel usa La Roca y El Líquido para combinar estabilidad y plasticidad controlada."},
            {"text": "La atención RoPE y GQA conserva contexto con menor uso de caché de claves y valores."},
            {"text": "Los expertos dispersos se monitorizan con entropía, carga y desbalance del router."},
            {"text": "El Ciclo de Sueño usa replay diverso para consolidar secuencias relevantes."},
            {"text": "La memoria episódica recupera estados por similitud y conserva trazabilidad."},
            {"text": "La neuromodulación usa sorpresa para priorizar observaciones sin alterar pesos de forma oculta."},
        ]
        with gzip.open(corpus / "train-00000.jsonl.gz", "wt", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        tokenizer = temporary_path / "tokenizer.json"
        subprocess.run([sys.executable, "engine/train_tokenizer.py", "--corpus-dir", str(corpus), "--output", str(tokenizer), "--vocab-size", "128"], cwd=root, check=True)
        output = temporary_path / "run"
        run_torchrun(root, corpus, tokenizer, output, max_steps=1)
        checkpoint = output / "latest.pt"
        assert checkpoint.is_file(), "El rango 0 no escribió latest.pt durante la primera corrida FSDP"
        first = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert first["strategy"] == "fsdp" and first["step"] == 1
        run_torchrun(root, corpus, tokenizer, output, max_steps=2, resume=True)
        resumed = torch.load(checkpoint, map_location="cpu", weights_only=False)
        assert resumed["strategy"] == "fsdp" and resumed["step"] == 2, "La reanudación FSDP no avanzó el checkpoint"
        assert (output / "metrics_rank_0.jsonl").read_text(encoding="utf-8").count("\n") == 2
        assert (output / "metrics_rank_1.jsonl").is_file(), "Falta telemetría distribuida del rango 1"
        print(json.dumps({"fsdp_e2e": "OK", "world_size": 2, "checkpoint_step": 2, "output": str(output)}))
        shutil.rmtree(output, ignore_errors=True)


if __name__ == "__main__":
    main()
