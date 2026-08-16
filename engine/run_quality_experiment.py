"""Experimento reproducible de Aethel con corpus real, BPE y reporte de estabilidad.

No genera texto de entrenamiento: fragmenta un archivo real proporcionado por el usuario,
entrena el piloto breve y guarda las métricas emitidas por el runner.
"""
from __future__ import annotations

import argparse
import gzip
import json
import shutil
import subprocess
import sys
from pathlib import Path


def write_real_shard(corpus: Path, destination: Path, documents: int) -> Path:
    text = corpus.read_text(encoding="utf-8")
    parts = [part.strip() for part in text.split("\n\n") if len(part.strip()) >= 80]
    if not parts:
        parts = [text]
    selected = parts[:documents]
    if len(selected) < 2:
        selected = [text[: max(80, len(text) // 2)], text[max(80, len(text) // 2) :]]
    destination.mkdir(parents=True, exist_ok=True)
    shard = destination / "train-00000.jsonl.gz"
    with gzip.open(shard, "wt", encoding="utf-8") as handle:
        for item in selected:
            handle.write(json.dumps({"text": item, "source": str(corpus.resolve())}, ensure_ascii=False) + "\n")
    return shard


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True, help="Archivo de corpus real, ya revisado por el usuario.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--documents", type=int, default=512)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    output = Path(args.output).resolve()
    shutil.rmtree(output, ignore_errors=True)
    corpus_dir = output / "corpus"
    write_real_shard(Path(args.corpus), corpus_dir, args.documents)
    tokenizer = output / "tokenizer.json"
    run_dir = output / "run"
    subprocess.run([sys.executable, "engine/train_tokenizer.py", "--corpus-dir", str(corpus_dir), "--output", str(tokenizer), "--vocab-size", "512"], cwd=root, check=True)
    subprocess.run([
        sys.executable, "engine/train_aethel_gpu.py", "--corpus-dir", str(corpus_dir), "--tokenizer", str(tokenizer),
        "--output", str(run_dir), "--max-steps", str(args.steps), "--seq-len", "32", "--batch-size", "1",
        "--gradient-accumulation", "1", "--dim", "128", "--layers", "2", "--heads", "4", "--kv-heads", "1",
        "--experts", "4", "--active-experts", "2", "--precision", "fp32", "--observe-every", "1",
        "--replay-every", "2", "--replay-batch-size", "1", "--save-every", str(args.steps),
    ], cwd=root, check=True)
    events = [json.loads(line) for line in (run_dir / "metrics_rank_0.jsonl").read_text(encoding="utf-8").splitlines()]
    report = {
        "corpus": str(Path(args.corpus).resolve()),
        "steps": len(events),
        "first_loss": events[0]["loss"],
        "last_loss": events[-1]["loss"],
        "replay_events": sum(event["replay_loss"] is not None for event in events),
        "max_router_imbalance": max(event["router_health"]["max_imbalance"] for event in events),
        "minimum_router_entropy": min(event["router_health"]["min_entropy"] for event in events),
        "all_router_events_healthy": all(event["router_health"]["healthy"] for event in events),
        "checkpoint": str((run_dir / "latest.pt").resolve()),
        "metrics": str((run_dir / "metrics_rank_0.jsonl").resolve()),
    }
    report_path = output / "quality_experiment_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
