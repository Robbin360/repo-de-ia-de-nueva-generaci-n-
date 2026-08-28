"""Verifica la cadencia compacta de telemetría sin abrir datos reales ni usar GPU."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "engine" / "train_aethel_gpu.py"
SUMMARY = ROOT / "training" / "summarize_d1a_router_metrics.py"


def fixture(root: Path) -> tuple[Path, Path, Path]:
    corpus = root / "corpus"
    corpus.mkdir()
    (corpus / "train-fixture.jsonl").write_text(
        json.dumps({"text": "a b c d e f g h " * 16}) + "\n",
        encoding="utf-8",
    )
    manifest = root / "prepared_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "fixture": True}) + "\n", encoding="utf-8")
    tokenizer = Tokenizer(
        models.WordLevel(
            {"[UNK]": 0, **{token: index for index, token in enumerate("a b c d e f g h".split(), start=1)}},
            unk_token="[UNK]",
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return corpus, manifest, tokenizer_path


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus, manifest, tokenizer = fixture(root)
        output = root / "output"
        command = [
            "python3", str(TRAINER),
            "--corpus-dir", str(corpus), "--tokenizer", str(tokenizer), "--output", str(output),
            "--data-manifest", str(manifest), "--max-steps", "5", "--schedule-total-steps", "5",
            "--seq-len", "4", "--batch-size", "1", "--gradient-accumulation", "1",
            "--dim", "16", "--layers", "1", "--heads", "4", "--kv-heads", "2",
            "--experts", "2", "--active-experts", "1", "--memory-slots", "4", "--replay-capacity", "4",
            "--learning-rate", "0.001", "--min-learning-rate", "0.0001", "--warmup-steps", "1",
            "--precision", "fp32", "--seed", "29", "--save-every", "2", "--metrics-every", "2",
            "--console-every", "99", "--observe-every", "99", "--allow-pytorch-fallback",
        ]
        subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
        events = [json.loads(line) for line in (output / "metrics_rank_0.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [event["step"] for event in events] == [2, 4, 5]
        assert [event["telemetry_window"]["micro_steps"] for event in events] == [2, 2, 1]
        assert sum(event["telemetry_window"]["micro_steps"] for event in events) == 5
        for event in events:
            window = event["telemetry_window"]
            assert window["router_health"]["healthy_micro_steps"] + window["router_health"]["unhealthy_micro_steps"] == window["micro_steps"]

        diagnostic = output / "router_diagnostic.json"
        subprocess.run(
            ["python3", str(SUMMARY), "--metrics", str(output / "metrics_rank_0.jsonl"), "--output", str(diagnostic), "--diagnostic-id", "AETHEL_EDGE_LONG_PHASE_V1"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        report = json.loads(diagnostic.read_text(encoding="utf-8"))
        assert report["steps"] == {"first": 2, "last": 5, "telemetry_events": 3}
        assert report["router_health"]["healthy_micro_steps"] + report["router_health"]["unhealthy_micro_steps"] == 5
    print("AETHEL_COMPACT_TELEMETRY_CONTRACT_OK")


if __name__ == "__main__":
    main()
