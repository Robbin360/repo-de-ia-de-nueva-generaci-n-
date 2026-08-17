"""Valida que la comparación ARC use corpus real, mismas muestras y reporte explícito."""
import json
import tempfile
from pathlib import Path

from compare_arc_baseline import run


class Args:
    corpus = "engine/corpora/aethel_repo_corpus.txt"
    steps = 3
    batch_size = 1
    seq_len = 16
    dim = 32
    layers = 1
    heads = 4
    kv_heads = 1
    experts = 2
    arc_steps = 2
    arc_threshold = 0.35
    arc_penalty = 0.001
    seed = 17


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        Args.output = str(Path(directory) / "comparison.json")
        result = run(Args)
        saved = json.loads(Path(Args.output).read_text(encoding="utf-8"))
        assert result["protocol"] == "same-corpus-same-batches-same-seed"
        assert saved["corpus_sha256"] == result["corpus_sha256"]
        assert result["arc"]["parameters"] > result["baseline"]["parameters"]
        assert result["arc"]["adaptive_effective_steps_total"] > 0
        assert result["baseline"]["peak_rss_bytes"] > 0
        assert result["arc"]["peak_rss_bytes"] > 0
        assert "no autoriza" in result["interpretation"]
    print("OK: comparación ARC-baseline reproducible")


if __name__ == "__main__":
    main()
