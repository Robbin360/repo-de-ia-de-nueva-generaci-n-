"""Pruebas del validador que protege la corrida GPU frente a manifiestos inválidos."""
import json
import os
import subprocess
import tempfile
from pathlib import Path

from validate_training_readiness import validate


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    report = validate(root / "training/corpus_manifest.example.json", root / "TRAINING_CURRICULUM.md", root / "training/BENCHMARK_PROTOCOL.md")
    assert report["status"] == "READY_FOR_HUMAN_APPROVAL"
    assert report["warnings"]
    with tempfile.TemporaryDirectory() as directory:
        root_path = Path(directory)
        invalid = Path(directory) / "manifest.json"
        invalid.write_text(json.dumps({"approval_required": False, "sources": [], "filters": {}}), encoding="utf-8")
        blocked = validate(invalid, root / "TRAINING_CURRICULUM.md", root / "training/BENCHMARK_PROTOCOL.md")
        assert blocked["status"] == "BLOCKED"
        assert blocked["errors"]
        approved = root_path / "approved_manifest.json"
        approved.write_text(json.dumps({"approval_required": True, "sources": [{"id": "approved", "dataset": "example/dataset", "split": "train", "text_column": "text", "revision": "abcdef1234567", "license_review": "aprobada", "provenance_url": "https://example.invalid", "enabled": True, "approved": True}], "filters": {"remove_simple_pii": True, "deduplicate_exact": True, "deduplicate_near": "required-at-scale", "exclude_evaluation_sets": True}}), encoding="utf-8")
        holdout = root_path / "holdout.jsonl"
        tokenizer = root_path / "tokenizer.json"
        benchmark = root_path / "mmlu.jsonl"
        for path in (holdout, tokenizer, benchmark):
            path.write_text("{}\n", encoding="utf-8")
        evaluation = root_path / "evaluation.json"
        evaluation.write_text(json.dumps({"approved": True, "holdout_path": str(holdout), "tokenizer_path": str(tokenizer), "seed": 17, "benchmark_references": {"mmlu": str(benchmark)}}), encoding="utf-8")
        accepted = validate(approved, root / "TRAINING_CURRICULUM.md", root / "training/BENCHMARK_PROTOCOL.md", evaluation, True)
        assert accepted["status"] == "READY_FOR_HUMAN_APPROVAL"
        evaluation.write_text(json.dumps({"approved": True, "holdout_path": str(root_path / "missing-holdout.jsonl"), "tokenizer_path": str(tokenizer), "seed": 17, "benchmark_references": {"mmlu": str(benchmark)}}), encoding="utf-8")
        inaccessible = validate(approved, root / "TRAINING_CURRICULUM.md", root / "training/BENCHMARK_PROTOCOL.md", evaluation, True)
        assert inaccessible["status"] == "BLOCKED"
        assert any("ruta de evaluación inaccesible" in error for error in inaccessible["errors"])
        data_dir = root_path / "data"
        run_dir = root_path / "runs"
        data_dir.mkdir()
        run_dir.mkdir()
        (data_dir / "corpus_manifest.json").write_text(invalid.read_text(encoding="utf-8"), encoding="utf-8")
        launcher = subprocess.run(["bash", "training/run_aethel_gpu.sh"], cwd=root, env={**os.environ, "AETHEL_DATA_DIR": str(data_dir), "AETHEL_RUN_DIR": str(run_dir), "AETHEL_EVALUATION_CONFIG": str(evaluation), "AETHEL_MAX_DOCUMENTS": "1"}, capture_output=True, text=True, timeout=15)
        assert launcher.returncode == 2
        assert "BLOCKED" in launcher.stdout
        assert not (run_dir / "prepared").exists()
    print("OK: validación de preparación de entrenamiento")


if __name__ == "__main__":
    main()
