"""Prueba CPU de equivalencia entre entrenamiento continuo y entrenamiento reanudado."""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "engine" / "train_aethel_gpu.py"


def create_fixture(root: Path) -> tuple[Path, Path, Path]:
    corpus = root / "corpus"
    corpus.mkdir()
    tokens = "a b c d e f g h " * 16
    (corpus / "train-en.jsonl").write_text(json.dumps({"text": tokens}) + "\n", encoding="utf-8")
    manifest = root / "package_manifest.json"
    manifest.write_text(json.dumps({"dataset_id": "resume-e2e-fixture", "revision": 1}) + "\n", encoding="utf-8")

    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, **{token: index for index, token in enumerate("a b c d e f g h".split(), start=1)}}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer_path = root / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    return corpus, manifest, tokenizer_path


def trainer_command(corpus: Path, manifest: Path, tokenizer: Path, output: Path, max_steps: int, resume: bool) -> list[str]:
    command = [
        "python3", str(TRAINER),
        "--corpus-dir", str(corpus), "--tokenizer", str(tokenizer), "--output", str(output),
        "--max-steps", str(max_steps), "--schedule-total-steps", "4",
        "--seq-len", "4", "--batch-size", "1", "--gradient-accumulation", "1",
        "--dim", "16", "--layers", "1", "--heads", "4", "--kv-heads", "2",
        "--experts", "2", "--active-experts", "1", "--memory-slots", "4", "--replay-capacity", "4",
        "--learning-rate", "0.001", "--min-learning-rate", "0.0001", "--warmup-steps", "1",
        "--precision", "fp32", "--seed", "23", "--save-every", "1", "--observe-every", "99",
        "--allow-pytorch-fallback", "--data-manifest", str(manifest),
    ]
    if resume:
        command.append("--resume")
    return command


def assert_tensor_mapping_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor], label: str) -> None:
    assert left.keys() == right.keys(), f"Claves distintas en {label}"
    for key in left:
        assert torch.equal(left[key], right[key]), f"Tensor distinto en {label}: {key}"


def assert_nested_equal(left: object, right: object, label: str) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        assert torch.equal(left, right), f"Tensor distinto en {label}"
        return
    if isinstance(left, dict) and isinstance(right, dict):
        assert left.keys() == right.keys(), f"Claves distintas en {label}"
        for key in left:
            assert_nested_equal(left[key], right[key], f"{label}.{key}")
        return
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        assert len(left) == len(right), f"Longitud distinta en {label}"
        for index, (item_left, item_right) in enumerate(zip(left, right)):
            assert_nested_equal(item_left, item_right, f"{label}[{index}]")
        return
    assert left == right, f"Valor distinto en {label}"


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        corpus, manifest, tokenizer = create_fixture(root)
        output = root / "run"
        subprocess.run(trainer_command(corpus, manifest, tokenizer, output, 2, resume=False), cwd=ROOT, check=True, text=True)
        first = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        assert first["step"] == 2
        assert first["schedule_total_steps"] == 4
        assert first["resume_contract"]["trainer_profile"]["schedule_total_steps"] == 4

        subprocess.run(trainer_command(corpus, manifest, tokenizer, output, 4, resume=True), cwd=ROOT, check=True, text=True)
        resumed = torch.load(output / "latest.pt", map_location="cpu", weights_only=False)
        assert resumed["step"] == 4
        assert resumed["session_target_step"] == 4
        assert "optimizer" in resumed and "rng_state" in resumed and "runtime_state" in resumed
        steps = [json.loads(line)["step"] for line in (output / "metrics_rank_0.jsonl").read_text(encoding="utf-8").splitlines()]
        assert steps == [1, 2, 3, 4]

        continuous_output = root / "continuous"
        subprocess.run(
            trainer_command(corpus, manifest, tokenizer, continuous_output, 4, resume=False),
            cwd=ROOT,
            check=True,
            text=True,
        )
        continuous = torch.load(continuous_output / "latest.pt", map_location="cpu", weights_only=False)
        assert continuous["step"] == 4
        assert_tensor_mapping_equal(continuous["model"], resumed["model"], "modelo")
        assert_nested_equal(continuous["optimizer"], resumed["optimizer"], "optimizador")
        assert torch.equal(continuous["rng_state"]["torch_cpu"], resumed["rng_state"]["torch_cpu"])
    print("AETHEL_TRAINING_RESUME_EQUIVALENCE_VERIFIED")


if __name__ == "__main__":
    main()
