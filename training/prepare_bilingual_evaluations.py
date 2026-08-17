"""Descarga de forma explícita benchmarks retenidos; no los convierte en corpus."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = {
    "mgsm": {"dataset": "juletxara/mgsm", "revision": "b2f13d426afe3be8d69a7e739b36724db8b66bbc", "languages": ["en", "es"], "license": "CC BY-SA 4.0"},
    "belebele": {"dataset": "facebook/belebele", "revision": "7899cdfa4e1e0d733fd77c848e2c273cb1d32be2", "languages": ["eng_Latn", "spa_Latn"], "license": "CC BY-SA 4.0"},
}


def dump_rows(path: Path, rows) -> dict:
    with path.open("w", encoding="utf-8") as handle:
        count = 0
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return {"path": path.name, "rows": count, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def run(args: argparse.Namespace) -> None:
    if not args.allow_network:
        raise RuntimeError("La descarga de benchmarks retenidos exige --allow-network.")
    from datasets import load_dataset

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict = {"purpose": "Evaluación retenida; prohibido usar estos archivos para entrenamiento o tokenizador.", "assets": []}
    for language in SOURCES["mgsm"]["languages"]:
        for split in ("train", "test"):
            data = load_dataset(SOURCES["mgsm"]["dataset"], language, split=split, revision=SOURCES["mgsm"]["revision"], trust_remote_code=False)
            item = dump_rows(output / f"mgsm-{language}-{split}.jsonl", data)
            item.update({"benchmark": "mgsm", "language": language, "split": split, "license": SOURCES["mgsm"]["license"]})
            manifest["assets"].append(item)
    for language in SOURCES["belebele"]["languages"]:
        data = load_dataset(SOURCES["belebele"]["dataset"], language, split="test", revision=SOURCES["belebele"]["revision"], trust_remote_code=False)
        item = dump_rows(output / f"belebele-{language}-test.jsonl", data)
        item.update({"benchmark": "belebele", "language": language, "split": "test", "license": SOURCES["belebele"]["license"]})
        manifest["assets"].append(item)
    (output / "evaluation_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-network", action="store_true")
    run(parser.parse_args())
