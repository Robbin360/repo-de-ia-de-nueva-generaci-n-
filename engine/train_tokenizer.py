"""Entrena un tokenizador BPE versionado desde los JSONL preparados por prepare_corpus.py."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


def iter_texts(corpus_dir: Path):
    for path in sorted(corpus_dir.glob("*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("text"):
                    yield record["text"]


def run(args: argparse.Namespace) -> None:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

    corpus_dir = Path(args.corpus_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFKC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=args.vocab_size, min_frequency=args.min_frequency, special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"])
    tokenizer.train_from_iterator(iter_texts(corpus_dir), trainer=trainer, length=args.max_documents or None)
    tokenizer.save(str(output))
    payload = output.read_bytes()
    manifest = {"path": str(output), "sha256": hashlib.sha256(payload).hexdigest(), "vocab_size": tokenizer.get_vocab_size(), "corpus_dir": str(corpus_dir), "requested_vocab_size": args.vocab_size}
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--vocab-size", type=int, default=32000)
    parser.add_argument("--min-frequency", type=int, default=2)
    parser.add_argument("--max-documents", type=int, default=0)
    run(parser.parse_args())

