"""Evaluación aislada del checkpoint preservado de Aethel Edge.

La herramienta carga pesos sólo para inferencia y pérdida sobre el holdout
preparado. Nunca construye optimizador, llama backward, reanuda, observa,
persiste memoria ni escribe junto al checkpoint de entrada.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import torch

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from aethel_nextgen import AethelNextGen, NextGenConfig
from aethel_resume import RESUME_SCHEMA, require_full_resume_payload, sha256_file


STATUS_READY = "AETHEL_EDGE_EVALUATION_READY"
STATUS_BLOCKED = "AETHEL_EDGE_EVALUATION_BLOCKED"
RECEIPT_NAME = "edge_evaluation_receipt.json"
CONTROL_PROMPTS = (
    {"id": "es_control", "language": "es", "prompt": "Aethel responde:"},
    {"id": "en_control", "language": "en", "prompt": "Aethel replies:"},
)


def records(path: Path) -> Iterator[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and isinstance(row.get("text"), str) and row["text"].strip():
                yield row


def parameter_fingerprint(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        tensor = parameter.detach().contiguous().view(torch.uint8).cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(parameter.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó CUDA para evaluar Edge, pero CUDA no está disponible.")
    return device


def config_from_checkpoint(raw_config: Any) -> NextGenConfig:
    if not isinstance(raw_config, dict):
        raise ValueError("El checkpoint no contiene una configuración serializada válida.")
    fields = set(NextGenConfig.__dataclass_fields__)
    unknown = sorted(set(raw_config) - fields)
    missing = sorted(fields - set(raw_config))
    if unknown or missing:
        raise ValueError(
            "La configuración del checkpoint no coincide estrictamente con NextGenConfig "
            f"(desconocidos={unknown}, ausentes={missing})."
        )
    return NextGenConfig(**raw_config)


def load_and_validate(
    checkpoint_path: Path, tokenizer_path: Path, data_manifest_path: Path, validation_path: Path
) -> tuple[dict[str, Any], NextGenConfig, dict[str, Any]]:
    if not checkpoint_path.is_file() or not tokenizer_path.is_file() or not data_manifest_path.is_file() or not validation_path.is_file():
        raise FileNotFoundError("Falta checkpoint, tokenizador, manifiesto o holdout de evaluación.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise ValueError("El checkpoint Edge debe ser un payload completo.")
    require_full_resume_payload(checkpoint)
    required = ("config", "tokenizer", "tokenizer_sha256")
    missing = [field for field in required if field not in checkpoint]
    if missing:
        raise ValueError("El payload no cumple el contrato de evaluación; faltan: " + ", ".join(missing))
    resume_contract = checkpoint["resume_contract"]
    if not isinstance(resume_contract, dict) or resume_contract.get("schema") != RESUME_SCHEMA:
        raise ValueError("El checkpoint no declara el contrato de reanudación Edge v2.")
    if resume_contract.get("config") != checkpoint["config"]:
        raise ValueError("La configuración del contrato no coincide con la del checkpoint.")
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != checkpoint["tokenizer_sha256"] or tokenizer_hash != resume_contract.get("tokenizer_sha256"):
        raise ValueError("El hash del tokenizador de entrada no coincide con el checkpoint Edge.")
    manifest_hash = sha256_file(data_manifest_path)
    if manifest_hash != resume_contract.get("data_manifest_sha256"):
        raise ValueError("El manifiesto de datos no coincide con el contrato guardado del checkpoint.")
    config = config_from_checkpoint(checkpoint["config"])
    return checkpoint, config, {
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "tokenizer_sha256": tokenizer_hash,
        "prepared_manifest_sha256": manifest_hash,
        "resume_schema": resume_contract["schema"],
    }


def controlled_generate(
    model: AethelNextGen, tokenizer: Any, prompt: dict[str, str], device: torch.device, max_new_tokens: int
) -> dict[str, Any]:
    token_ids = list(tokenizer.encode(prompt["prompt"]).ids)
    if not token_ids:
        raise ValueError(f"El tokenizador no produjo tokens para {prompt['id']}.")
    context_ids, emitted = token_ids[-32:], []
    jitter_values: list[float] = []
    model.reset_session()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            logits, _, metrics = model(torch.tensor([context_ids], dtype=torch.long, device=device))
            if not bool(torch.isfinite(logits).all().item()):
                raise RuntimeError(f"Logits no finitos durante la generación {prompt['id']}.")
            jitter = float(metrics.get("router_selection_jitter_noise", float("nan")))
            if jitter != 0.0:
                raise RuntimeError(f"El jitter del router se activó durante evaluación: {jitter}.")
            jitter_values.append(jitter)
            next_id = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            emitted.append(next_id)
            context_ids = (context_ids + [next_id])[-64:]
    return {
        "id": prompt["id"],
        "language": prompt["language"],
        "prompt": prompt["prompt"],
        "generated_token_ids": emitted,
        "generated_text": tokenizer.decode(emitted),
        "observed_router_selection_jitter_noise": jitter_values,
    }


def evaluate_checkpoint(
    *,
    checkpoint_path: Path,
    tokenizer_path: Path,
    data_manifest_path: Path,
    validation_path: Path,
    output: Path,
    requested_device: str = "auto",
    seq_len: int = 1024,
    max_segments_per_language: int = 256,
    max_new_tokens: int = 32,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"La salida de evaluación debe ser inédita: {output}")
    if not 1 <= max_new_tokens <= 64 or not 1 <= max_segments_per_language <= 256:
        raise ValueError("Los límites de evaluación solicitados no son válidos.")
    checkpoint, config, provenance = load_and_validate(checkpoint_path, tokenizer_path, data_manifest_path, validation_path)
    if seq_len != config.max_seq_len:
        raise ValueError("La evaluación Edge exige el mismo max_seq_len declarado por el checkpoint.")
    from tokenizers import Tokenizer

    device = resolve_device(requested_device)
    output.mkdir(parents=True, exist_ok=False)
    memory_path = output / "volatile_memory" / "episodic_memory.jsonl"
    model = AethelNextGen(config, memory_path=memory_path).to(device)
    try:
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval()
        if model.training:
            raise RuntimeError("El modelo no entró en modo eval.")
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        fingerprint_before = parameter_fingerprint(model)
        losses: dict[str, list[float]] = defaultdict(list)
        with torch.inference_mode():
            for row in records(validation_path):
                language = str(row.get("language", "unknown"))
                if language not in ("en", "es") or len(losses[language]) >= max_segments_per_language:
                    continue
                ids = tokenizer.encode(row["text"]).ids
                for start in range(0, len(ids) - seq_len, seq_len):
                    if len(losses[language]) >= max_segments_per_language:
                        break
                    x = torch.tensor(ids[start : start + seq_len], dtype=torch.long, device=device).unsqueeze(0)
                    y = torch.tensor(ids[start + 1 : start + seq_len + 1], dtype=torch.long, device=device).unsqueeze(0)
                    model.reset_session()
                    _, loss, _ = model(x, y)
                    if loss is None or not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f"Pérdida no finita en validación {language}.")
                    losses[language].append(float(loss.float().cpu()))
                if all(len(losses[item]) >= max_segments_per_language for item in ("en", "es")):
                    break
        if any(len(losses[language]) != max_segments_per_language for language in ("en", "es")):
            raise RuntimeError("El holdout no produjo el número autorizado de segmentos para ambos idiomas.")
        generations = [controlled_generate(model, tokenizer, prompt, device, max_new_tokens) for prompt in CONTROL_PROMPTS]
        fingerprint_after = parameter_fingerprint(model)
        if fingerprint_before != fingerprint_after:
            raise RuntimeError("Los parámetros cambiaron durante evaluación; el resultado queda bloqueado.")
        memory_paths = (
            memory_path,
            memory_path.with_name("liquid_versions.jsonl"),
            memory_path.with_name("curiosity_events.jsonl"),
            memory_path.with_name("semantic_memory.jsonl"),
        )
        written_memory = [str(path) for path in memory_paths if path.exists()]
        if written_memory:
            raise RuntimeError("La evaluación intentó persistir memoria: " + ", ".join(written_memory))
        by_language = {
            language: {
                "segments": len(values),
                "loss": sum(values) / len(values),
                "perplexity": math.exp(min(sum(values) / len(values), 20.0)),
            }
            for language, values in sorted(losses.items())
        }
        all_losses = [value for values in losses.values() for value in values]
        report = {
            "schema": "aethel-edge-checkpoint-evaluation/v1",
            "status": STATUS_READY,
            "checkpoint": str(checkpoint_path.resolve()),
            "step": int(checkpoint["step"]),
            "device": str(device),
            "config": asdict(config),
            "provenance": provenance,
            "split": "prepared_validation_holdout",
            "segments": len(all_losses),
            "max_segments_per_language": max_segments_per_language,
            "loss": sum(all_losses) / len(all_losses),
            "perplexity": math.exp(min(sum(all_losses) / len(all_losses), 20.0)),
            "by_language": by_language,
            "max_new_tokens": max_new_tokens,
            "generations": generations,
            "parameter_fingerprint_before": fingerprint_before,
            "parameter_fingerprint_after": fingerprint_after,
            "limits": {
                "training_started": False,
                "optimizer_created": False,
                "backward_called": False,
                "checkpoint_modified": False,
                "persistent_memory_written": False,
                "resume_started": False,
                "network_requests": 0,
                "promotion_authorized": False,
            },
            "integrity_note": "Se validaron checkpoint, tokenizador y manifiesto. Kaggle expone el holdout descomprimido; no se afirma un hash independiente de los gzip originales.",
            "interpretation": "Mide pérdida y generación mínima controlada; no demuestra razonamiento, bilingüismo nativo, eficiencia, seguridad ni promoción.",
        }
        (output / RECEIPT_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception:
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación aislada de Aethel Edge.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-segments-per-language", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    try:
        report = evaluate_checkpoint(**vars(args))
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": STATUS_BLOCKED, "reason": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
    print(json.dumps({"status": report["status"], "receipt": str(args.output / RECEIPT_NAME)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
