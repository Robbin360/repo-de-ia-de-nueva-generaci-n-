"""Carga estricta y generación mínima del checkpoint router-selection-jitter-v1.

El evaluador es deliberadamente estrecho: nunca construye optimizador, no lee
corpus ni holdout y no llama a rutas que persisten memoria o modifican pesos.
Su salida demuestra recuperabilidad técnica, no calidad del modelo.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

ENGINE_DIR = Path(__file__).resolve().parents[1] / "engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from aethel_nextgen import AethelNextGen, NextGenConfig


STATUS_READY = "CHECKPOINT_GENERATION_READY"
STATUS_BLOCKED = "CHECKPOINT_GENERATION_BLOCKED"
RECEIPT_NAME = "checkpoint_generation_receipt.json"
CONTROL_PROMPTS = (
    {"id": "es_control", "language": "es", "prompt": "Aethel responde:"},
    {"id": "en_control", "language": "en", "prompt": "Aethel replies:"},
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_parameters(model: torch.nn.Module) -> str:
    """Huella de parámetros, sin depender de una conversión NumPy de BF16."""
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
        raise RuntimeError("Se solicitó CUDA para la evaluación, pero CUDA no está disponible.")
    return device


def strict_payload(checkpoint_path: Path) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("El checkpoint debe ser un payload empaquetado, no un state_dict crudo.")
    required = ("model", "config", "step", "tokenizer", "tokenizer_sha256")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"El checkpoint no cumple el contrato de evaluación; faltan: {', '.join(missing)}.")
    if not isinstance(payload["model"], dict) or not payload["model"]:
        raise ValueError("El campo model debe contener un state_dict no vacío.")
    if not isinstance(payload["config"], dict):
        raise ValueError("El campo config debe ser un objeto serializado.")
    if not isinstance(payload["tokenizer_sha256"], str) or len(payload["tokenizer_sha256"]) != 64:
        raise ValueError("El checkpoint no incluye un hash SHA-256 válido del tokenizador.")
    return payload


def config_from_checkpoint(raw_config: dict[str, Any]) -> NextGenConfig:
    fields = set(NextGenConfig.__dataclass_fields__)
    unknown = sorted(set(raw_config) - fields)
    missing = sorted(fields - set(raw_config))
    if unknown or missing:
        details = []
        if unknown:
            details.append(f"campos no reconocidos: {', '.join(unknown)}")
        if missing:
            details.append(f"campos ausentes: {', '.join(missing)}")
        raise ValueError("La configuración del checkpoint no coincide de forma estricta con NextGenConfig (" + "; ".join(details) + ").")
    return NextGenConfig(**raw_config)


def controlled_generate(model: AethelNextGen, tokenizer: Any, prompt: dict[str, str], device: torch.device, max_new_tokens: int) -> dict[str, Any]:
    token_ids = list(tokenizer.encode(prompt["prompt"]).ids)
    if not token_ids:
        raise ValueError(f"El tokenizador no produjo tokens para {prompt['id']}.")
    context_ids = token_ids[-32:]
    emitted_ids: list[int] = []
    observed_jitter: list[float] = []
    finite_logits = True
    model.reset_session()
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            tokens = torch.tensor([context_ids], dtype=torch.long, device=device)
            logits, _, metrics = model(tokens)
            finite_logits = finite_logits and bool(torch.isfinite(logits).all().item())
            if not finite_logits:
                raise RuntimeError(f"Se detectaron logits no finitos durante {prompt['id']}.")
            observed = float(metrics.get("router_selection_jitter_noise", float("nan")))
            observed_jitter.append(observed)
            if observed != 0.0:
                raise RuntimeError(f"El jitter de selección se activó en evaluación ({observed}) durante {prompt['id']}.")
            next_token = int(torch.argmax(logits[:, -1, :], dim=-1).item())
            emitted_ids.append(next_token)
            context_ids = (context_ids + [next_token])[-64:]
    if not emitted_ids:
        raise RuntimeError(f"La generación controlada no emitió tokens para {prompt['id']}.")
    return {
        "id": prompt["id"],
        "language": prompt["language"],
        "prompt": prompt["prompt"],
        "prompt_token_count": len(token_ids),
        "generated_token_ids": emitted_ids,
        "generated_text": tokenizer.decode(emitted_ids),
        "finite_logits": finite_logits,
        "observed_router_selection_jitter_noise": observed_jitter,
    }


def evaluate_checkpoint(checkpoint_path: Path, output: Path, requested_device: str = "auto", max_new_tokens: int = 32) -> dict[str, Any]:
    """Ejecuta el contrato completo y escribe un único recibo sólo al tener éxito."""
    if not 1 <= max_new_tokens <= 64:
        raise ValueError("max_new_tokens debe estar entre 1 y 64.")
    if output.exists():
        raise FileExistsError(f"La salida debe ser inédita y no existe esa garantía: {output}")

    payload = strict_payload(checkpoint_path)
    checkpoint_hash_before = sha256_file(checkpoint_path)
    tokenizer_path = Path(str(payload["tokenizer"]))
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"No existe el tokenizador citado por el checkpoint: {tokenizer_path}")
    tokenizer_hash = sha256_file(tokenizer_path)
    if tokenizer_hash != payload["tokenizer_sha256"]:
        raise ValueError("El hash del tokenizador no coincide con el hash registrado por el checkpoint.")

    from tokenizers import Tokenizer

    config = config_from_checkpoint(payload["config"])
    device = resolve_device(requested_device)
    output.mkdir(parents=True, exist_ok=False)
    memory_path = output / "volatile_memory" / "episodic_memory.jsonl"
    model = AethelNextGen(config, memory_path=memory_path).to(device)
    try:
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        if model.training:
            raise RuntimeError("El modelo no entró en modo eval para la generación controlada.")
        parameter_fingerprint_before = fingerprint_parameters(model)
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
        generations = [controlled_generate(model, tokenizer, prompt, device, max_new_tokens) for prompt in CONTROL_PROMPTS]
        parameter_fingerprint_after = fingerprint_parameters(model)
        if parameter_fingerprint_before != parameter_fingerprint_after:
            raise RuntimeError("Los parámetros cambiaron durante inferencia; la evaluación queda bloqueada.")
        checkpoint_hash_after = sha256_file(checkpoint_path)
        if checkpoint_hash_before != checkpoint_hash_after:
            raise RuntimeError("El archivo del checkpoint cambió durante la evaluación; la evaluación queda bloqueada.")
        persistent_memory_paths = (
            memory_path,
            memory_path.with_name("liquid_versions.jsonl"),
            memory_path.with_name("curiosity_events.jsonl"),
            memory_path.with_name("semantic_memory.jsonl"),
        )
        written_memory_paths = [str(path) for path in persistent_memory_paths if path.exists()]
        if written_memory_paths:
            raise RuntimeError("La inferencia intentó persistir memoria: " + ", ".join(written_memory_paths))
        report = {
            "schema": "aethel-router-jitter-checkpoint-evaluation/v1",
            "status": STATUS_READY,
            "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_sha256_before": checkpoint_hash_before,
            "checkpoint_sha256_after": checkpoint_hash_after,
            "step": int(payload["step"]),
            "config": asdict(config),
            "tokenizer": str(tokenizer_path.resolve()),
            "tokenizer_sha256": tokenizer_hash,
            "device": str(device),
            "model_training": bool(model.training),
            "parameter_fingerprint_before": parameter_fingerprint_before,
            "parameter_fingerprint_after": parameter_fingerprint_after,
            "max_new_tokens": max_new_tokens,
            "generations": generations,
            "limits": {
                "training_started": False,
                "optimizer_created": False,
                "holdout_content_read": False,
                "raw_corpus_read": False,
                "network_requests": 0,
                "checkpoint_modified": False,
                "persistent_memory_written": False,
                "promotion_authorized": False,
            },
            "interpretation": "Carga e inferencia mínima verificadas; no constituye validación de calidad, razonamiento, bilingüismo, eficiencia ni promoción.",
        }
        receipt_path = output / RECEIPT_NAME
        receipt_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report
    except Exception:
        # La carpeta queda intacta como evidencia de error y no se reutiliza.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Carga estricta y generación mínima del checkpoint Aethel jitter.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()
    try:
        report = evaluate_checkpoint(args.checkpoint, args.output, args.device, args.max_new_tokens)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": STATUS_BLOCKED, "reason": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
    print(json.dumps({"status": report["status"], "receipt": str(args.output / RECEIPT_NAME)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
