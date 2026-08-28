"""Inspecciona checkpoints sin construir un modelo ni mutar archivos.

Un checkpoint apto para reanudar Aethel debe incluir pesos bajo ``model`` y
metadatos de configuración, paso y tokenizador. Los state_dict históricos
crudos se describen, pero se rechazan cuando se solicita reanudación segura.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _state_dict(payload: Any) -> tuple[dict[str, torch.Tensor], str]:
    if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
        state = payload["model"]
        origin = "packaged"
    elif isinstance(payload, dict) and all(isinstance(value, torch.Tensor) for value in payload.values()):
        state = payload
        origin = "raw_state_dict"
    else:
        raise ValueError("El archivo no contiene un state_dict Aethel reconocible.")
    if not state or not all(isinstance(value, torch.Tensor) for value in state.values()):
        raise ValueError("El state_dict contiene valores no tensoriales o está vacío.")
    return state, origin


def inspect_checkpoint(path: Path, require_reproducible: bool = False) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state, origin = _state_dict(payload)
    required = ("config", "step", "tokenizer")
    missing_metadata = [field for field in required if not isinstance(payload, dict) or field not in payload]
    faithful_required = ("optimizer", "scaler", "rng_state", "runtime_state", "resume_contract")
    faithful_resume_missing = [field for field in faithful_required if not isinstance(payload, dict) or field not in payload]
    shapes = {key: list(value.shape) for key, value in state.items()}
    report: dict[str, Any] = {
        "path": str(path.resolve()),
        "origin": origin,
        "tensor_count": len(state),
        "parameter_count": int(sum(value.numel() for value in state.values())),
        "missing_metadata": missing_metadata,
        "sample_shapes": dict(list(shapes.items())[:20]),
        "faithful_resume_missing": faithful_resume_missing,
        "reproducible_resume": origin == "packaged" and not missing_metadata and not faithful_resume_missing,
    }
    if isinstance(payload, dict) and isinstance(payload.get("config"), dict):
        report["config"] = payload["config"]
    if isinstance(payload, dict) and "step" in payload:
        report["step"] = int(payload["step"])
    if require_reproducible and not report["reproducible_resume"]:
        details = ", ".join(missing_metadata + faithful_resume_missing) if missing_metadata or faithful_resume_missing else "payload sin empaquetar"
        raise ValueError(
            "No se permite reanudar fielmente: el checkpoint necesita pesos, metadatos, optimizador, scaler, RNG, estado runtime y contrato verificables "
            f"({details}). Inspección disponible, carga bloqueada."
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--require-reproducible", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = inspect_checkpoint(args.checkpoint, args.require_reproducible)
    except (OSError, RuntimeError, ValueError) as error:
        print(json.dumps({"status": "rejected", "checkpoint": str(args.checkpoint), "reason": str(error)}, ensure_ascii=False))
        raise SystemExit(2) from error
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ready", **report}, ensure_ascii=False))


if __name__ == "__main__":
    main()
