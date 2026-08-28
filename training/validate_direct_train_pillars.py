#!/usr/bin/env python3
"""Valida la evidencia emitida por una corrida directa de Aethel.

No carga los pesos, no abre el corpus ni evalúa holdout. Lee sólo los artefactos
de salida del entrenamiento para distinguir una corrida completa de una
afirmación de capacidad. La salud o calidad no se fabrica: se reporta medida.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import fmean
from typing import Any


REQUIRED_PILLARS = (
    "La Roca",
    "El Líquido",
    "Ciclo de Sueño",
    "Neuromodulación",
    "Espacio de Trabajo Global",
)


def load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Métrica JSONL inválida en línea {line_number}: {error}") from error
        if not isinstance(event, dict):
            raise ValueError(f"La línea {line_number} no contiene un evento JSON.")
        events.append(event)
    if not events:
        raise ValueError("No hay eventos de entrenamiento para validar.")
    return events


def require_finite(event: dict[str, Any], key: str) -> float:
    value = event.get(key)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"Métrica no finita o ausente: {key}")
    return float(value)


def validate(events: list[dict[str, Any]], output: Path, expected_steps: int) -> dict[str, Any]:
    final = events[-1]
    if final.get("step") != expected_steps:
        raise ValueError(f"El último paso es {final.get('step')}; se esperaban {expected_steps}.")
    if not (output / "latest.pt").is_file() or not (output / "recovery_receipt.json").is_file():
        raise ValueError("Falta latest.pt o recovery_receipt.json; la corrida no es recuperable.")

    losses = [require_finite(event, "loss") for event in events]
    throughputs = [require_finite(event, "tokens_per_second") for event in events]
    runtime = final.get("runtime")
    memory = final.get("memory")
    if not isinstance(runtime, dict) or not isinstance(memory, dict):
        raise ValueError("Falta telemetría runtime o manifiesto de memoria en el último evento.")

    pillars = runtime.get("pillars")
    if not isinstance(pillars, dict) or any(pillars.get(name) is not True for name in REQUIRED_PILLARS):
        raise ValueError("La telemetría no confirma todos los pilares obligatorios.")
    liquid = memory.get("liquid")
    replay = memory.get("replay")
    if not isinstance(liquid, dict) or int(liquid.get("version", 0)) < 1:
        raise ValueError("El Líquido no produjo una versión persistida.")
    if not isinstance(replay, dict) or int(replay.get("replay_records", 0)) < 1:
        raise ValueError("El Ciclo de Sueño no produjo registros de replay.")
    if int(memory.get("episodic_records", 0)) < 1 or int(memory.get("semantic", {}).get("semantic_records", 0)) < 1:
        raise ValueError("La memoria episódica o semántica no produjo evidencia persistida.")
    if not isinstance(runtime.get("neuromodulation"), (int, float)) or not isinstance(runtime.get("surprise"), (int, float)):
        raise ValueError("Falta evidencia numérica de neuromodulación.")

    routing = final.get("routing")
    health_events = [event.get("router_health") for event in events]
    if not isinstance(routing, list) or not routing or not all(isinstance(item, dict) for item in health_events):
        raise ValueError("Falta evidencia de routing MoE o salud del router.")
    config = final.get("config")
    if not isinstance(config, dict):
        raise ValueError("Falta configuración del núcleo para validar eficiencia estructural.")
    experts = int(config.get("experts", 0))
    active_experts = int(config.get("active_experts", 0))
    heads = int(config.get("heads", 0))
    kv_heads = int(config.get("kv_heads", 0))
    if not (0 < active_experts < experts and 0 < kv_heads < heads):
        raise ValueError("La configuración no acredita MoE disperso y GQA para la eficiencia estructural.")

    healthy_steps = sum(bool(item.get("healthy")) for item in health_events)
    max_imbalance = max(float(item.get("max_imbalance", float("inf"))) for item in health_events)
    min_entropy = min(float(item.get("min_entropy", float("-inf"))) for item in health_events)
    report = {
        "schema": "aethel-direct-train-validation/v2",
        "status": "AETHEL_DIRECT_TRAIN_ARTIFACTS_VALIDATED",
        "quality_status": "MEASURED_NOT_PROMOTED",
        "validation_scope": "artifact_and_telemetry_only",
        "steps": {"count": len(events), "final": final["step"], "expected": expected_steps},
        "checkpoint": {"path": str((output / "latest.pt").resolve()), "recoverable": True},
        "pillars": {
            "La Roca": {"telemetry": True, "replay_loss_events": sum(event.get("replay_loss") is not None for event in events)},
            "El Líquido": {"telemetry": True, "version": int(liquid["version"])},
            "Ciclo de Sueño": {"telemetry": True, "replay_records": int(replay["replay_records"])},
            "Memoria": {"episodic_records": int(memory["episodic_records"]), "semantic_records": int(memory["semantic"]["semantic_records"])},
            "Neuromodulación": {"telemetry": True, "final_value": float(runtime["neuromodulation"]), "final_surprise": float(runtime["surprise"])},
            "Espacio de Trabajo Global": {"telemetry": True},
            "MoE": {"experts": experts, "active_experts": active_experts, "healthy_steps": healthy_steps, "unhealthy_steps": len(events) - healthy_steps, "min_entropy": min_entropy, "max_imbalance": max_imbalance},
        },
        "efficiency": {
            "sparse_activation_ratio": active_experts / experts,
            "gqa_kv_ratio": kv_heads / heads,
            "parameters_trainable": int(final.get("parameters_trainable", 0)),
            "tokens_per_second_mean": fmean(throughputs),
            "tokens_per_second_final": throughputs[-1],
            "loss_initial": losses[0],
            "loss_final": losses[-1],
            "baseline_comparison": "PENDING_SEPARATE_BASELINE_RUN",
        },
        "limits": {
            "checkpoint_loaded": False,
            "corpus_read": False,
            "holdout_read": False,
            "quality_promoted": False,
            "not_proven": [
                "razonamiento", "bilinguismo_nativo", "eficiencia_relativa_a_baseline",
                "inmutabilidad_y_rollback_de_la_roca", "promocion_de_adaptador_lora",
                "consolidacion_sueno_completa", "workspace_competitivo", "runtime_rust_desplegado",
            ],
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--expected-steps", type=int, default=768)
    args = parser.parse_args()
    report = validate(load_events(args.metrics), args.output, args.expected_steps)
    destination = args.output / "aethel_direct_validation.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))
    print("AETHEL_DIRECT_TRAIN_ARTIFACTS_VALIDATED")


if __name__ == "__main__":
    main()
