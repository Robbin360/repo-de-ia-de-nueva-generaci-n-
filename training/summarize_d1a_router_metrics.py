"""Resume métricas D1A/D1B/D1C sin cargar pesos, corpus ni holdout."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("No hay valores numéricos para resumir.")
    return {"count": len(values), "minimum": min(values), "mean": sum(values) / len(values), "maximum": max(values)}


def load_events(metrics_path: Path) -> list[dict[str, Any]]:
    if metrics_path.name != "metrics_rank_0.jsonl":
        raise ValueError("D1A sólo acepta un archivo llamado metrics_rank_0.jsonl; no se aceptan checkpoints.")
    events: list[dict[str, Any]] = []
    previous_step = 0
    with metrics_path.open("rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"JSONL de métricas inválido en línea {line_number}: {error}") from error
            if not isinstance(event, dict) or not isinstance(event.get("step"), int) or event["step"] <= previous_step:
                raise ValueError(f"Paso inválido o no creciente en línea {line_number}.")
            if not isinstance(event.get("loss"), (float, int)) or not isinstance(event.get("routing"), list):
                raise ValueError(f"Evento sin loss/routing válido en línea {line_number}.")
            previous_step = event["step"]
            events.append(event)
    if not events:
        raise ValueError("metrics_rank_0.jsonl no contiene eventos.")
    return events


def summarize(events: list[dict[str, Any]], diagnostic_id: str = "D1A") -> dict[str, Any]:
    allowed_ids = {
        "D1A",
        "D1B",
        "D1C",
        "D1D",
        "D1E",
        "DIRECT_TRAIN_V1",
        "DIRECT_TRAIN_ROUTER_FIX_V1",
        "DIRECT_TRAIN_ROUTER_JITTER_V1",
        "AETHEL_EDGE_LONG_PHASE_V1",
    }
    if diagnostic_id not in allowed_ids:
        raise ValueError("diagnostic_id no está reconocido por el resumen de router.")
    layer_count = len(events[0]["routing"])
    if layer_count < 1:
        raise ValueError("No hay capas de routing en el primer evento.")
    layers: list[dict[str, list[float]]] = [
        {"entropy": [], "max_load": [], "imbalance": [], "bias": []} for _ in range(layer_count)
    ]
    healthy = 0
    unhealthy = 0
    healthy_micro_steps = 0
    unhealthy_micro_steps = 0
    losses: list[float] = []
    tokens: list[int] = []
    for event in events:
        routing = event["routing"]
        if not isinstance(routing, list) or len(routing) != layer_count:
            raise ValueError("El número de capas de routing cambia entre eventos.")
        losses.append(float(event["loss"]))
        if isinstance(event.get("tokens_seen"), int):
            tokens.append(event["tokens_seen"])
        health = event.get("router_health")
        if not isinstance(health, dict) or not isinstance(health.get("healthy"), bool):
            raise ValueError("Evento sin router_health.healthy booleano.")
        healthy += int(health["healthy"])
        unhealthy += int(not health["healthy"])
        window = event.get("telemetry_window")
        window_health = window.get("router_health") if isinstance(window, dict) else None
        if isinstance(window_health, dict):
            healthy_micro_steps += int(window_health.get("healthy_micro_steps", 0))
            unhealthy_micro_steps += int(window_health.get("unhealthy_micro_steps", 0))
        else:
            healthy_micro_steps += int(health["healthy"])
            unhealthy_micro_steps += int(not health["healthy"])
        for index, details in enumerate(routing):
            if not isinstance(details, dict):
                raise ValueError(f"Routing inválido para capa {index}.")
            for name in ("entropy", "max_load", "imbalance"):
                value = details.get(name)
                if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
                    raise ValueError(f"{name} inválido para capa {index}.")
                layers[index][name].append(float(value))
            bias = details.get("bias")
            if not isinstance(bias, list) or not bias or not all(isinstance(value, (float, int)) for value in bias):
                raise ValueError(f"bias inválido para capa {index}.")
            layers[index]["bias"].extend(float(value) for value in bias)

    layer_summary = []
    for index, values in enumerate(layers):
        bias = values["bias"]
        layer_summary.append(
            {
                "layer": index,
                "entropy": numeric_summary(values["entropy"]),
                "max_load": numeric_summary(values["max_load"]),
                "imbalance": numeric_summary(values["imbalance"]),
                "bias": {
                    "count": len(bias),
                    "minimum": min(bias),
                    "maximum": max(bias),
                    "max_abs": max(abs(value) for value in bias),
                },
            }
        )
    return {
        "schema": f"aethel-{diagnostic_id.lower()}-router-diagnostic/v1",
        "status": f"{diagnostic_id}_METRICS_SUMMARIZED",
        "diagnostic_id": diagnostic_id,
        "steps": {"first": events[0]["step"], "last": events[-1]["step"], "telemetry_events": len(events)},
        "loss": numeric_summary(losses),
        "tokens_seen_final": tokens[-1] if tokens else None,
        "router_health": {
            "healthy_telemetry_events": healthy,
            "unhealthy_telemetry_events": unhealthy,
            "healthy_micro_steps": healthy_micro_steps,
            "unhealthy_micro_steps": unhealthy_micro_steps,
        },
        "layers": layer_summary,
        "config": events[0].get("config"),
        "limits": {
            "checkpoint_loaded": False,
            "raw_corpus_read": False,
            "holdout_content_read": False,
            "network_requests": 0,
            "promotion_authorized": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--diagnostic-id",
        choices=(
            "D1A",
            "D1B",
            "D1C",
            "D1D",
            "D1E",
            "DIRECT_TRAIN_V1",
            "DIRECT_TRAIN_ROUTER_FIX_V1",
            "DIRECT_TRAIN_ROUTER_JITTER_V1",
            "AETHEL_EDGE_LONG_PHASE_V1",
        ),
        default="D1A",
    )
    args = parser.parse_args()
    report = summarize(load_events(args.metrics), diagnostic_id=args.diagnostic_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
