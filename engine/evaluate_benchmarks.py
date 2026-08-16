"""Agrega resultados reales de MMLU, GSM8K y HumanEval sin inventar puntuaciones.

El generador y el ejecutor de código son responsabilidades separadas: este script
solo compara predicciones almacenadas con respuestas de referencia. Para HumanEval,
consume resultados de un sandbox aislado; nunca ejecuta código generado dentro del
proceso de entrenamiento.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


FINAL_NUMBER = re.compile(r"####\s*(-?\d+(?:\.\d+)?)|(-?\d+(?:\.\d+)?)\s*$")


def normalize_mmlu(value: object) -> str:
    return str(value).strip().upper()[:1]


def normalize_gsm8k(value: object) -> str:
    match = FINAL_NUMBER.search(str(value).replace(",", ""))
    return (match.group(1) or match.group(2)) if match else str(value).strip()


def load_jsonl(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        record_id = str(record.get("id", ""))
        if not record_id:
            raise ValueError(f"{path}:{line_number}: falta id")
        if record_id in records:
            raise ValueError(f"{path}:{line_number}: id duplicado {record_id}")
        records[record_id] = record
    return records


def evaluate(reference: dict[str, dict], predictions: dict[str, dict]) -> dict:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    missing: dict[str, int] = defaultdict(int)
    for record_id, item in reference.items():
        task = str(item["task"]).lower()
        totals[task] += 1
        prediction = predictions.get(record_id)
        if prediction is None:
            missing[task] += 1
            continue
        if task == "mmlu":
            is_correct = normalize_mmlu(prediction.get("answer")) == normalize_mmlu(item.get("answer"))
        elif task == "gsm8k":
            is_correct = normalize_gsm8k(prediction.get("answer")) == normalize_gsm8k(item.get("answer"))
        elif task == "humaneval":
            # El campo pass solo puede proceder de un ejecutor aislado del benchmark.
            is_correct = prediction.get("pass") is True
        else:
            raise ValueError(f"Tarea no admitida: {task}")
        correct[task] += int(is_correct)
    return {
        "protocol": "aethel-benchmark-v1",
        "scores": {
            task: {
                "correct": correct[task],
                "total": total,
                "missing_predictions": missing[task],
                "accuracy": correct[task] / total if total else None,
            }
            for task, total in sorted(totals.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True, help="JSONL con task, id y respuesta oficial")
    parser.add_argument("--predictions", required=True, help="JSONL con predicciones reales de Aethel o un sandbox")
    parser.add_argument("--output", required=True, help="Ruta del informe JSON")
    args = parser.parse_args()
    report = evaluate(load_jsonl(Path(args.reference)), load_jsonl(Path(args.predictions)))
    report["reference"] = str(Path(args.reference).resolve())
    report["predictions"] = str(Path(args.predictions).resolve())
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
