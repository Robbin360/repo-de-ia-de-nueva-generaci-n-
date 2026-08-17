"""Valida controles de datos y evaluación antes de una corrida GPU; no descarga datasets."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


REQUIRED_FILTERS = {"remove_simple_pii": True, "deduplicate_exact": True, "deduplicate_near": "required-at-scale", "exclude_evaluation_sets": True}
IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{7,64}$", re.IGNORECASE)


def validate(manifest_path: Path, curriculum_path: Path, benchmark_path: Path, evaluation_config_path: Path | None = None, require_approved: bool = False) -> dict:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    warnings: list[str] = []
    if manifest.get("approval_required") is not True:
        errors.append("approval_required debe ser true")
    sources = manifest.get("sources", [])
    if not sources:
        errors.append("el manifiesto no contiene fuentes")
    for source in sources:
        for key in ("id", "dataset", "split", "text_column", "revision", "license_review", "provenance_url", "enabled"):
            if key not in source:
                errors.append(f"fuente {source.get('id', '<sin id>')} carece de {key}")
        if source.get("enabled") is True:
            warnings.append(f"fuente {source.get('id')} habilitada: requiere aprobación humana antes de descargar")
        revision = str(source.get("revision", ""))
        if not IMMUTABLE_REVISION.fullmatch(revision):
            message = f"fuente {source.get('id')} sin revisión inmutable fijada"
            (errors if require_approved else warnings).append(message)
        if require_approved and source.get("enabled") is not True:
            errors.append(f"fuente {source.get('id')} no está habilitada en un manifiesto de lanzamiento")
        if require_approved and source.get("approved") is not True:
            errors.append(f"fuente {source.get('id')} no tiene aprobación explícita")
    filters = manifest.get("filters", {})
    for key, expected in REQUIRED_FILTERS.items():
        actual = filters.get(key)
        if key == "deduplicate_near":
            if not isinstance(actual, str) or not actual.startswith(expected):
                errors.append("deduplicate_near debe requerir un job revisado a escala")
        elif actual is not expected:
            errors.append(f"filtro obligatorio ausente o inválido: {key}")
    for path, phrase in ((curriculum_path, "Condición para avanzar"), (benchmark_path, "No se ejecuta código generado")):
        if not path.exists():
            errors.append(f"documento requerido ausente: {path}")
        elif phrase not in path.read_text(encoding="utf-8"):
            errors.append(f"documento requerido no contiene control esperado: {path.name}")
    if require_approved:
        if evaluation_config_path is None or not evaluation_config_path.exists():
            errors.append("falta una configuración de evaluación aprobada")
        else:
            evaluation = json.loads(evaluation_config_path.read_text(encoding="utf-8"))
            for key in ("approved", "holdout_path", "tokenizer_path", "seed", "benchmark_references"):
                if key not in evaluation:
                    errors.append(f"configuración de evaluación carece de {key}")
            if evaluation.get("approved") is not True:
                errors.append("la configuración de evaluación no está aprobada")
            if not str(evaluation.get("holdout_path", "")).strip() or not str(evaluation.get("tokenizer_path", "")).strip():
                errors.append("la configuración de evaluación debe fijar holdout y tokenizador")
            references = evaluation.get("benchmark_references")
            if not isinstance(references, dict) or not references:
                errors.append("la configuración de evaluación debe incluir referencias de benchmark")
            expected_paths = {"holdout_path": evaluation.get("holdout_path"), "tokenizer_path": evaluation.get("tokenizer_path")}
            if isinstance(references, dict):
                expected_paths.update({f"benchmark:{name}": value for name, value in references.items()})
            for label, value in expected_paths.items():
                path = Path(str(value))
                if not str(value).strip() or not path.is_file():
                    errors.append(f"ruta de evaluación inaccesible: {label}")
    return {
        "status": "READY_FOR_HUMAN_APPROVAL" if not errors else "BLOCKED",
        "manifest": str(manifest_path),
        "sources": len(sources),
        "require_approved": require_approved,
        "evaluation_config": str(evaluation_config_path) if evaluation_config_path else None,
        "errors": errors,
        "warnings": warnings,
        "statement": "Este validador no aprueba licencias, no habilita fuentes y no inicia una descarga o una corrida.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="training/corpus_manifest.example.json")
    parser.add_argument("--curriculum", default="TRAINING_CURRICULUM.md")
    parser.add_argument("--benchmarks", default="training/BENCHMARK_PROTOCOL.md")
    parser.add_argument("--evaluation-config")
    parser.add_argument("--require-approved", action="store_true", help="Bloquea fuentes sin revisión, aprobación y evaluación explícita.")
    parser.add_argument("--output")
    args = parser.parse_args()
    evaluation = Path(args.evaluation_config) if args.evaluation_config else None
    report = validate(Path(args.manifest), Path(args.curriculum), Path(args.benchmarks), evaluation, args.require_approved)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    if report["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
