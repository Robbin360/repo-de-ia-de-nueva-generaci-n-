# CELDA 8 — Verificar el release D1C V2 de corrección de resumen en modo bloqueado
# Estado: D1C V2 preparado localmente; esta celda no está autorizada para ejecución.
# Alcance: localiza sólo el marcador de release. No selecciona GPU, no copia código,
# no lee Dataset train/holdout, no carga pesos, no toca outputs/checkpoints y no entrena.

from __future__ import annotations

import json
from pathlib import Path


SOURCE_DATASET = Path("/kaggle/input/datasets/felixtremigual/aethel-nextgen-source-e0-v1")
REQUIRED_RELEASE = "d1c-v2-summary-cli-fix-train-only"


def resolve_release(source_dataset: Path) -> tuple[Path, dict[str, object]]:
    matches: list[tuple[Path, dict[str, object]]] = []
    for marker_path in source_dataset.rglob("aethel_kaggle_source_release.json"):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker.get("release") == REQUIRED_RELEASE:
            matches.append((marker_path.parent, marker))
    if len(matches) != 1:
        rendered = "\n".join(f"- {root}" for root, _ in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente un release '{REQUIRED_RELEASE}'. Candidatos:\n{rendered}\n"
            "No se inició GPU ni diagnóstico."
        )
    root, marker = matches[0]
    if marker.get("d1c_v2_execution_authorized") is not False or marker.get("training_authorized") is not False:
        raise RuntimeError("El marcador D1C V2 debe mantenerse no autorizado para esta CELDA 8 bloqueada.")
    return root, marker


source_root, source_marker = resolve_release(SOURCE_DATASET)
print("CELDA 8 — Verificar el release D1C V2 de corrección de resumen en modo bloqueado")
print(f"SOURCE_INPUT_D1C_V2: {source_root}")
print(f"SOURCE_RELEASE: {source_marker['release']}")
print("D1C_V2_CELL_PREPARED_NOT_EXECUTED")
print(
    "D1C V2 permanece bloqueada: no se seleccionó GPU, no se copió código, no se leyó "
    "Dataset train/holdout, no se cargaron pesos y no se entrenó."
)
print("D1C_V2_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT")
