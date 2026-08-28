# D1B — celda preparatoria bloqueada del diagnóstico de sesgo del router MoE.
# Añadir como celda nueva al notebook. No habilita ni contiene una rama de ejecución.

from __future__ import annotations

import json
from pathlib import Path


# Deben permanecer así. Esta celda sólo localiza el release de código D1B y
# declara sus límites; no selecciona GPU, no copia código y no abre datos.
D1B_EXECUTION_ENABLED = False
EXPECTED_RELEASE = "d1b-v1-router-bias-step-001-train-only"
SOURCE_DATASET = "aethel-nextgen-source-e0-v1"
INPUT_ROOT = Path("/kaggle/input")


def resolve_d1b_source() -> Path:
    matches: list[Path] = []
    for marker in sorted(INPUT_ROOT.rglob("aethel_kaggle_source_release.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        source_root = marker.parent.parent
        if (
            SOURCE_DATASET in str(marker)
            and payload.get("release") == EXPECTED_RELEASE
            and (source_root / "training" / "run_kaggle_d1b_router_bias_diagnostic.sh").is_file()
            and (source_root / "training" / "validate_aethel_train_only_mount.py").is_file()
            and (source_root / "training" / "summarize_d1a_router_metrics.py").is_file()
        ):
            matches.append(source_root)

    if len(matches) != 1:
        rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
        raise RuntimeError(
            f"Se esperaba exactamente un release D1B '{EXPECTED_RELEASE}'. Candidatos:\n"
            f"{rendered}\nNo se seleccionó GPU, no se leyó Dataset y no se entrenó."
        )
    return matches[0]


source_input = resolve_d1b_source()
print(f"SOURCE_INPUT_D1B: {source_input}")
print(f"SOURCE_RELEASE: {EXPECTED_RELEASE}")

if D1B_EXECUTION_ENABLED:
    raise RuntimeError(
        "D1B sigue bloqueada: esta celda no contiene una rama de ejecución. "
        "Una futura celda habilitable requiere autorizaciones B5/B6 separadas."
    )

print("D1B_CELL_PREPARED_NOT_EXECUTED")
print(
    "D1B permanece bloqueada: no se seleccionó GPU, no se copió código, "
    "no se leyó Dataset train/holdout, no se cargaron pesos y no se entrenó."
)
print("D1B_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT")
