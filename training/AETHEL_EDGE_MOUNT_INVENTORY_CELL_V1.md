# CELDA 1 — Inventario seguro del montaje de `aethel-edge-corpus-v1`

Sustituya temporalmente la CELDA 1 que falló por la siguiente única celda. Esta inspección no usa GPU, no abre archivos JSONL o gzip, no crea salidas y no entrena; sólo enumera rutas y tamaños que Kaggle expone en el input ya montado.

```python
# CELDA 1 — Inventario seguro del montaje Edge: rutas y tamaños, sin GPU ni entrenamiento
from pathlib import Path
import json

INPUT_ROOT = Path("/kaggle/input")
EXPECTED_DATASET = "aethel-edge-corpus-v1"

manifest_paths = sorted(
    path
    for path in INPUT_ROOT.rglob("prepared_manifest.json")
    if EXPECTED_DATASET in str(path)
)
if len(manifest_paths) != 1:
    rendered = "\n".join(f"- {path}" for path in manifest_paths) or "- ninguno"
    raise RuntimeError(
        "Se esperaba exactamente un prepared_manifest.json dentro del input Edge. "
        f"Candidatos:\n{rendered}\nNo se usó GPU ni se leyó contenido de corpus."
    )

manifest_path = manifest_paths[0]
dataset_root = manifest_path.parent
inventory = []
for path in sorted(dataset_root.rglob("*")):
    if path.is_file():
        inventory.append(
            {
                "path": str(path.relative_to(dataset_root)),
                "bytes": path.stat().st_size,
            }
        )

print("CELDA 1 — INVENTARIO_EDGE_MONTAJE_SEGURO")
print(f"DATASET_ROOT: {dataset_root}")
print(f"FILE_COUNT: {len(inventory)}")
print(json.dumps(inventory, ensure_ascii=False, indent=2))
print("INVENTARIO_FINALIZADO — no se usó GPU, no se leyeron ejemplos y no se entrenó.")
```

Copie y pegue el resultado completo de esta celda. No ejecute CELDA 2 ni CELDA 3 hasta que se interprete el inventario.
