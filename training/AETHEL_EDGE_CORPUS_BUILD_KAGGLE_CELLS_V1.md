# Aethel Edge V1 — construcción de corpus, tres celdas separadas

> **Esta guía prepara sólo el corpus Edge. No entrena, no selecciona GPU, no carga checkpoints y no evalúa holdout.** Mantiene el entrenamiento y la construcción de datos como autorizaciones independientes.

Use el mismo cuaderno Kaggle de construcción, tras actualizar el input de código y reiniciar la sesión, o un cuaderno separado llamado `Aethel — Construcción de Corpus Edge V1 — Reintento OpenR1`. Adjunte únicamente dos entradas: el bundle de código `aethel-direct-train-source-v1` que contiene la revisión `edge-corpus-build-openr1-aligned-flags-v1`, y el dataset base `aethel-nextgen-data-v1`. La segunda entrada aporta únicamente el tokenizador BPE existente; conservarlo hace posible comparar o reanudar una fase posterior sin cambiar el vocabulario. El resultado deberá guardarse como un **dataset privado nuevo** denominado `aethel-edge-corpus-v1`; no reemplace `aethel-nextgen-data-v1` ni las salidas fallidas anteriores.

| Celda | Operación | Red / GPU / pesos |
|---|---|---|
| **CELDA 1** | Resuelve el código y el tokenizador, sin leer corpus. | No / No / No |
| **CELDA 2** | Confirma que sólo están activas las cuatro fuentes autorizadas. | No / No / No |
| **CELDA 3** | Construye el corpus con la autorización ya registrada. | Sí / No / No |

## CELDA 1 — localizar las dos entradas sin construir ni descargar

```python
# CELDA 1 — Aethel Edge: resolver bundle y tokenizador base, sin red ni GPU
from pathlib import Path
import json

INPUT_ROOT = Path("/kaggle/input")
EXPECTED_REVISION = "edge-corpus-build-openr1-aligned-flags-v1"

release_markers = list(INPUT_ROOT.rglob("aethel_direct_train_source_release.json"))
matches = []
for marker in release_markers:
    try:
        release = json.loads(marker.read_text(encoding="utf-8"))
    except Exception:
        continue
    if release.get("revision") == EXPECTED_REVISION and (marker.parent.parent / "engine" / "prepare_bilingual_corpus.py").is_file():
        matches.append(marker.parent.parent)
if len(matches) != 1:
    rendered = "\n".join(f"- {path}" for path in matches) or "- ninguno"
    raise RuntimeError(f"Se esperaba un único bundle Edge '{EXPECTED_REVISION}'. Candidatos:\n{rendered}")

SOURCE_ROOT = matches[0]
tokenizers = [path.parent for path in INPUT_ROOT.rglob("tokenizer.json") if "aethel-nextgen-data-v1" in str(path)]
if len(tokenizers) != 1:
    rendered = "\n".join(f"- {path}" for path in tokenizers) or "- ninguno"
    raise RuntimeError(f"Se esperaba un único tokenizer base de aethel-nextgen-data-v1. Candidatos:\n{rendered}")
BASE_DATA_ROOT = tokenizers[0]

print("CELDA 1 — entradas Edge resueltas; no se inició red, GPU, entrenamiento ni carga de pesos.")
print(f"SOURCE_ROOT: {SOURCE_ROOT}")
print(f"BASE_DATA_ROOT: {BASE_DATA_ROOT}")
```

## CELDA 2 — confirmar exactamente las fuentes autorizadas

```python
# CELDA 2 — Aethel Edge: validar fuentes autorizadas, sin red ni lectura de shards
manifest_path = SOURCE_ROOT / "training" / "aethel_edge_v1.manifest.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("approval_required") is not True:
    raise RuntimeError("El manifiesto debe requerir aprobación explícita.")

EXPECTED_SOURCES = {
    "fineweb-en-sample-10bt-proposed",
    "fineweb2-es-proposed",
    "hplt2-es-proposed",
    "openr1-math-proposed",
}
sources = manifest.get("sources", [])
if {source.get("id") for source in sources} != EXPECTED_SOURCES:
    raise RuntimeError("El manifiesto debe contener únicamente las cuatro fuentes Edge autorizadas.")
if any(source.get("approved") is not True or source.get("enabled") is not True for source in sources):
    raise RuntimeError("Toda fuente Edge debe estar explícitamente aprobada y habilitada antes de usar la red.")

for source in sources:
    print({
        "id": source.get("id"),
        "license": source.get("license"),
        "approved": source.get("approved"),
        "enabled": source.get("enabled"),
        "document_limit": source.get("document_limit"),
    })

print("CELDA 2 OK — fuentes autorizadas verificadas; no hubo red, GPU, entrenamiento ni carga de pesos.")
```

## CELDA 3 — construir el corpus tras autorización explícita de fuentes

> La autorización explícita de FineWeb sample-10BT EN, FineWeb2 ES, HPLT ES y OpenR1-Math para este reintento ya quedó registrada en el bundle `edge-corpus-build-openr1-aligned-flags-v1`. La CELDA 3 descarga datos de red. Antes de crear una salida o abrir shards, valida la capacidad por idioma y las configuraciones Hugging Face usando sólo el dataset y su revisión; si detecta un plan inviable, otra configuración o una fuente no aprobada, se detiene correctamente.

```python
# CELDA 3 — Aethel Edge: construcción con red autorizada; no entrena ni usa GPU
import os
import subprocess

EDGE_DATA_OUTPUT = Path("/kaggle/working/aethel-edge-corpus-v1-retry-openr1-aligned-flags")
if EDGE_DATA_OUTPUT.exists():
    raise RuntimeError(f"La salida debe ser inédita y no se sobrescribe: {EDGE_DATA_OUTPUT}")

active_sources = [source for source in manifest["sources"] if source.get("enabled")]
if not active_sources or any(source.get("approved") is not True for source in active_sources):
    raise RuntimeError("No existe una aprobación manifestada para construir el corpus; no se inicia la red.")

environment = dict(os.environ)
environment.update({
    "SOURCE_ROOT": str(SOURCE_ROOT),
    "BASE_DATA_ROOT": str(BASE_DATA_ROOT),
    "EDGE_DATA_OUTPUT": str(EDGE_DATA_OUTPUT),
})
subprocess.run(["bash", str(SOURCE_ROOT / "training" / "run_kaggle_build_edge_corpus_v1.sh")], check=True, env=environment)

for required in ("prepared_manifest.json", "validation.jsonl.gz", "tokenizer.json"):
    if not (EDGE_DATA_OUTPUT / required).is_file():
        raise RuntimeError(f"Falta el artefacto requerido: {required}")
if not list(EDGE_DATA_OUTPUT.glob("train-*.jsonl.gz")):
    raise RuntimeError("No se generaron shards de entrenamiento.")

print("AETHEL_EDGE_CORPUS_BUILD_READY")
print(f"OUTPUT: {EDGE_DATA_OUTPUT}")
print("Acción manual obligatoria: use Save Version de Kaggle inmediatamente y cree el dataset privado aethel-edge-corpus-v1 antes de cerrar la sesión.")
```

La salida preparada conserva hashes de fuentes, filtros, semilla, shards y una partición de validación estable. **No constituye una evaluación**, no habilita fuentes de evaluación reservadas y no prueba bilingüismo, razonamiento ni matemáticas del modelo.
