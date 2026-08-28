# Aethel Edge V1 — primera sesión larga, tres celdas

> **Esta guía prepara y, sólo después de una autorización explícita final, inicia una primera sesión de entrenamiento Edge. No reanuda ni carga pesos existentes, no evalúa el holdout, no promociona un checkpoint y no demuestra bilingüismo, razonamiento, matemáticas ni eficiencia.**

Use el cuaderno existente **`Aethel — Entrenamiento Directo Dataset V1`**. Debe tener exactamente **dos inputs**: el bundle de código privado `aethel-direct-train-source-v1`, actualizado a la revisión `edge-long-session-phase1-kaggle-jsonl-v1`, y el corpus privado `aethel-edge-corpus-v1`. Quite `aethel-nextgen-data-v1`; no añada un tercer input. Reinicie la sesión antes de ejecutar las celdas para asegurar que Kaggle monta sólo esas dos entradas.

| Celda | Operación | GPU / datos / pesos |
|---|---|---|
| **CELDA 1** | Resuelve entradas, verifica manifiesto, estructura y hashes. | No / sólo metadatos y hashes / No |
| **CELDA 2** | Comprueba GPU disponible, perfil de entrenamiento y consistencia del corpus. | Sí, sólo inspección / No lee JSONL / No |
| **CELDA 3** | Primera sesión larga desde cero y empaquetado de preservación. | Sí / shards de entrenamiento / pesos nuevos |

## CELDA 1 — resolver y verificar las dos entradas, sin GPU ni entrenamiento

```python
# CELDA 1 — Aethel Edge: resolver exactamente dos inputs y validar metadatos
from pathlib import Path
import hashlib
import json

INPUT_ROOT = Path("/kaggle/input")
WORK_ROOT = Path("/kaggle/working")
EXPECTED_REVISION = "edge-long-session-phase1-kaggle-jsonl-v1"
EXPECTED_DATASET = "aethel-edge-corpus-v1"
EXPECTED_SHARDS = 10
OUTPUT_ROOT = WORK_ROOT / "aethel-edge-phase-1-183680-v1"
PRESERVATION_PACKAGE = WORK_ROOT / "aethel-edge-phase-1-183680-v1-preservation.tar.gz"

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

source_matches = []
for marker_path in INPUT_ROOT.rglob("aethel_direct_train_source_release.json"):
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:
        continue
    candidate = marker_path.parent.parent
    if (
        marker.get("release") == "aethel-direct-train-source-v1"
        and marker.get("revision") == EXPECTED_REVISION
        and marker.get("dataset_required") == EXPECTED_DATASET
        and (candidate / "training" / "run_kaggle_edge_long_session_v1.sh").is_file()
    ):
        source_matches.append((candidate, marker))
if len(source_matches) != 1:
    rendered = "\n".join(f"- {path}" for path, _ in source_matches) or "- ninguno"
    raise RuntimeError(f"Se esperaba un único bundle Edge '{EXPECTED_REVISION}'. Candidatos:\n{rendered}")
SOURCE_ROOT, RELEASE = source_matches[0]

prepared_matches = [path for path in INPUT_ROOT.rglob("prepared_manifest.json") if EXPECTED_DATASET in str(path)]
if len(prepared_matches) != 1:
    rendered = "\n".join(f"- {path}" for path in prepared_matches) or "- ninguno"
    raise RuntimeError(f"Se esperaba un único corpus privado '{EXPECTED_DATASET}'. Candidatos:\n{rendered}")
prepared_path = prepared_matches[0]
prepared = json.loads(prepared_path.read_text(encoding="utf-8"))

if prepared.get("schema_version") != 1:
    raise RuntimeError("El prepared_manifest.json no corresponde al esquema Edge esperado.")
language_counts = prepared.get("language_counts", {})
for language, minimum in {"en": 120000, "es": 120000}.items():
    counts = language_counts.get(language, {})
    total = int(counts.get("accepted", 0)) + int(counts.get("validation", 0))
    if total < minimum:
        raise RuntimeError(f"Corpus insuficiente para {language}: {total} < {minimum}.")

manifest_shards = prepared.get("shards", [])
manifest_names = {item.get("path") for item in manifest_shards}
compressed_shards = sorted(prepared_path.parent.rglob("train-*.jsonl.gz"))
plaintext_shards = sorted(prepared_path.parent.rglob("train-*.jsonl"))
if compressed_shards and plaintext_shards:
    raise RuntimeError("El montaje mezcla shards .jsonl.gz y .jsonl; no se iniciará entrenamiento.")
observed_shards = compressed_shards or plaintext_shards
observed_names = {path.name for path in observed_shards}
logical_observed_names = {
    name if name.endswith(".gz") else f"{name}.gz"
    for name in observed_names
}
parents = {path.parent for path in observed_shards}
representation = "gzip-original" if compressed_shards else "kaggle-descomprimido-jsonl"
layout = {
    "prepared_manifest": str(prepared_path),
    "representation": representation,
    "expected_shards": len(manifest_shards),
    "observed_shards": len(observed_shards),
    "observed_shard_roots": sorted(str(path) for path in parents),
    "missing_logical_names": sorted(name for name in manifest_names if name not in logical_observed_names),
    "unexpected_logical_names": sorted(name for name in logical_observed_names if name not in manifest_names),
}
print("EDGE_DATA_LAYOUT:", json.dumps(layout, ensure_ascii=False))
if len(manifest_shards) != EXPECTED_SHARDS or len(observed_shards) != EXPECTED_SHARDS or manifest_names != logical_observed_names or len(parents) != 1:
    raise RuntimeError("La estructura de shards no coincide con el manifiesto. Se imprimió EDGE_DATA_LAYOUT; no se iniciará entrenamiento.")
EDGE_DATA_ROOT = next(iter(parents))
EDGE_MANIFEST_PATH = prepared_path
EDGE_TOKENIZER_PATH = EDGE_DATA_ROOT / "tokenizer.json"
validation = prepared.get("validation", {})
validation_name = str(validation.get("path", ""))
validation_path = EDGE_DATA_ROOT / validation_name
if not validation_path.is_file() and validation_name.endswith(".gz"):
    validation_path = EDGE_DATA_ROOT / validation_name.removesuffix(".gz")
if not validation_path.is_file():
    raise RuntimeError("Falta el holdout de validación declarado; no se iniciará entrenamiento.")
if representation == "gzip-original":
    for item in manifest_shards:
        path = EDGE_DATA_ROOT / item["path"]
        if sha256_file(path) != item.get("sha256"):
            raise RuntimeError(f"Hash no coincide para el shard: {path.name}")
    if sha256_file(validation_path) != validation.get("sha256"):
        raise RuntimeError("No coincide el hash del holdout de validación; no se iniciará entrenamiento.")
else:
    print("GZIP_HASH_COMPARISON: no aplicable; Kaggle expone JSONL descomprimido. Se verificaron nombres lógicos, conteo, manifiesto y tokenizador; no se afirma verificación independiente del contenido.")
if not EDGE_TOKENIZER_PATH.is_file():
    raise RuntimeError("Falta tokenizer.json en el corpus Edge.")
if OUTPUT_ROOT.exists() or PRESERVATION_PACKAGE.exists():
    raise RuntimeError("La salida o el TAR previstos ya existen; no se sobrescribe ni se reanuda.")

print("CELDA 1 — inputs Edge y estructura verificados; no se usó GPU, no se leyó JSONL y no se entrenó.")
print(f"SOURCE_ROOT: {SOURCE_ROOT}")
print(f"SOURCE_RELEASE: {RELEASE['release']}")
print(f"SOURCE_REVISION: {RELEASE['revision']}")
print(f"EDGE_DATA_ROOT: {EDGE_DATA_ROOT}")
print(f"EDGE_MANIFEST_PATH: {EDGE_MANIFEST_PATH}")
print(f"EDGE_TOKENIZER_PATH: {EDGE_TOKENIZER_PATH}")
print(f"PREPARED_MANIFEST_SHA256: {sha256_file(prepared_path)}")
print(f"TRAIN_SHARDS: {len(observed_shards)}")
print(f"OUTPUT_ROOT: {OUTPUT_ROOT}")
print(f"PRESERVATION_PACKAGE: {PRESERVATION_PACKAGE}")
```

## CELDA 2 — preflight de GPU y perfil, sin crear salida ni entrenar

```python
# CELDA 2 — Aethel Edge: GPU y perfil, sin entrenamiento ni carga de pesos
import torch

if not torch.cuda.is_available():
    raise RuntimeError("No hay GPU CUDA disponible; no se iniciará entrenamiento.")

profile = RELEASE.get("training_profile", {})
expected_profile = {
    "first_session_target_micro_steps": 183680,
    "schedule_total_micro_steps": 734720,
    "checkpoint_every_micro_steps": 4000,
    "metrics_every_micro_steps": 256,
    "sequence_length": 1024,
    "batch_size": 1,
    "gradient_accumulation": 16,
    "precision": "bf16",
    "dim": 512,
    "layers": 4,
    "heads": 8,
    "kv_heads": 2,
    "experts": 8,
    "active_experts": 2,
}
if {key: profile.get(key) for key in expected_profile} != expected_profile:
    raise RuntimeError("El perfil Edge del bundle no coincide con el perfil aprobado para esta sesión.")
if RELEASE.get("training_authorized") is not False:
    raise RuntimeError("El bundle debe permanecer no autorizado hasta una confirmación explícita separada.")
if RELEASE.get("checkpoint_loading_authorized") is not False:
    raise RuntimeError("La primera sesión debe iniciar desde cero, sin cargar checkpoint.")

print("CELDA 2 — preflight GPU Edge; no se creó salida, no se leyó JSONL y no se entrenó.")
print(f"CUDA_AVAILABLE: {torch.cuda.is_available()}")
print(f"GPU_COUNT: {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"GPU_{index}: {torch.cuda.get_device_name(index)}")
print(f"BF16_REPORTED_BY_TORCH: {torch.cuda.is_bf16_supported()}")
print("EXECUTION_STRATEGY: single / world_size=1 (no se usa entrenamiento distribuido)")
print(f"EDGE_PARAMETERS_PROFILE: dim={profile['dim']}, layers={profile['layers']}, experts={profile['experts']}, active_experts={profile['active_experts']}")
print(f"SESSION_TARGET_MICRO_STEPS: {profile['first_session_target_micro_steps']}")
print(f"SCHEDULE_TOTAL_MICRO_STEPS: {profile['schedule_total_micro_steps']}")
print(f"CHECKPOINT_EVERY_MICRO_STEPS: {profile['checkpoint_every_micro_steps']}")
print("AETHEL_EDGE_PRECHECK_READY — no ejecutes CELDA 3 hasta autorizar expresamente la primera sesión GPU.")
```

## CELDA 3 — primera sesión larga, sólo tras autorización explícita final

> **No ejecute esta celda todavía.** Inicia entrenamiento real y crea pesos nuevos. Debe ejecutarse únicamente tras una confirmación explícita separada para esta primera sesión GPU. Es una inicialización desde cero: no establezca `RESUME_CHECKPOINT` ni conecte un dataset de artefactos. Si termina correctamente, use **Save Version** en Kaggle inmediatamente y antes de cerrar, reiniciar o abandonar la sesión.

```python
# CELDA 3 — Aethel Edge: primera sesión larga desde cero; requiere autorización final
import os
import subprocess

if "RESUME_CHECKPOINT" in os.environ:
    raise RuntimeError("Esta es la primera sesión Edge: RESUME_CHECKPOINT no debe estar definido.")
if OUTPUT_ROOT.exists() or PRESERVATION_PACKAGE.exists():
    raise RuntimeError("La salida o el TAR ya existen; no se sobrescribe ni se reanuda.")

environment = dict(os.environ)
environment.update({
    "SOURCE_ROOT": str(SOURCE_ROOT),
    "EDGE_DATA_ROOT": str(EDGE_DATA_ROOT),
    "EDGE_CORPUS_ROOT": str(EDGE_DATA_ROOT),
    "EDGE_MANIFEST_PATH": str(EDGE_MANIFEST_PATH),
    "EDGE_TOKENIZER_PATH": str(EDGE_TOKENIZER_PATH),
    "OUTPUT_ROOT": str(OUTPUT_ROOT),
    "PRESERVATION_PACKAGE": str(PRESERVATION_PACKAGE),
    "PHASE_ID": "edge-phase-1-183680-v1",
    "SESSION_TARGET_STEP": "183680",
    "SCHEDULE_TOTAL_STEPS": "734720",
})
subprocess.run(
    ["bash", str(SOURCE_ROOT / "training" / "run_kaggle_edge_long_session_v1.sh")],
    check=True,
    env=environment,
)

required = [
    OUTPUT_ROOT / "latest.pt",
    OUTPUT_ROOT / "tokenizer.json",
    OUTPUT_ROOT / "recovery_receipt.json",
    OUTPUT_ROOT / "edge_session_preservation_receipt.json",
    OUTPUT_ROOT / "SAVE_KAGGLE_VERSION_NOW.txt",
    PRESERVATION_PACKAGE,
]
missing = [str(path) for path in required if not path.is_file() or (path == PRESERVATION_PACKAGE and path.stat().st_size == 0)]
if missing:
    raise RuntimeError("Faltan artefactos de preservación; no cierre la sesión. Faltantes:\n" + "\n".join(missing))

print("AETHEL_EDGE_PHASE_1_COMPLETE")
print(f"CHECKPOINT: {OUTPUT_ROOT / 'latest.pt'}")
print(f"PRESERVATION_PACKAGE: {PRESERVATION_PACKAGE}")
print("ACCIÓN MANUAL OBLIGATORIA: pulsa Save Version en Kaggle inmediatamente antes de salir.")
```

Al terminar, el resultado será un **checkpoint experimental no promocionado** y su paquete de preservación, no un modelo validado como bilingüe, razonador o funcional. La continuación posterior no se inicia automáticamente: primero se inspeccionará el recibo de preservación y se definirá cómo montar el checkpoint sin romper el contrato de dos inputs.
