# Contrato de auditoría estática D0 — Seed E0 V8

## Finalidad

**D0** es una auditoría de configuración y evidencia estática. Su única salida permitida es enlazar, de forma verificable, el release privado de código, la evidencia auditada de E0 V8 y el manifiesto raíz del Dataset congelado. No vuelve a evaluar E0, no genera una comparación de candidatos y no selecciona una configuración para D1.

> **Clasificación:** D0 se ejecutó una vez en Kaggle tras confirmaciones separadas para actualizar el Dataset privado de código, añadir la celda y ejecutarla. El resultado fue `D0_AUDIT_READY`. Esta evidencia no autoriza GPU, `Save Version`, `Save & Run All`, D1, reanudación de E0, carga de pesos ni promoción.

## Entradas autorizadas y lecturas prohibidas

| Elemento | Lectura D0 permitida | Invariante comprobada |
|---|---|---|
| `training/aethel_kaggle_source_release.json` | JSON de identidad del bundle de código | `release = d0-v1-e0-v8-static-audit` y alcance estático D0. |
| `training/e0_v8_d0_evidence.json` | JSON versionado de evidencia, sin pesos ni texto | Paso final 4.992, contrato estructural del checkpoint, router final no saludable y límites D0. |
| `package_manifest.json` del Dataset privado | Sólo metadatos raíz y su SHA-256 | ID del Dataset, conteos congelados, hash del tokenizer y exclusión del holdout del tokenizer. |

El auditor **no** importa PyTorch, no consulta ni activa CUDA, no abre `latest.pt` ni snapshots, no llama a `inspect_checkpoint.py`, no recorre directorios `corpus/`, no abre shards ni texto del holdout, no valida el corpus completo, no hace solicitudes de red, no entrena y no modifica ningún Dataset de Kaggle.

## Entry points

La comprobación local ya validada se ejecuta desde el código fuente con:

```bash
python3 training/audit_e0_v8_d0.py \
  --source-root /home/ubuntu/aethel-platform \
  --data-root /home/ubuntu/aethel-knowledge-corpus-v1-package \
  --output-dir /home/ubuntu/aethel-private-transfer/d0-local-output
```

La celda privada preparada es `AETHEL_D0_E0_V8_STATIC_AUDIT_CELL.py`. Primero localiza exactamente un bundle con el release D0, exige los tres archivos D0 requeridos y comprueba que `package_manifest.json` exista. Sólo después copia código al directorio nuevo de trabajo de Kaggle y ejecuta el auditor. Si ya existe ese directorio, la celda se detiene: D0 no borra salidas previas ni entradas montadas.

## Evidencia de ejecución Kaggle

La celda se ejecutó contra `SOURCE_INPUT_D0` montado bajo la versión `(9)` del Dataset privado de código y resolvió exactamente `d0-v1-e0-v8-static-audit`. La salida impresa por el notebook declaró `status: D0_AUDIT_READY`; no se reutilizaron las celdas históricas de preflight, smoke CUDA o lanzamiento E0 V8.

| Campo verificado en la salida | Valor observado |
|---|---|
| `completed_steps` | `4992` |
| `dataset.package_manifest_sha256` | `91cfd0e2b14ba6a863143f17ff85629e5f28c88cf13b09627ab8ef34bc78435a` |
| `dataset.tokenizer_sha256` | `4a3608e4e45c9117415d1f4fa236aebe20771dc3a3ce85760d9fb9d218fa0815` |
| `limits.checkpoint_loaded` / `limits.gpu_used` | `false` / `false` |
| `limits.raw_corpus_read` / `limits.holdout_content_read` | `false` / `false` |
| `limits.network_requests` | `0` |
| `router_final_healthy` | `false` |

La salida también conservó los hashes de evidencia (`9716646baaf664348fa34db04c406c1d571b223c71fa9d19b4440e42057237e7`) y del marcador fuente (`09455a52a65daa02bf976043cbc105c03247f3a7bfe25cb3a3ebe0e89e00549f`). Estos valores acreditan consistencia de contrato, no calidad nueva, causalidad de la brecha EN/ES ni elegibilidad para D1.

## Reporte esperado e interpretación

El único resultado correcto es `D0_AUDIT_READY` junto con `output/d0_audit.json`. El reporte debe contener los campos siguientes; son una prueba del contrato estático, no métricas nuevas de calidad.

| Campo | Valor o condición esperada |
|---|---|
| `source_release` | `d0-v1-e0-v8-static-audit` |
| `e0_evidence_release` | `e0-v8-canonical-cuda-device-check` |
| `dataset.package_manifest_sha256` | `91cfd0e2b14ba6a863143f17ff85629e5f28c88cf13b09627ab8ef34bc78435a` |
| `dataset.manifest_metadata_verified` | `true` |
| `checkpoint.checkpoint_loaded` | `false` |
| `holdout_scope.holdout_content_read` | `false` |
| `limits.gpu_used` | `false` |
| `limits.network_requests` | `0` |
| `router_final_healthy` | `false` |

Un fallo de release, hash, manifiesto, evidencia o límite bloquea D0. El comportamiento correcto es detenerse sin inferir una causa y sin proponer D1. No se permite suavizar el contrato, sustituir el Dataset, usar el holdout para reparar la discrepancia EN/ES ni transformar un fallo en una autorización de GPU.

## Límite de decisión

Incluso si D0 produce `D0_AUDIT_READY`, no cambia la clasificación de E0 V8: sigue siendo un experimento PyTorch experimental, no promocionable, con una brecha EN/ES observada y router MoE final no saludable. D0 tampoco autoriza un candidato, un benchmark, serving, Edge, Pro, Sueño/LoRA o movimiento/reanudación del checkpoint.

Una eventual **D1** requiere un plan de diagnóstico separado, una nueva versión de código, una confirmación inmediata para usar GPU y otra confirmación específica para ejecutar la celda. D1 sólo podrá usar shards de entrenamiento; el holdout permanecerá aislado hasta una auditoría final del candidato ya fijado por telemetría de entrenamiento.
