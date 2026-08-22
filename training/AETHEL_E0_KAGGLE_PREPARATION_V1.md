# Preparación Controlada de Aethel Seed E0 para Kaggle

**Estado:** guía de preparación. No crea ni actualiza un Dataset, no genera un bundle, no reserva GPU y no inicia entrenamiento.

## Finalidad

E0 es el primer experimento de laboratorio de Aethel Seed sobre datos reales y congelados. Su resultado, si llega a ejecutarse de forma autorizada, será evidencia experimental de pesos, pérdida, perplejidad, telemetría de routing, checkpoint y reanudación; **no** será una promoción comercial ni validará las rutas Triton de producción.

La configuración objetivo de Seed usa cuatro capas, dimensión 512, ocho cabezas, dos cabezas KV, ocho expertos MoE con routing top-2, secuencia 1024, batch 2 y acumulación de gradiente 16. Estos son parámetros de configuración del experimento, no resultados medidos.

## Archivos de código ya aptos y versionados en GitHub

Los siguientes archivos están en el repositorio privado y deben tomarse de la misma revisión de `main` al generar el bundle de fuente privado:

| Archivo | Función |
|---|---|
| `training/build_kaggle_nextgen_source_bundle.sh` | Construye un `aethel-nextgen-source.tar.gz` sin datos, pesos ni artefactos locales. |
| `training/run_kaggle_seed_offline.sh` | Impone autorización humana, valida el paquete antes de CUDA, conserva checkpoints y separa holdouts. |
| `training/AETHEL_SEED_OFFLINE_RUNBOOK_V1.md` | Runbook de ejecución y recuperación. |
| `training/validate_aethel_knowledge_package.py` | Valida hashes, splits y tokenizer de forma offline. |
| `training/inspect_local_aethel_host.py` | Inspecciona código, Dataset, salida, CUDA y Triton sin entrenar. |
| `training/run_triton_cuda_acceptance.py` | Registra aceptación/rechazo de CUDA sin habilitar contratos automáticamente. |
| `engine/train_aethel_gpu.py` | Entrenador Seed con checkpoint y reanudación. |
| `engine/evaluate_nextgen.py` | Evaluación independiente de holdout inglés/español. |
| `engine/aethel_model.py` y `engine/triton_bridge.py` | Modelo, contratos y rutas Triton de estado parcial. |

## Activos locales que no se suben a GitHub

| Ubicación local | Contenido | Regla |
|---|---|---|
| `/home/ubuntu/aethel-knowledge-corpus-v1-package/` | Paquete congelado: 22 shards comprimidos, tokenizer, manifiestos, hashes y reportes. | Respaldar privado íntegro. Sólo puede llegar a Kaggle como Dataset privado tras confirmación inmediata. |
| `/home/ubuntu/aethel-artifacts/nextgen_aligned/` | Experimento CPU de 100 pasos y metadatos. | Conservar como evidencia experimental; no subir a GitHub ni promover. |
| `/home/ubuntu/aethel-artifacts/nextgen_previous/` | Experimento CPU previo de 100 pasos y metadatos. | Conservar como evidencia experimental; no subir a GitHub ni promover. |
| `engine/artifacts/aethel_real.pt` | Artefacto histórico ya versionado y no certificado. | No cargar, deserializar, ejecutar ni promocionar. |

No se suben `node_modules`, `runtime/aethel-memory-rust/target`, cachés, logs del servidor, `.env`, cookies, credenciales ni secretos.

## Relevo seguro para otro chat

El siguiente chat debe partir de la rama `main` más reciente del repositorio privado, leer esta guía junto con el documento de continuidad y generar el bundle de código desde esa misma revisión. GitHub contiene únicamente fuentes, scripts, documentación y metadatos aptos; **no** contiene el Dataset congelado, checkpoints CPU externos, secretos ni una sesión de Kaggle.

El otro chat puede preparar explicaciones, validar archivos de código y construir el bundle privado de fuentes. Las acciones sobre la cuenta Kaggle del usuario —incluida la creación o actualización del Dataset privado, la carga del bundle, la configuración de Notebook, GPU y E0— pertenecen exclusivamente a una sesión My Browser realmente conectada y requieren las confirmaciones indicadas más abajo.

## Secuencia obligatoria en Kaggle

1. **Sesión personal.** Verificar que la navegación opera en My Browser del usuario. Si hay sandbox, login, CAPTCHA o pantalla de conexión, detenerse sin alternativa automática.
2. **Confirmación por acción.** Antes de crear/actualizar Dataset, adjuntar entradas, editar/crear Notebook, seleccionar GPU, guardar una versión o iniciar una corrida, pedir confirmación explícita e inmediata para esa acción exacta.
3. **Entrada de datos.** El Dataset debe ser privado. Puede usar el nombre previsto `aethel-nextgen-data-v1`; si el slug final difiere del predeterminado, fijar `AETHEL_DATA_DIR` a su ruta real de `/kaggle/input/<slug>`. El contenido debe conservar la estructura raíz del paquete congelado y los 22 shards bajo `corpus/`.
4. **Entrada de código.** Generar y subir, también como entrada privada separada, el bundle `aethel-nextgen-source.tar.gz` y su manifiesto desde la misma revisión de `main`. El bundle no sustituye el Dataset de conocimiento.
5. **Preflight sin red.** Mantener Internet desactivado. Montar código y datos, ejecutar el validador offline y conservar `package_preflight.json` antes de consultar CUDA.
6. **Prueba de CUDA/Triton.** Cuando haya un host GPU autorizado, ejecutar primero `training/run_triton_cuda_acceptance.py`. Un resultado `NOT_RUN`, fallo o evidencia parcial no habilita las rutas Triton estrictas.
7. **E0 sólo con doble autorización.** `AETHEL_RUN_AUTHORIZED=YES` autoriza la corrida y `AETHEL_LAB_FALLBACK_AUTHORIZED=YES` permite exclusivamente un fallback PyTorch experimental mientras Triton estricto siga bloqueado. Sin ambas variables, E0 debe detenerse.

## Entorno de Notebook preparado, no autorizado

```python
import os

os.environ["AETHEL_SOURCE_DIR"] = "/kaggle/working/aethel-nextgen-source"
os.environ["AETHEL_DATA_DIR"] = "/kaggle/input/aethel-nextgen-data-v1"
os.environ["AETHEL_OUTPUT_DIR"] = "/kaggle/working/aethel-runs/aethel-seed-e0"
os.environ["AETHEL_MAX_STEPS"] = "4992"
os.environ["AETHEL_SAVE_EVERY"] = "192"
os.environ["AETHEL_KEEP_SNAPSHOTS"] = "3"
os.environ["AETHEL_RUN_AUTHORIZED"] = ""
os.environ["AETHEL_LAB_FALLBACK_AUTHORIZED"] = "NO"
```

El lanzador sólo debe invocarse después de la doble autorización correspondiente. Una ejecución E0 válida debe conservar `latest.pt`, `step_*.pt`, `recovery_receipt.json`, `metrics_rank_0.jsonl`, `package_preflight.json`, `checkpoint_inspection.json`, `evaluation_holdout_en.json` y `evaluation_holdout_es.json` en salida privada.

## Límites de interpretación

E0 no autoriza cambiar La Roca, usar holdout para entrenamiento, promover memoria Líquida, promover un candidato de Sueño, anunciar capacidad comercial, declarar soporte Triton de producción ni iniciar Edge. Cada afirmación posterior debe salir de artefactos realmente observados y verificables.
