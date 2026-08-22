# Runbook de Aethel Seed Offline v1

**Estado:** preparado y validado localmente; **no autoriza** reservar GPU, iniciar una corrida, publicar datos ni promover pesos.  
**Propósito:** ejecutar el primer experimento E0 de Aethel Seed con un paquete de datos congelado, controles de integridad, checkpoints atómicos y evaluación exclusivamente sobre holdout.

## Principio operativo

La corrida no debe construir corpus ni descargar fuentes de Internet. El único Dataset admisible es el paquete privado `aethel-knowledge-corpus-v1-package`, cuyo manifiesto enumera los shards, sus hashes y el tokenizador generado exclusivamente a partir de `train`. La ruta `training/run_kaggle_seed_offline.sh` ejecuta el validador offline antes de comprobar CUDA; por tanto, un paquete, manifiesto o tokenizador incompatibles se bloquean sin consumir una GPU.

> El objetivo de E0 no es declarar Aethel comercial ni comparable con un modelo de frontera. Es producir por primera vez pesos, pérdida, perplejidad, telemetría de routing y evidencia de reanudación sobre datos reales, preservando la separación entre `train` y `holdout`.

| Activo | Ubicación esperada en la sesión | Control obligatorio |
|---|---|---|
| Código Aethel | `AETHEL_SOURCE_DIR` | Debe incluir el lanzador, entrenador, evaluador, inspector y pruebas Triton. |
| Paquete congelado | `AETHEL_DATA_DIR` | Debe contener `package_manifest.json`, `metadata.json`, `validation_report.json`, `tokenizer.json` y `corpus/`. |
| Salida de corrida | `AETHEL_OUTPUT_DIR` | Debe incluir checkpoint, tokenizador copiado, recibo de recuperación, métricas, preflight y evaluaciones. |

## Preparación de Notebook

En Kaggle se adjuntan como entradas privadas el bundle de código Aethel y el paquete congelado de conocimiento. La opción de Internet debe permanecer desactivada porque el flujo no necesita red. Se selecciona un acelerador GPU y se fija una salida bajo `/kaggle/working`; esa salida sólo se considera persistida al terminar la corrida y guardar la versión del Notebook, o al exportarla explícitamente a un Dataset privado de artefactos.

La celda de arranque debe declarar únicamente rutas y límites de experimento. No debe copiar secretos a una celda, incorporar credenciales a Git ni volver a ejecutar materializadores de corpus.

```python
import os

os.environ["AETHEL_SOURCE_DIR"] = "/kaggle/working/aethel-nextgen-source"
os.environ["AETHEL_DATA_DIR"] = "/kaggle/input/aethel-knowledge-corpus-v1-package"
os.environ["AETHEL_OUTPUT_DIR"] = "/kaggle/working/aethel-runs/aethel-seed-e0"
os.environ["AETHEL_MAX_STEPS"] = "4992"
os.environ["AETHEL_SAVE_EVERY"] = "192"
os.environ["AETHEL_KEEP_SNAPSHOTS"] = "3"

# Se deja vacío hasta una autorización explícita de ejecución.
os.environ["AETHEL_RUN_AUTHORIZED"] = ""

# Sigue siendo NO: la CUDA estricta permanece bloqueada hasta que atención
# causal de prefill y dispatch/combina MoE cuenten con kernels Triton validados.
os.environ["AETHEL_LAB_FALLBACK_AUTHORIZED"] = "NO"
```

La llamada posterior es `bash "$AETHEL_SOURCE_DIR/training/run_kaggle_seed_offline.sh"`. Si `AETHEL_RUN_AUTHORIZED` está vacío, el proceso termina deliberadamente con código 3. Si el paquete está incompleto o el tokenizador no coincide con su hash, termina antes de consultar el acelerador. Si Triton estricto no está completo, termina con código 6 antes de actualizar un solo peso.

## Condición de ejecución experimental

Un E0 de laboratorio con operaciones PyTorch en CUDA requiere dos aprobaciones explícitas: `AETHEL_RUN_AUTHORIZED=YES` y `AETHEL_LAB_FALLBACK_AUTHORIZED=YES`. Esta segunda aprobación sólo habilita una medición de línea base; no elimina el contrato de Triton, no valida kernels pendientes y no permite promoción a La Roca, a un candidato de Sueño ni a un producto comercial.

| Puerta | Evidencia de aprobación | Resultado al fallar |
|---|---|---|
| Autorización humana | `AETHEL_RUN_AUTHORIZED=YES` | Bloqueo antes de lectura del Dataset. |
| Integridad de paquete | `package_preflight.json` válido | Bloqueo antes de CUDA si faltan controles, shard o tokenizador. |
| GPU | Dispositivo CUDA presente | Bloqueo si el Notebook no tiene acelerador GPU. |
| Contrato de kernels | Triton completo, o permiso E0 de fallback | Bloqueo comercial estricto; fallback etiquetado como experimental. |
| Reanudación | `latest.pt` inspeccionable y `recovery_receipt.json` | No se puede comparar ni retomar desde un checkpoint incompleto. |

## Checkpoint y reanudación

El entrenador escribe `latest.pt` de forma atómica sólo después de una actualización de optimizador y conserva tres snapshots portátiles `step_*.pt`. El checkpoint incluye pesos, optimizador, paso, configuración, hash de tokenizador y estrategia de distribución. La reanudación se bloquea si cambia la topología o el tokenizador.

Una salida válida debe contener los siguientes artefactos. La ausencia de cualquiera impide tratar la corrida como reproducible.

| Artefacto | Función |
|---|---|
| `package_preflight.json` | Registro de hashes, idioma y separación de splits del paquete montado. |
| `latest.pt` | Punto de reanudación completo, incluyendo optimizador. |
| `step_*.pt` | Copias portátiles sin optimizador para contingencia. |
| `recovery_receipt.json` | Paso seguro más reciente y contrato de recuperación. |
| `metrics_rank_0.jsonl` | Pérdida, tokens/s, carga/entropía de expertos y configuración por paso. |
| `checkpoint_inspection.json` | Confirmación de que el checkpoint contiene metadatos reproducibles. |
| `evaluation_holdout_en.json` / `evaluation_holdout_es.json` | Pérdida y perplejidad en datos retenidos, nunca usados para actualización. |

Un Commit de Notebook conserva resultados sólo si la ejecución llega a finalizar y el usuario guarda la versión. No protege frente a una interrupción abrupta antes de que el output se persista. Por ello, un experimento gratuito debe ser deliberadamente corto y comenzar con un presupuesto de pasos que quepa en una sesión; un entrenamiento Edge prolongado necesita un destino persistente y una GPU estable.

## Lectura de resultados y límites

La lectura inicial se limita a valores observados: pérdida y perplejidad en ambos idiomas, estado de router, VRAM, tokens/s, continuidad del paso tras restaurar `latest.pt`, e igualdad de hashes de La Roca. No se fijan umbrales de éxito antes de E0 ni se reportan cifras hipotéticas.

Una E0 que complete sus artefactos no autoriza las siguientes acciones: cambiar La Roca, usar holdout para ajustar, consolidar recuerdos de Líquido, crear una promoción de Sueño, habilitar acciones externas o anunciar capacidad comercial. Esas acciones permanecen sujetas al contrato experimental y a las puertas P0–P6 de `AETHEL_COGNITIVE_EXPERIMENT_CONTRACT_V1.md`.
