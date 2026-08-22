# Directiva de Ejecución Kaggle: Dataset Privado y Aethel Seed E0

**Estado:** instrucción operativa para una sesión que tenga acceso real a **My Browser** y a la cuenta Kaggle personal del usuario. No constituye autorización permanente. Cada acción externa sensible requiere confirmación inmediata del usuario.

## 1. Misión inmediata

La tarea inmediata no es entrenar Edge ni Pro. Es crear o actualizar el Dataset privado congelado y, después, producir la primera evidencia experimental **Aethel Seed E0**. E0 debe demostrar que el pipeline puede validar datos, iniciar un entrenamiento controlado, crear checkpoints atómicos, reanudar y evaluar el holdout en inglés y español. Sus resultados, si existen, serán experimentales y no promocionables.

## 2. Control de acceso y confirmaciones

Usar exclusivamente la sesión personal de Kaggle expuesta por **My Browser**. Si la sesión muestra sandbox, login, CAPTCHA, aviso de conexión o exige intervención manual, detenerse. No usar una alternativa aislada.

Solicitar confirmación explícita e inmediata, por separado, antes de cada una de estas acciones: crear/actualizar un Dataset, subir una versión de datos, crear/editar un Notebook, adjuntar un input, seleccionar acelerador, guardar una versión o iniciar una ejecución. No publicar ningún recurso.

## 3. Dataset privado

| Elemento | Valor requerido |
|---|---|
| Nombre previsto | `aethel-nextgen-data-v1` |
| Visibilidad | Privada |
| Origen local controlado | `/home/ubuntu/aethel-knowledge-corpus-v1-package/` |
| Contenido | 22 shards JSONL.gz bajo `corpus/`, tokenizer, manifiestos, splits y reportes de validación en la raíz |
| Contratos en GitHub | `training/dataset_contracts/` |

El paquete congelado contiene 40.000 documentos de Wikipedia, con 38.023 de entrenamiento y 1.977 holdout; la separación lingüística documentada es 19.011/989 en inglés y 19.012/988 en español. Estos son metadatos de paquete validado, no una garantía de capacidad de modelo. No alterar shards, hashes, tokenizer, particiones ni holdout.

Tras crear o actualizar la versión privada, verificar visualmente que la visibilidad sigue siendo privada y que la estructura es exacta. No iniciar E0 en el mismo paso sin la siguiente confirmación independiente.

## 4. Bundle de código privado

Generar desde la misma revisión de `main` el bundle sin datos ni pesos:

```bash
bash training/build_kaggle_nextgen_source_bundle.sh
```

El resultado esperado es `aethel-nextgen-source.tar.gz` junto con su manifiesto. Subirlo como input privado separado sólo después de confirmación inmediata. GitHub ya contiene el código, guías, dependencias, validadores y metadatos necesarios; no contiene los shards ni checkpoints.

## 5. Qué es Seed E0

E0 es un **experimento de calibración de laboratorio**. La configuración propuesta tiene cuatro capas, dimensión 512, ocho cabezas de atención, dos cabezas KV, ocho expertos MoE con routing top-2, contexto 1.024, batch 2 y acumulación de gradiente 16. La interfaz describe aproximadamente 97,16 M de parámetros teóricos; el número admisible será el que se registre desde una instancia/checkpoint ejecutado, no una cifra declarada antes de correr.

E0 no prueba inteligencia comercial. Mide únicamente la integridad del circuito: datos congelados, tokenizer, pérdida/perplejidad observadas, checkpoints, reanudación, telemetría de routing y evaluación independiente por idioma.

## 6. Secuencia E0 autorizable

1. Adjuntar el Dataset privado y el bundle de código privado al Notebook, tras confirmación.
2. Mantener Internet desactivado. Extraer el bundle y fijar rutas reales de datos, fuente y salida.
3. Ejecutar `training/validate_aethel_knowledge_package.py` y conservar `package_preflight.json` antes de consultar CUDA.
4. Con GPU autorizada, ejecutar primero `training/run_triton_cuda_acceptance.py`. `NOT_RUN`, fallo o cobertura parcial no habilitan Triton estricto.
5. Pedir una confirmación separada para iniciar E0. La ruta actual exige `AETHEL_RUN_AUTHORIZED=YES` y, mientras Triton estricto siga bloqueado, `AETHEL_LAB_FALLBACK_AUTHORIZED=YES` para un fallback PyTorch experimental no promocionable.
6. Invocar `training/run_kaggle_seed_offline.sh` y preservar salida privada: `latest.pt`, `step_*.pt`, `recovery_receipt.json`, `metrics_rank_0.jsonl`, `package_preflight.json`, `checkpoint_inspection.json`, `evaluation_holdout_en.json` y `evaluation_holdout_es.json`.
7. Antes del límite de sesión, guardar una versión/commit del Notebook sólo con confirmación inmediata. Al reanudar, validar hashes y receipts antes de continuar.

La configuración preparada fija como referencia `AETHEL_MAX_STEPS=4992`, guardado cada 192 pasos y retención de tres snapshots. Son parámetros de ejecución previstos, no una corrida activa.

## 7. Ruta de variantes y objetivo de producto

| Variante | Arquitectura objetivo | Parámetros de diseño | Objetivo y puerta de avance |
|---|---|---:|---|
| Seed E0 | 4 capas, dim 512, 8 cabezas, 2 KV heads, 8 expertos top-2 | Se registrará desde ejecución; ≈97,16 M teóricos en interfaz | Evidencia de Dataset, router, checkpoint, reanudación y holdout bilingüe. |
| Edge | Denso, 28 capas, dim 2.560, 20 cabezas, 5 KV heads | ≈2,2 B | Asistente técnico bilingüe privado, cuantizable, con memoria gobernada; requiere evidencia E0 real, datos suficientes, memoria/infraestructura planificada, calidad y latencia observadas. |
| Pro | 32 capas, dim 4.096, 32 cabezas, 8 KV heads, 8 expertos top-2, SwiGLU 11.008 | ≈36,4 B totales; ≈10,4 B activos/token | Producto empresarial de razonamiento/conocimiento técnico bilingüe; requiere kernels Triton prefill/dispatch validados, router sano, recuperación distribuida y coste medido. |
| Research | Sin microarquitectura congelada | ≈139 B de referencia de familia | Investigación multi-GPU posterior a evidencia Edge/Pro; no es lanzamiento inicial. |

Todas las variantes comparten RoPE, GQA cuando aplica, KV-cache, telemetría de router y los cinco pilares cognitivos: **La Roca** inmutable, **Líquido** versionado y reversible, **Curiosidad** acotada, **Sueño** con candidatos LoRA aislados y aprobación independiente, y **MoE** top-2 con capacidad, overflow y balanceo verificables.

## 8. Prohibiciones de promoción

Una ejecución E0 no permite afirmar soporte Triton de producción, iniciar Edge, vender un modelo, modificar La Roca, entrenar con holdout, promover memoria Líquida ni promover candidatos de Sueño. Cada avance requiere evidencia de artefactos observables, pruebas por idioma, seguridad, utilidad y coste medido.

## 9. Archivos que debe leer el siguiente chat

- `AETHEL_PROJECT_HANDOFF_2026-08-22.md`
- `training/AETHEL_E0_KAGGLE_PREPARATION_V1.md`
- `training/AETHEL_SEED_OFFLINE_RUNBOOK_V1.md`
- `training/AETHEL_TECHNICAL_VARIANTS_SPEC_V1.md`
- `training/dataset_contracts/README.md`
- `training/AETHEL_TRITON_CUDA_ACCEPTANCE_MATRIX_V1.md`

