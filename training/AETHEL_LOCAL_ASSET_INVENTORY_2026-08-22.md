# Inventario local de activos de Aethel

**Fecha de inspección:** 22 de agosto de 2026  
**Propósito:** separar código sincronizado, activos locales que deben preservarse, salidas regenerables y artefactos que no poseen evidencia suficiente para promoción.

> **Regla de persistencia:** estar presente en esta máquina no significa estar en GitHub, Kaggle, S3 ni en un host GPU. Un activo sólo se considera recuperable fuera de este host si existe una copia identificada en el destino correspondiente y se ha verificado su hash.

## Clasificación actual

| Activo o ruta | Presencia y evidencia | Ubicación de persistencia | Acción permitida | Acción prohibida sin autorización |
|---|---|---|---|---|
| Código, documentos y tests | 729 archivos versionados; rama `main` sincronizada hasta `54b2e858`. | GitHub privado y checkout local. | Editar, probar, versionar y restaurar desde Git. | Confundir el código con un modelo entrenado. |
| Dataset congelado | 194 MB, 22 shards, tokenizador y manifiestos; hashes de manifiesto, reporte y tokenizador registrados abajo. | Sólo `/home/ubuntu/aethel-knowledge-corpus-v1-package/` en este entorno. | Leer, validar hashes y montar como entrada de una corrida autorizada. | Alterar, borrar, publicar o subir a Kaggle sin autorización inmediata. |
| `aethel-artifacts/nextgen_previous` | Checkpoint local CPU de paso 100 (66.971.417 bytes), métricas y memoria episódica. | Sólo `/home/ubuntu/aethel-artifacts/nextgen_previous/`. | Preservar y auditar estáticamente. | Declararlo Seed, evaluación holdout, benchmark o candidato de promoción. |
| `aethel-artifacts/nextgen_aligned` | Checkpoint local CPU de paso 100 (69.371.620 bytes), métricas, memoria episódica y versiones líquidas. | Sólo `/home/ubuntu/aethel-artifacts/nextgen_aligned/`. | Preservar y auditar estáticamente. | Cargarlo, reanudarlo, promocionarlo o usarlo como inferencia sin proceso confiable y autorización. |
| `engine/artifacts/aethel_real.pt` | Archivo versionado de 991.114 bytes; contenedor PyTorch inspeccionado de forma estática. | GitHub y checkout local. | Conservar con su sidecar `aethel_real.audit.json`. | Usarlo como evidencia de Aethel entrenado, benchmark, inferencia o reanudación. |
| `node_modules/`, `runtime/.../target/` | Dependencias y productos de compilación regenerables. | Sólo esta máquina. | Recrear mediante gestores de dependencias y Cargo. | Tratarlo como fuente de verdad, Dataset o checkpoint. |
| `.manus-logs/` | Logs locales de desarrollo, aproximadamente 1,2 MB al inspeccionarse. | Sólo esta máquina y rotación del entorno. | Consultar para depuración. | Considerarlos telemetría de entrenamiento o archivo persistente. |

## Huellas de integridad registradas

| Archivo | SHA-256 observado |
|---|---|
| `aethel-knowledge-corpus-v1-package/package_manifest.json` | `91cfd0e2b14ba6a863143f17ff85629e5f28c88cf13b09627ab8ef34bc78435a` |
| `aethel-knowledge-corpus-v1-package/package_validation_report.json` | `ef39fa3984536c974e40c97d25f701583ce4e4ecb6200dd74d5060657bd439b3` |
| `aethel-knowledge-corpus-v1-package/tokenizer.json` | `4a3608e4e45c9117415d1f4fa236aebe20771dc3a3ce85760d9fb9d218fa0815` |
| `aethel-artifacts/nextgen_previous/nextgen_step_100.pt` | `1785400f0bef47bc009098821e60b57b7c427366318a40573f0a543eba22bbaa` |
| `aethel-artifacts/nextgen_aligned/nextgen_step_100.pt` | `20f137b5dce18d0ef539f6b76cf79a33c61c34ba7e49b2359eb3fb9d83882863` |
| `engine/artifacts/aethel_real.pt` | `fa423241ff0d94ea5819e9628c41d16940a4e846c5c625c030a9cbc0a9162122` |

## Interpretación de las corridas locales externas

Los directorios `nextgen_previous` y `nextgen_aligned` contienen señales de experimentos de CPU de 100 pasos. Sus JSONL declaran dispositivos CPU, presupuestos de aproximadamente 5,58 M y 5,78 M parámetros respectivamente, y trazas de memoria/líquido en el segundo caso. La existencia de esos registros demuestra que hubo **experimentos locales acotados**; no establece la procedencia completa de datos, tokenizador, configuración inmutable, evaluación holdout independiente, calidad bilingüe ni reproducibilidad fuera de esta máquina. [1]

Por ello, ambos se clasifican como **artefactos experimentales locales, no certificados y no promocionables**. Se conservan para investigación y para una posible auditoría aislada, no para alimentar afirmaciones de producto ni para saltar la primera corrida Seed trazable.

## Ruta segura de preservación futura

Antes de que esta máquina pueda dejar de ser la única copia, hay que decidir explícitamente un destino y una política: Dataset privado de Kaggle, almacenamiento privado de objetos, repositorio de artefactos o volumen persistente de un host propio. La copia deberá tener manifiesto, hashes, licencia/procedencia, ACL privada y verificación posterior. Para checkpoints futuros, el mínimo es configuración, Dataset/tokenizador hash, paso, estados requeridos, métricas, evaluación holdout y metadatos de hardware. [2]

No se copió, borró, publicó ni deserializó ningún activo durante este inventario.

## Referencias

[1]: `/home/ubuntu/aethel-artifacts/nextgen_previous/nextgen_metrics.jsonl`, `/home/ubuntu/aethel-artifacts/nextgen_aligned/nextgen_metrics.jsonl` y los dos contenedores `nextgen_step_100.pt` — Inspección estática de rutas, tamaños, hashes y JSONL locales.  
[2]: `AETHEL_SEED_OFFLINE_RUNBOOK_V1.md` y `AETHEL_AUDIT_RESPONSE_2026-08-22.md` — Requisitos de evidencia, artefactos y promoción.
