# Auditoría de artefactos — Seed E0 V8

> **Estado:** evidencia de una corrida Seed E0 completada y persistida. Esto **no** certifica utilidad, calidad comercial, promoción de modelo, soporte Triton estricto ni disponibilidad de Edge/Pro.

## Alcance y fuente de evidencia

La evidencia fue aportada desde el panel **Output** de Kaggle para `aethel-e0-seed-v8/output`. No se descargó, cargó ni deserializó `latest.pt` ni los snapshots. La auditoría usa únicamente logs y metadatos textuales persistidos.

| Puerta o artefacto | Evidencia inspeccionada | Resultado verificable |
|---|---|---|
| Release de código | `launch_manifest.json` | `e0-v8-canonical-cuda-device-check` desde la carpeta de input Kaggle `(8)` |
| Preflight | Log persistido | 22 shards validados; red cero; autorizaciones previas en `false` |
| Smoke de memoria | Salida de celda | `VERIFIED_LIQUID_CUDA_ALIGNMENT` antes de E0 |
| Entrenamiento | `metrics_rank_0.jsonl` | 4,992 registros, pasos 1–4,992, `cuda`, `world_size: 1` |
| Checkpoint final | `checkpoint_inspection.json` | `latest.pt` presente, 150 tensores, metadatos completos, paso 4,992 |
| Recuperación | `recovery_receipt.json` | snapshot final y tres snapshots retenidos, con contrato de reanudación |
| Holdout EN | `evaluation_holdout_en.json` | 256 segmentos aislados evaluados |
| Holdout ES | `evaluation_holdout_es.json` | 256 segmentos aislados evaluados |

## Persistencia y reanudación

`recovery_receipt.json` registra `latest.pt` en el paso final **4,992**, con snapshots retenidos `step_00004608.pt`, `step_00004800.pt` y `step_00004992.pt`. El recibo fija el SHA-256 de tokenizer `4a3608e4e45c9117415d1f4fa236aebe20771dc3a3ce85760d9fb9d218fa0815` y describe un contrato de reanudación que requiere importar el Output comprometido como Dataset privado e indicar explícitamente `AETHEL_RESUME_CHECKPOINT`.

La inspección estructural declara 150 tensores, metadatos completos y `reproducible_resume: true`. También informa `parameter_count: 113,539,140`, mientras que la telemetría del entrenamiento informa `parameters_trainable: 97,154,564`. Ambos campos se conservan textualmente y **no deben tratarse como la misma medida** hasta documentar su semántica exacta.

## Métricas de entrenamiento observadas

| Indicador | Resultado observado |
|---|---:|
| Paso inicial / final | 1 / 4,992 |
| Tokens vistos finales | 10,223,616 |
| Pérdida en paso 1 | 10.422280311584473 |
| Pérdida final | 7.821859836578369 |
| Mínimo puntual de pérdida | 6.273062705993652 (paso 3,850) |
| Media de pérdida, últimas 500 muestras | 7.7812168397903445 |
| Rendimiento mediano | 8,054.020597191363 tokens/s |
| Dispositivo / world size | `cuda` / 1 |
| Parámetros entrenables reportados | 97,154,564 |

La serie confirma que el intento no fue simulado ni se abortó antes de la primera persistencia. La pérdida desciende respecto del inicio, pero esta información aislada no establece calidad conversacional, razonamiento, seguridad ni comparabilidad con otro modelo.

## Evaluación holdout aislada

| Split | Segmentos | Pérdida | Perplejidad |
|---|---:|---:|---:|
| EN | 256 | 7.771877005696297 | 2372.921097307039 |
| ES | 256 | 10.774345595389605 | 47779.19608484159 |

Los dos ficheros apuntan a `latest.pt`, se ejecutaron en CUDA y se identifican como `prepared_validation_holdout`. La diferencia EN/ES es una señal experimental que exige diagnóstico y baselines aprobadas; no es una justificación para promoción.

## Router, memoria y autonomía acotada

El registro contiene 156 pasos con `router_health.healthy: true` y 4,836 con `false`. En el último paso reporta `healthy: false`, `max_imbalance: 0.16326904296875` y `min_entropy: 0.47724199295043945`; en toda la serie, el mínimo de entropía fue `0.3333333432674408` y el máximo de imbalance `0.1875`. Por ello el router top-2 se considera **telemetría existente, no aceptación de salud o producción**.

Al cierre, la telemetría registra 4,916 memory hits, 49 registros episódicos, 10 semánticos, 49 de replay y versión líquida 49. Curiosidad mantuvo 2,048 acciones `observe_only`, cero acciones externas y `external_action_enabled: false`. Esto acredita el funcionamiento registrado de las salvaguardas acotadas, no autonomía abierta.

## Límites vigentes

- `latest.pt` y snapshots siguen como artefactos privados de Kaggle: no se copiaron a GitHub ni se promovieron.
- El histórico `engine/artifacts/aethel_real.pt` permanece prohibido y no se relaciona con esta auditoría.
- No se ejecutó un benchmark comparativo, un conjunto de prompts cualitativos ni evaluación de seguridad.
- La ruta Triton estricta, Edge, Pro y afirmaciones comerciales siguen bloqueadas por sus propios requisitos de evidencia.
