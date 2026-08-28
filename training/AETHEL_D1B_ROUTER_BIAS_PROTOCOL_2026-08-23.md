# Protocolo D1B — prueba controlada del sesgo de router MoE

**Fecha:** 23 de agosto de 2026
**Estado:** `D1B_DIAGNOSTIC_COMPLETE_ROUTER_NOT_IMPROVED`
**Naturaleza:** diagnóstico único D1B completado desde inicialización nueva; el router no superó su puerta de salud, por lo que no hay candidato, promoción ni autorización de D2/D3.

## 1. Propósito limitado

D1B fue una única prueba *train-only* desde inicialización nueva para examinar si una corrección más lenta del sesgo de carga del router reducía las violaciones observadas de entropía sin exceder el umbral de desequilibrio. No fue un entrenamiento de candidato, una mejora demostrada, una evaluación de calidad lingüística ni un paso hacia serving.

La evidencia segura D1A registró 78 de 768 pasos saludables y 690 no saludables. El máximo desequilibrio observado fue 0,187500, por debajo del umbral de 0,30, mientras la entropía mínima fue 0,333333, por debajo del umbral de 0,50. [1] El entrenador etiqueta un paso como saludable solamente si se cumplen **ambos** límites. [2]

> **Hipótesis falsable:** al reducir exclusivamente `router_bias_step` de `0.05` a `0.01`, una nueva ventana D1B con las mismas condiciones tendrá una proporción de pasos saludables mayor que la de D1A, sin observar desequilibrio máximo superior a `0.30`. Esta no es una afirmación causal: D1A no demostró que el paso del sesgo fuera la causa de la entropía baja.

## 2. Único cambio propuesto y controles fijos

La implementación actual actualiza una media móvil de carga y agrega al sesgo una corrección proporcional a `router_bias_step`, limitada por `router_bias_limit`. [3] D1A observó valores absolutos de sesgo de hasta aproximadamente 0,43 bajo un límite de 0,50. [1] Esto justifica investigar un ajuste más lento, pero no permite concluir que exista saturación dañina.

| Elemento | D1A observado | D1B propuesto | Regla |
|---|---:|---:|---|
| `router_bias_step` | 0,05 | **0,01** | Único cambio de configuración intencional. |
| `router_bias_limit` | 0,50 | 0,50 | Debe permanecer fijo. |
| Umbral de entropía | 0,50 | 0,50 | Se informa sin redefinir el criterio. |
| Umbral de desequilibrio | 0,30 | 0,30 | Se informa sin redefinir el criterio. |
| Pasos / batch / secuencia | 768 / 2 / 1024 | 768 / 2 / 1024 | Misma ventana y 1.572.864 tokens esperados. |
| Topología base | dim 512, 4 capas, 8 expertos, top-2 | Sin cambios | Cualquier diferencia exige nuevo protocolo. |
| Inicialización y reanudación | Nueva / no reanudada | Nueva / no reanudada | Prohibido cargar E0, D1A o cualquier D1B previo. |
| Datos | Sólo shards `train` congelados | Sólo shards `train` congelados | Holdout no se abre ni se usa para seleccionar. |
| Runtime | Fallback PyTorch experimental | Igual si la precondición se mantiene | Triton estricto continúa bloqueado. |

La semilla, el orden de datos, el tipo de precisión y los demás argumentos del lanzamiento deberán fijarse explícitamente en una futura implementación y compararse con el registro D1A. Si alguno no puede mantenerse o documentarse, D1B deja de ser una comparación controlada y debe volver a planificación.

## 3. Datos permitidos, telemetría y análisis

El único insumo de datos de una eventual D1B serán los shards `train` del Dataset privado congelado. No se abrirá contenido holdout EN/ES, no se alterará el Dataset y no habrá selección, parada anticipada ni cambios de hiperparámetros guiados por holdout. [1] D1B tampoco utilizará pesos, outputs o checkpoints de D1A como entrada.

La telemetría permitida es exclusivamente `metrics_rank_0.jsonl` de la nueva salida D1B. El resumidor ya rechaza todo nombre distinto de `metrics_rank_0.jsonl`, no carga checkpoints y produce pérdidas descriptivas, salud del router y estadísticas por capa. [4] Tras la autorización B1, la adaptación local explícita acepta únicamente `D1A` o `D1B` como identidad y emite para D1B el esquema `aethel-d1b-router-diagnostic/v1` y el estado `D1B_METRICS_SUMMARIZED`. Esa preparación quedó incluida en el release privado de código D1B, sin autorizar ninguna ejecución.

| Resultado a reportar | Uso permitido | Uso prohibido |
|---|---|---|
| Pasos y tokens finales | Verificar que la ventana se completó | Declarar calidad de modelo. |
| Pérdida mínimo/media/máximo | Contexto descriptivo de estabilidad | Elegir candidato o comparar con holdout. |
| Pasos saludables/no saludables | Contrastar la hipótesis D1B frente a la línea D1A | Declarar router “resuelto” automáticamente. |
| Entropía, desequilibrio, carga máxima y sesgo por capa | Describir el efecto de la única modificación | Atribuir causalidad concluyente. |
| Flags de límites | Confirmar ausencia de checkpoint de entrada, holdout y red | Inferir persistencia o validez de outputs. |

## 4. Decisiones documentales permitidas después de D1B

La comparación se hará contra el resumen fijo de D1A, no contra archivos de salida D1A. Una revisión posterior sólo podrá clasificar la evidencia; no activa otra corrida.

| Clasificación posterior | Condición descriptiva | Consecuencia permitida |
|---|---|---|
| `D1B_INVALID_OR_INCOMPLETE` | Falla preflight, telemetría incompleta, límites incumplidos o ventana diferente | Bloquear D2/D3; corregir sólo mediante un nuevo plan. |
| `D1B_ROUTER_NOT_IMPROVED` | La proporción saludable no aumenta, o el desequilibrio supera 0,30 | Bloquear D2/D3; no iterar parámetros automáticamente. |
| `D1B_ROUTER_SIGNAL_REVIEWABLE` | La proporción saludable aumenta y el desequilibrio no supera 0,30 | Preparar una discusión documental sobre el siguiente diagnóstico; sin ejecutar D2 ni promoción. |

Incluso la tercera clasificación no valida un modelo, no autoriza correr D2/D3 y no cambia que el candidato de modelo sólo podría considerarse tras puertas independientes de diseño, datos, reproducibilidad, coste, seguridad y una evaluación final holdout aislada.

## 4.2 Resultado D1B observado

La salida final compartida por el usuario informó `D1B_METRICS_SUMMARIZED` y `D1B_DIAGNOSTIC_COMPLETE` tras 768 pasos y 1.572.864 tokens. La pérdida mínima/media/máxima fue 7,667897 / 9,273529 / 10,441325. El router reportó 44 pasos saludables y 724 no saludables, equivalentes a 5,729166 % y 94,270833 %, respectivamente. La entropía mínima observada fue 0,333333 y el desequilibrio máximo 0,187500. [5]

El resultado se clasifica como **`D1B_ROUTER_NOT_IMPROVED`**: los 44 pasos saludables no superan los 78 pasos saludables de D1A y el router permanece no saludable. Por ello, la modificación única `router_bias_step: 0.05 → 0.01` no satisface la hipótesis operativa de D1B. Esta comparación es diagnóstica y no atribuye causalidad concluyente. Continúan bloqueados D2, D3, holdout, selección de candidato, promoción y serving.

La revisión comparativa independiente está registrada en [`AETHEL_D1A_D1B_ROUTER_EVIDENCE_REVIEW_2026-08-23.md`](AETHEL_D1A_D1B_ROUTER_EVIDENCE_REVIEW_2026-08-23.md). Descarta para esta configuración y ventana la hipótesis de que una corrección de sesgo más lenta por sí sola resuelva la salud del router; no atribuye causalidad, no selecciona una alternativa y no autoriza ninguna ejecución posterior.

La misma salida reafirmó `checkpoint_loaded=false`, `raw_corpus_read=false`, `holdout_content_read=false`, `network_requests=0` y `promotion_authorized=false`. No se inspeccionaron, cargaron, movieron, descargaron ni reanudaron outputs o checkpoints D1B.

## 4.1 Preparación local registrada

Después de una autorización B1 específica, se añadieron localmente el lanzador bloqueado `run_kaggle_d1b_router_bias_diagnostic.sh` y `test_d1b_local_contracts.py`. El lanzador exige el release exacto `d1b-v1-router-bias-step-001-train-only`, tres autorizaciones D1B explícitas, ausencia de `AETHEL_RESUME_CHECKPOINT`, salida inexistente y el único cambio `--router-bias-step 0.01`; conserva el límite 0,50, la ventana de 768 pasos, el preflight *train-only* y el rechazo de holdout. Las pruebas D1A/D1B locales, las dos sintaxis Bash, el control de diff, Vitest y TypeScript pasaron.

Tras una autorización B3 separada, se actualizó `aethel_kaggle_source_release.json` y se construyó el bundle de código privado D1B, SHA-256 `488990206ad61eaa9098cc68e32b0c9c0bbce197724415401c616ae0102ce0c0`. La captura aportada por el usuario mostró **“Your dataset version was created successfully”** después de la subida al Dataset privado de código. La interfaz no mostró número de versión ni ruta de montaje, por lo que esos datos no se afirman. No se modificó `aethel-nextgen-data-v1`, no se creó ni ejecutó notebook, y no hubo GPU ni manejo de outputs/checkpoints.

Después de una autorización B4 para preparación, se creó y validó localmente `training/AETHEL_D1B_ROUTER_BIAS_BLOCKED_CELL.py`, con copia manual en `/home/ubuntu/aethel-private-transfer/AETHEL_D1B_ROUTER_BIAS_BLOCKED_CELL.py`. La celda exige un único release D1B, imprime `D1B_CELL_PREPARED_NOT_EXECUTED` y contiene de forma intencional **ninguna rama de ejecución**: no importa `torch`, `subprocess` o `shutil`; no define ruta de datos ni variables de autorización; no copia código ni selecciona GPU. La prueba estática `test_d1b_blocked_cell.py` pasó y la celda rechaza localmente un montaje inexistente antes de cualquier acceso a datos.

Con una confirmación adicional, el usuario añadió y ejecutó únicamente esta celda en Kaggle. El registro observado seleccionó `SOURCE_INPUT_D1B` bajo el montaje `(11)`, confirmó `SOURCE_RELEASE: d1b-v1-router-bias-step-001-train-only`, emitió `D1B_CELL_PREPARED_NOT_EXECUTED` y declaró `D1B_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT`. El mismo log afirma que no se seleccionó GPU, no se copió código, no se leyó train/holdout, no se cargaron pesos y no se entrenó. Esto acredita exclusivamente preparación bloqueada del notebook; B5–B6 continúan cerradas.

Tras una autorización B5a específica, se creó y validó sólo localmente `training/AETHEL_D1B_ROUTER_BIAS_EXECUTION_CELL.py`, con copia manual en `/home/ubuntu/aethel-private-transfer/AETHEL_D1B_ROUTER_BIAS_EXECUTION_CELL.py` y prueba `test_d1b_execution_cell.py`. Esta variante habilitable inicia cerrada y exige cinco valores simultáneos: ejecución D1B, corrida D1B, GPU, token final y fallback PyTorch experimental. Mientras una puerta sea distinta, imprime `D1B_EXECUTION_PENDING_FINAL_AUTHORIZATION` antes de resolver el Dataset de datos, copiar código, crear salida, seleccionar GPU o invocar el lanzador. La prueba local confirmó el orden de esa barrera, la ausencia de reanudación y el único valor de sesgo `0.01`.

Tras una autorización B5b separada, el usuario reemplazó la celda del notebook y ejecutó únicamente su comprobación con las cinco puertas cerradas. El log seleccionó el mismo montaje privado `(11)`, confirmó `SOURCE_RELEASE: d1b-v1-router-bias-step-001-train-only`, emitió `D1B_EXECUTION_PENDING_FINAL_AUTHORIZATION` y reafirmó que no se seleccionó GPU, no se copió código, no se leyó Dataset train/holdout, no se cargaron pesos y no se entrenó. Esto acredita sólo que el bloqueo de la variante funciona en el notebook; B5c–B6 continúan cerradas.

## 5. Puertas antes de cualquier D1B

Este protocolo no concede permisos entre etapas. Las autorizaciones deben ser específicas, inmediatas y separadas.

| Puerta | Requisito | Todavía prohibido |
|---|---|---|
| B0 — planificación | Este documento versionado y revisado | Todo cambio de código, Dataset, Kaggle o GPU. |
| B1 — implementación local | **Completada** tras autorización separada: lanzador, identidad de resumen y contratos D1B locales. | Subir código, abrir notebook, seleccionar GPU o ejecutar. |
| B2 — validación local | **Completada**: pruebas de contrato, sintaxis, diff, Vitest y TypeScript, sin datos/pesos. | Cambiar Dataset privado o usar Kaggle. |
| B3 — fuente privada | **Completada**: release D1B privado creado con éxito; el número de versión/montaje no fue mostrado. | Alterar `aethel-nextgen-data-v1` o ejecutar notebook. |
| B4 — notebook | **Completada**: celda añadida y ejecutada sólo en modo bloqueado, con release montado `(11)` y `D1B_CELL_PREPARED_NOT_EXECUTED`. | GPU, `Save & Run All`, guardar versión o ejecutar. |
| B5a — variante local | **Completada**: variante habilitable local con cinco puertas todavía cerradas. | Reemplazar celda, GPU, Kaggle o ejecutar D1B. |
| B5b — reemplazo verificado | **Completada**: variante añadida y ejecutada sólo con sus cinco puertas cerradas; emitió `D1B_EXECUTION_PENDING_FINAL_AUTHORIZATION`. | GPU, copia de código, Dataset o entrenamiento. |
| B5c — GPU y ejecución | **Completada**: GPU T4 ×2 comprobada y D1B ejecutada una vez hasta `D1B_DIAGNOSTIC_COMPLETE`. | Reanudación, pesos de entrada, holdout, D2/D3, promoción y serving. |
| B6 — preservación | Confirmación separada para cualquier Save Version posterior | Inspeccionar, descargar, mover, cargar o abrir checkpoints. |

Si la pantalla cambia, falta un montaje, falla el preflight, se detecta un output existente o una autorización es ambigua, se detiene antes de copiar código o usar GPU. No hay permiso para `Save & Run All`, repetición, modificación del Dataset de datos, descarga de artefactos o revisión de `.pt` en ninguna puerta de este documento.

## 5.1 Convención operativa de celdas

La plantilla habilitable se entregó como **CELDA 5** completa, con las cinco puertas configuradas conforme a B5c-1 y un encabezado visible que declara propósito y estado. La regla para futuras celdas queda definida en [`AETHEL_NOTEBOOK_CELL_CONVENTION_2026-08-23.md`](AETHEL_NOTEBOOK_CELL_CONVENTION_2026-08-23.md): cada celda nueva debe declarar dentro del código `CELDA <número> — <propósito>` y su estado operativo.

La interfaz compartida mostró **GPU T4 ×2** seleccionada y la Celda 5 se ejecutó una vez tras la autorización final. Su salida final quedó clasificada en la sección 4.2. Siguen prohibidos checkpoints de entrada, holdout, reanudación, D2/D3, promoción y serving.

## 6. Relación con el primer modelo

D1B fue otra prueba diagnóstica, no el primer modelo, y no superó su puerta de router. No existe autorización para D2/D3. Cualquier futura propuesta deberá partir de un nuevo plan basado en esta evidencia, mantener el Dataset congelado y el holdout sellado, y obtener confirmaciones separadas de código, fuente, notebook, GPU y ejecución. La evaluación holdout nunca elegirá variantes de D1B/D2/D3.

## Referencias

[1]: ./d1a_v1_router_diagnostic_evidence.json "Resumen seguro de evidencia D1A"
[2]: ../engine/train_aethel_gpu.py "Regla de salud del router en el entrenador GPU"
[3]: ../engine/aethel_model.py "Ajuste de media móvil y sesgo del router MoE"
[4]: ./summarize_d1a_router_metrics.py "Resumidor de métricas que no carga checkpoints"
[5]: ./d1b_v1_router_diagnostic_evidence.json "Resumen seguro de evidencia D1B"
