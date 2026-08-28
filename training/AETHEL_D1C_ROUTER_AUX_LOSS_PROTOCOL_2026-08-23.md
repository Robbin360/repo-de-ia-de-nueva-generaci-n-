# D1C — Diagnóstico controlado del peso auxiliar MoE

## Estado y frontera

> **Estado: `D1C_SUMMARY_BLOCKED_AFTER_AUTHORIZED_RUN`.** La interfaz aportada por el usuario confirmó que la **Version 13** privada del Dataset de código se creó correctamente. La CELDA 6 bloqueada resolvió de forma única el release `d1c-v1-router-aux-loss-005-train-only` desde el montaje D1C y emitió `D1C_CELL_PREPARED_NOT_EXECUTED` y `D1C_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT`. Después, la CELDA 7 habilitable se pegó sólo para comprobar sus cinco puertas cerradas y emitió `D1C_EXECUTION_PENDING_FINAL_AUTHORIZATION`; tras confirmaciones separadas de edición, GPU y corrida exacta, se ejecutó una vez. El registro aportado muestra que la fase final intentó resumir como `D1C`, pero la Version 13 contenía un contrato CLI obsoleto que aceptaba sólo `D1A` y `D1B`; por ello faltan `D1C_METRICS_SUMMARIZED` y `D1C_DIAGNOSTIC_COMPLETE`. No se modificó el Dataset privado `aethel-nextgen-data-v1`, ni se abrió, descargó, movió, cargó o inspeccionó output o checkpoint alguno.

D1C no es un candidato, un entrenamiento de producto, una evaluación, una promoción ni una autorización para D2, D3 o serving. Mientras falte el resumen seguro, no se clasifica la hipótesis ni se compara D1C con D1A/D1B. La ruta Triton estricta continúa bloqueada; el intento autorizado usó únicamente el fallback PyTorch experimental.

## Hipótesis única

Los diagnósticos D1A y D1B compartieron el peso auxiliar implícito `0.01`; D1B sólo varió `router_bias_step` y no mejoró el router. La prueba local A3 confirma el signo esperado de la señal auxiliar cuando la densidad se concentra, pero no su escala eficaz. Por ello, D1C formula una única hipótesis falsable:

> Manteniendo el baseline D1A, aumentar sólo `router_aux_loss_weight` de **0.01** a **0.05** aumentará materialmente los pasos saludables del router sin deteriorar de forma material la pérdida media.

| Control | D1A de referencia | D1C preparado |
|---|---:|---:|
| Inicialización | Nueva | Nueva; sin reanudar |
| Datos | Sólo train | Sólo train; holdout sellado |
| Pasos / tokens | 768 / 1.572.864 | 768 / 1.572.864 |
| Seed / ventana | 17 / misma ventana | 17 / misma ventana |
| Arquitectura | 512, 4 capas, GQA 8/2, 8 expertos top-2 | Idéntica |
| `router_bias_step` / límite | 0.05 / 0.5 | 0.05 / 0.5 |
| `router_aux_loss_weight` | 0.01 implícito | **0.05 explícito** |
| Runtime | fallback PyTorch experimental | Igual, si se autoriza |

## Criterios predefinidos

El resultado sólo se consideraría **apoyado para revisión documental posterior** si se cumplen todos los criterios siguientes: al menos **117/768** pasos saludables (15,234375 %, frente a 78/768 o 10,156250 % en D1A), entropía mínima estrictamente superior a 0,333333, desequilibrio máximo no superior a 0,187500 y pérdida media no superior a 9,35257273 (un máximo de 1 % sobre la media D1A). No convierte el resultado en modelo listo ni candidato promovible.

Si un criterio falla, D1C se clasificaría `D1C_ROUTER_NOT_IMPROVED`; no habrá D1D automática ni repetición. Cualquier análisis posterior permanecería documental hasta una nueva autorización.

## Puertas que siguen cerradas

El lanzador local `run_kaggle_d1c_router_aux_loss_diagnostic.sh` exige el release exacto `d1c-v1-router-aux-loss-005-train-only`, salida nueva, ausencia de reanudación y tres autorizaciones de ejecución/GPU/fallback. Estas protecciones no son autorización. La creación privada de la Version 13, la CELDA 6 bloqueada, la comprobación cerrada de la CELDA 7 y su reemplazo posterior sin ejecución fueron confirmados por el usuario. Seleccionar o usar GPU y ejecutar la corrida requerirán confirmaciones posteriores, inmediatas y distintas. Save Version, outputs/checkpoints, holdout, promoción y serving requerirán sus propias confirmaciones.

La **CELDA 6** bloqueada está preparada localmente en `AETHEL_D1C_ROUTER_AUX_LOSS_BLOCKED_CELL.py`. Sólo localiza el release D1C exacto y emite `D1C_CELL_PREPARED_NOT_EXECUTED`; no incluye una rama de ejecución ni puede seleccionar GPU, copiar código, abrir Dataset, cargar pesos o entrenar. Su preparación local no autoriza incorporarla al notebook.

La **CELDA 7** habilitable se conserva localmente en `AETHEL_D1C_ROUTER_AUX_LOSS_EXECUTION_CELL.py`. El usuario la añadió manualmente al notebook y ejecutó una vez sólo con sus cinco puertas cerradas; resolvió el release exacto y emitió `D1C_EXECUTION_PENDING_FINAL_AUTHORIZATION` antes de copiar código, tocar Dataset, seleccionar GPU o invocar el lanzador. Tras autorización distinta para editarla, el usuario reemplazó su contenido por `AETHEL_CELDA_7_D1C_APROBADA.py`, con cinco puertas abiertas, y la ejecutó una vez bajo el alcance autorizado. La plantilla no forma parte de la Version 13. El fallo ocurrió después del intento de entrenamiento, en la invocación final del resumidor; no autoriza reanudación, repetición ni acceso a artefactos.

## Bloqueo de resumen y corrección local

El origen del bloqueo está aislado: `summarize_d1a_router_metrics.py` ya aceptaba `D1C` dentro de su función `summarize`, pero el argumento CLI `--diagnostic-id` aún limitaba sus elecciones a `D1A` y `D1B`. La corrección local incorpora `D1C` en esa lista y una prueba estática que evita esta divergencia futura. No cambia la arquitectura, los datos, el optimizador, la ventana, el peso auxiliar ni la lógica de entrenamiento.

Esta corrección **no** abre outputs, checkpoints, corpus ni holdout; tampoco convierte el intento en una ejecución satisfactoria. Antes de cualquier actualización privada del Dataset de código, reanudación o nueva corrida debe existir una autorización inmediata y específica. La única salida admisible para clasificar un futuro intento será el resumen seguro ya definido; no se usarán artefactos del intento bloqueado como sustituto.

El release local siguiente es `d1c-v2-summary-cli-fix-train-only`, acompañado de la **CELDA 8** bloqueada `AETHEL_D1C_V2_SUMMARY_FIX_BLOCKED_CELL.py`. El bundle local validado tiene SHA-256 `a659a615fbdfb7e48fcce0a5b7442fc07e6dd2a5516254442dfcc1a5a97aad07` y excluye explícitamente corpus, pesos, checkpoints, métricas crudas y bytecode. El usuario confirmó la creación manual de una nueva versión privada del Dataset de código con ese ZIP; el número de versión no fue compartido visualmente. Después, la CELDA 8 resolvió el release V2 exacto y emitió `D1C_V2_CELL_PREPARED_NOT_EXECUTED` junto a la ausencia intencional de rama de ejecución. V2 sólo transporta la corrección de aceptación CLI y una prueba de no regresión. La verificación bloqueada no autoriza GPU, retry, reanudar el intento V1, leer Dataset train/holdout, cargar pesos ni abrir/mover outputs o checkpoints.

Como preparación exclusivamente local posterior se añadió `AETHEL_D1C_V2_RETRY_EXECUTION_CELL.py` para la **CELDA 9**. Conserva cinco puertas cerradas y exige fuente V2 única, inicio nuevo, sólo train, directorio de trabajo/salida nuevos y eliminación explícita de cualquier `AETHEL_RESUME_CHECKPOINT` antes de su rama inaccesible. No está incorporada a la versión privada V2, no se ha entregado al notebook y no autoriza un retry ni acceso a datos o artefactos. Cualquier eventual retry debe definirse y aprobarse de nuevo como acción distinta.

El marcador local actual es `d1c-v3-retry-cell-train-only`. Su bundle de transferencia local `aethel-nextgen-source-d1c-v3-retry-cell-train-only.zip` tiene SHA-256 `7028a42ac0246ae1b455e0c7036f5e865b5fe6b9c16331867a3ce40dc0377f06`; contiene la corrección CLI V2 y la plantilla de retry con su prueba estática, y excluye corpus, pesos, checkpoints, métricas JSONL crudas y bytecode. El usuario aportó una captura que muestra el directorio V3 dentro del Dataset privado de código y confirmó que pegó manualmente una **CELDA 9 V3** que selecciona este release y conserva las cinco puertas cerradas. La primera comprobación bloqueada autorizada devolvió `candidatos: ninguno` porque el input V3 del notebook no estaba actualizado; se detuvo antes de GPU, retry, lectura de Dataset train/holdout, pesos, outputs o checkpoints. Tras actualizar el input del notebook, la misma CELDA 9 resolvió el release exacto y emitió `D1C_V3_CELL_PREPARED_NOT_EXECUTED` y `D1C_V3_RETRY_PENDING_FINAL_AUTHORIZATION`. No fue necesario reemplazarla ni hubo GPU, retry, datos, pesos, outputs o checkpoints.

La preparación local posterior se rige por [`AETHEL_D1C_V3_RETRY_DECISION_PROTOCOL_2026-08-23.md`](AETHEL_D1C_V3_RETRY_DECISION_PROTOCOL_2026-08-23.md). Define D1C V3-R1 sólo como un experimento nuevo, con inicialización nueva, train-only, salida inédita y autorizaciones separadas; no abre ninguna puerta de ejecución.

El release local siguiente es `d1c-v4-v3-r1-launcher-profile-train-only`. Su empaquetador exclusivo V4 validó los archivos de transferencia `aethel-nextgen-source-d1c-v4-v3-r1-launcher-profile-train-only.tar.gz` (SHA-256 `7905caff0c40552b0ae6780f5991827f0106cb34b6dafa1bd51f9508db061c51`) y `.zip` (SHA-256 `08d51374a9684340d7ffe47d48a2f9edf6eb36b0bb123b72ae56bd0f397c043a`), ambos sin corpus, JSONL, pesos, checkpoints, métricas crudas ni bytecode. V4 incorpora el perfil de release V3-R1 y la plantilla local de **CELDA 10** con seis puertas cerradas. El usuario aportó una captura de **Version 16 — complete** con estado **Success** para su versión privada de código. V4 no se ha adjuntado al notebook ni se ha añadido o ejecutado CELDA 10; no hubo GPU, Dataset, retry, outputs/checkpoints, holdout, promoción ni serving.

## Cierre D1C V3-R1 — resultado clasificable

La única corrida D1C V3-R1 autorizada terminó y el resumidor emitió `D1C_DIAGNOSTIC_COMPLETE`. El cierre compartido por el usuario indica 768 pasos, 1.572.864 tokens, inicialización sin checkpoint cargado, `holdout_content_read=false`, `network_requests=0`, `router_aux_loss_weight=0.05`, `require_triton=false` y runtime CUDA/PyTorch experimental.

La clasificación determinista es **`D1C_ROUTER_NOT_IMPROVED`**. Los resultados fueron 67/768 pasos saludables (8,72 %) frente al mínimo 117/768 (15,234375 %), entropía mínima 0,3333333433 frente al requisito estricto `> 0,333333`, desequilibrio máximo 0,1875 frente al límite `<= 0,1875`, y pérdida media 9,43690848 frente al máximo 9,35257273. Sólo el desequilibrio máximo cumplió; fallaron los otros tres criterios.

Esto valida que el flujo train-only y el resumen corregido funcionaron, pero **no valida la hipótesis auxiliar, no produce un modelo funcional promovible y no autoriza D1D**. No se abrieron, descargaron, movieron ni deserializaron outputs o checkpoints. Holdout, promoción, serving y cualquier nuevo entrenamiento permanecen bloqueados hasta una decisión y autorización específicas.
