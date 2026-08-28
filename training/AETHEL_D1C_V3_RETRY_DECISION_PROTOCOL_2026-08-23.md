# D1C V3-R1 — Protocolo local de decisión para un retry aislado

> **Estado: `D1C_V3_R1_PREPARATION_ONLY`.** Este documento prepara una decisión; no autoriza editar un notebook, seleccionar GPU, leer Dataset, entrenar, guardar una versión, abrir outputs/checkpoints, evaluar holdout, promover ni servir un modelo.

## Por qué existe este protocolo

El único intento D1C V1 se ejecutó una vez desde inicialización nueva y sólo con train, pero quedó **no clasificable** porque el resumidor seguro rechazó el identificador CLI `D1C` al cierre. Sus outputs y checkpoints continúan protegidos y no se usarán para reconstruir métricas. La corrección V2 del resumidor fue verificada únicamente en modo bloqueado. La CELDA 9 V3 confirmó después que el release exacto está montado, pero no leyó datos ni ejecutó un retry.

Por tanto, un eventual **D1C V3-R1** sería un experimento nuevo y aislado, no una reanudación, reparación retrospectiva ni interpretación del intento V1. Su único propósito sería obtener, si se autorizara y completara, un resumen seguro generado por un flujo de train-only desde una inicialización nueva.

## Invariantes no negociables

| Área | Regla para D1C V3-R1 |
|---|---|
| Identidad | Etiqueta operativa de retry `D1C-V3-R1`; el CLI de resumen permanece `D1C` sólo porque el contrato corregido admite ese identificador. |
| Código | Release exacto `d1c-v3-retry-cell-train-only`; no se reutiliza la Version 13 V1 ni el release V2 como fuente de ejecución. |
| Inicialización | Nueva, sin pesos E0, sin checkpoint D1C V1 y con `AETHEL_RESUME_CHECKPOINT` ausente. |
| Datos | Sólo particiones train; holdout EN/ES permanece sellado y fuera de tokenización, selección y evaluación. |
| Configuración | 768 pasos, seed 17, misma ventana/topología D1A, `router_bias_step=0.05`, `router_aux_loss_weight=0.05`. |
| Runtime | Fallback PyTorch experimental explícitamente autorizado; Triton estricto sigue bloqueado. |
| Directorios | Raíz de trabajo y salida nuevas, inexistentes y exclusivas de `D1C-V3-R1`; ningún borrado o reutilización. |
| Interpretación | Sólo el resumen seguro integrado podría producir una clasificación; métricas crudas, outputs y checkpoints permanecen inaccesibles salvo autorización nueva y específica. |

## Criterio de decisión predefinido

La hipótesis sigue siendo idéntica a D1C: frente a D1A, subir exclusivamente `router_aux_loss_weight` de 0.01 a 0.05 podría aumentar los pasos saludables sin deteriorar materialmente la pérdida media. La definición de apoyo documental requeriría simultáneamente los cuatro límites ya fijados: al menos 117/768 pasos saludables, entropía mínima mayor que 0.333333, desequilibrio máximo no mayor que 0.187500 y pérdida media no mayor que 9.35257273.

El incumplimiento de cualquier límite se clasificaría como `D1C_ROUTER_NOT_IMPROVED`. Aun si todos se cumplen, el resultado sólo habilitaría revisión documental; no sería modelo funcional, benchmark, promoción, serving, D2 ni D3.

## Secuencia de puertas separadas

| Puerta | Acción limitada | Requisito de autorización | Lo que sigue prohibido |
|---|---|---|---|
| G0 | Revisar este protocolo local | Confirmación de alcance local ya dada | Notebook, GPU, Dataset y ejecución. |
| G1 | Reemplazar manualmente CELDA 9 por una plantilla V3-R1 habilitable pero cerrada | Autorización explícita de **edición de notebook** | Ejecutar la celda, GPU y retry. |
| G2 | Verificar CELDA 9 V3-R1 con puertas cerradas | Autorización explícita de **una comprobación bloqueada** | Copiar código, leer train/holdout, GPU y retry. |
| G3 | Abrir la puerta de GPU y confirmar su tipo disponible | Autorización explícita de **selección/uso de GPU** | Iniciar entrenamiento y tocar artefactos. |
| G4 | Ejecutar una vez el retry exacto de 768 pasos | Autorización explícita de **corrida exacta**, inicialización nueva y fallback PyTorch | Reanudar, usar holdout, abrir artefactos o lanzar D2/D3. |
| G5 | Guardar una versión privada si el usuario lo decide | Autorización explícita de **Save Version** | Descargar, abrir, mover, promover o servir sus artefactos. |
| G6 | Revisar un resumen seguro compartido por el usuario | Autorización explícita de **revisión de resumen** | Inspeccionar JSONL crudo, outputs o checkpoints. |

Si una puerta no recibe su autorización literal e inmediata, la plantilla debe detenerse antes de la siguiente acción. Ninguna puerta autoriza por implicación las demás.

## Contrato de salida y fallos cerrados

Antes de G4, la celda debe emitir únicamente un estado de preparación bloqueada; no puede invocar `subprocess`, copiar el release ni resolver el Dataset de datos. G4 debe fallar cerrado si la salida ya existe, falta cualquiera de las cinco confirmaciones, aparece un checkpoint de reanudación, el release no es V3 o se detecta una ruta de holdout.

Tras una eventual corrida autorizada, la única evidencia admisible de forma predeterminada sería el resumen seguro producido dentro del flujo exacto y compartido voluntariamente por el usuario. Una excepción para abrir, mover, descargar, listar, deserializar, eliminar o reutilizar outputs/checkpoints requiere una autorización nueva y específica; este documento no la concede.

## Compatibilidad de release pendiente

El lanzador histórico `run_kaggle_d1c_router_aux_loss_diagnostic.sh` conserva, correctamente para D1C V1, el release exacto `d1c-v1-router-aux-loss-005-train-only`. Por ello no puede ejecutar un futuro V3-R1 sin una modificación de contrato: fallaría cerrado al comparar el marcador V3, incluso si todas las otras puertas estuvieran abiertas. La corrección local añade un **perfil de ejecución de release explícito**: V1 permanece como valor predeterminado; V3-R1 sólo podrá seleccionar `d1c-v4-v3-r1-launcher-profile-train-only` mediante `AETHEL_D1C_EXPECTED_RELEASE` y una autorización adicional `AETHEL_D1C_RELEASE_PROFILE_AUTHORIZED=YES`. Los perfiles no permitidos fallan cerrados. Esta corrección no se ha cargado a Kaggle, no cambia los hiperparámetros de D1C y no convierte V3-R1 en ejecutable.

## Estado actual

**No hay retry preparado para ejecutar.** Sólo están permitidos los contratos locales posteriores que conviertan estas reglas en una plantilla estática con las puertas cerradas. La CELDA 9 V3 existente ha validado el montaje y sigue bloqueada; no puede convertirse por sí sola en la ejecución V3-R1.

El archivo local [`aethel_d1c_v3_r1_authorization_contract.json`](aethel_d1c_v3_r1_authorization_contract.json) materializa estas puertas con todos sus valores en `false`; su prueba estática no abre ni verifica recursos externos.

La plantilla local [`AETHEL_D1C_V3_R1_RETRY_EXECUTION_CELL.py`](AETHEL_D1C_V3_R1_RETRY_EXECUTION_CELL.py) está destinada a una futura **CELDA 10** y conserva seis puertas cerradas: las cinco de ejecución más una específica para el perfil de release. Busca un release V4 futuro de forma anclada, devuelve la raíz del repositorio (no sólo `training/`) y no resuelve el Dataset de datos, copia archivos ni invoca el lanzador mientras una puerta permanezca cerrada.
