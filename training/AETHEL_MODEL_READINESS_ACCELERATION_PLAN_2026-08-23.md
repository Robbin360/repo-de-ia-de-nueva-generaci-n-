# Ruta de aceleración hacia un primer modelo Aethel verificable

**Estado:** `LOCAL_READINESS_PLAN_ONLY`  
**Fecha:** 23 de agosto de 2026  
**Autor:** Manus AI  
**Alcance:** ordenar el trabajo local y las autorizaciones futuras; no abre outputs, checkpoints, corpus, shards ni holdout, y no modifica Kaggle, el Dataset, notebook, GPU ni serving.

## Objetivo verificable

La meta inmediata no es declarar un producto, Edge o Pro. Es llegar a un **primer prototipo Aethel verificable**: una configuración candidata fijada antes de entrenar, una corrida reproducible desde inicialización nueva, una restauración reproducible, una generación token a token comprobada y una evaluación aislada EN/ES realizada sólo después de fijar el candidato. El prototipo no pasa a serving ni adquiere condición comercial por cumplir esos pasos.

> **Principio de aceleración:** se reduce trabajo perdido eliminando cambios de hiperparámetros sin hipótesis y preparando por adelantado las verificaciones locales, el release exacto y las autorizaciones separadas. No se acelera saltando Dataset, holdout, router, reproducibilidad o controles de artefactos.

## Punto de partida comprobado

E0 V8 acreditó que la cadena experimental pudo completar 4.992 pasos y producir evidencia aislada, pero su router final fue no saludable; no acredita un modelo funcional para serving. [1] D1A y D1B compararon una sola modificación `router_bias_step` en ventanas *train-only* equivalentes de 768 pasos desde inicialización nueva. D1A registró 78 pasos saludables; reducir el paso de 0,05 a 0,01 produjo 44, por lo que ese cambio no es una mejora seleccionable. [2] [3]

| Estado necesario | Situación actual | Bloqueo que impide avanzar | Evidencia requerida para cerrarlo |
|---|---|---|---|
| Router apto para candidato | No demostrado | D1A y D1B permanecen mayoritariamente no saludables. | Una hipótesis distinta, pre-registrada y evaluada sólo con train. |
| Configuración candidata fija | No existe | No hay intervención de router apoyada por evidencia. | Decisión documental basada en contrato y pruebas locales. |
| Corrida candidata reproducible | No autorizada | Depende de una configuración fijada y de permisos de fuente, notebook y GPU. | Release exacto, preflight, smoke CUDA y confirmaciones inmediatas. |
| Prototipo funcional verificable | No demostrado | No hay candidato fijado ni generación/restauración autorizadas. | Corrida candidata completa, restauración y generación comprobadas; evaluación holdout aislada posterior. |
| Producto o serving | Fuera de alcance | No existen puertas de seguridad, calidad ni operación cerradas. | Trabajo independiente posterior; no se infiere del prototipo. |

## Secuencia que maximiza información por acción

La ruta rápida prioriza evidencia que descarta causas antes de consumir una nueva sesión GPU. Cada etapa termina con una decisión binaria; si no se cumple, no se abre la siguiente.

| Etapa | Trabajo permitido | Resultado de salida | Prohibición que permanece |
|---|---|---|---|
| A1 — contrato de medición | Revisar y ampliar en local pruebas deterministas del cálculo de entropía, desequilibrio, límites y clasificación `healthy`; usar únicamente tensores sintéticos de prueba. | El criterio de salud queda inequívoco y cubierto por tests. | No se abren salidas D1A/D1B ni se usa Dataset. |
| A2 — hipótesis de mecanismo | Formular una sola explicación técnica nueva basada en el contrato del router y declarar predicción, variable única, controles y rechazo antes de escribir un lanzamiento. | Protocolo documental listo o conclusión de que falta evidencia. | No se inicia otro diagnóstico ni se modifica código de Kaggle. |
| A3 — preparación reproducible | Si A2 produce una hipótesis defendible, implementar en local sus contratos, pruebas, release seleccionable y celda inicialmente bloqueada. | Preflight local y barreras de autorización correctas. | No se actualizan Dataset, notebook, GPU ni artefactos. |
| A4 — diagnóstico aislado | Sólo con aprobaciones inmediatas separadas: actualizar código, comprobar notebook, seleccionar GPU y ejecutar una vez con sólo train. | Evidencia resumida segura que acepte o descarte la hipótesis. | Holdout, reanudación, promoción y serving continúan cerrados. |
| A5 — candidato fijo | Únicamente si el diagnóstico satisface criterios predefinidos, documentar una configuración candidata sin modificarla después. | Identidad de candidato y plan de corrida completos. | No se evalúa holdout para elegirlo. |
| A6 — prototipo verificable | Con nuevas aprobaciones: corrida candidata, restauración/generación y una evaluación EN/ES única y aislada después de fijar el candidato. | Informe que separe capacidad experimental, límites y artefactos. | No se habilita serving, Edge, Pro ni promoción automática. |

## Criterios de decisión que evitan iteración improductiva

La etapa A2 no debe seleccionar una variante porque “parece prometedora”. Debe contener una predicción que pueda fallar, exactamente una variable de configuración intencional, controles fijos, una ventana predeclarada, umbrales del router sin redefinir y una regla de parada. La hipótesis `router_bias_step=0.01` ya quedó descartada para la ventana observada y no se repetirá. [3]

Para que un diagnóstico merezca pasar de A4 a A5, deberá completar su ventana prevista, mantener las salvaguardas de límites, elevar la proporción de pasos saludables respecto de D1A **sin** superar el desequilibrio máximo de 0,30, y no requerir datos holdout. Superar esa puerta sólo permite congelar un candidato para planificación; no permite llamarlo modelo listo, ejecutar una corrida larga o habilitar una promoción.

## Autorizaciones que no se pueden comprimir

| Acción | Estado de esta ruta | Autorización necesaria |
|---|---|---|
| Documentación, tests y contratos locales sin datos protegidos | Puede prepararse localmente. | Alcance local explícito; no se traslada a Kaggle. |
| Cambiar el bundle de código privado | No concedida por este plan. | Confirmación inmediata para ese release concreto. |
| Editar o ejecutar notebook | No concedida por este plan. | Confirmación inmediata por edición y otra por ejecución. |
| Seleccionar o usar GPU | No concedida por este plan. | Confirmación inmediata de GPU y de la corrida concreta. |
| Cambiar `aethel-nextgen-data-v1` | Prohibido. | Confirmación inmediata que identifique la acción exacta. |
| Abrir, mover, descargar, cargar o reanudar outputs/checkpoints | Prohibido. | Confirmación inmediata que limite el artefacto y la operación. |
| Evaluar holdout, promover o servir | Prohibido. | Puertas independientes después de fijar un candidato. |

## A1 — Contrato local de medición del router

**A1 está completada localmente.** `engine/router_health.py` concentra la clasificación de salud usada por el entrenador y `training/test_router_health_contract.py` la cubre con telemetría sintética. El contrato verifica los umbrales inclusivos, agrega el mínimo de entropía y el máximo de desequilibrio entre capas, y rechaza de forma cerrada la ausencia, los campos incompletos, los valores no finitos y los umbrales fuera de la escala normalizada. No importa PyTorch, no abre archivos y no accede a pesos, corpus, shards, holdout, GPU, red ni Kaggle.

Este cierre reduce ambigüedad de medición; no explica todavía el origen de la baja entropía observada, no selecciona una intervención y no autoriza una nueva corrida. La posible etapa A2 sigue requiriendo una revisión documental independiente de mecanismos antes de proponer cualquier diagnóstico adicional.

## A2 — Preparación de la señal auxiliar

**A2 está completada sólo en local.** La auditoría y el contrato están en [`AETHEL_ROUTER_AUXILIARY_PREPARATION_2026-08-23.md`](AETHEL_ROUTER_AUXILIARY_PREPARATION_2026-08-23.md). El peso histórico `0.01` ahora es un parámetro explícito con el mismo valor predeterminado y prueba pura. A2 no elige un valor nuevo, no define D1C y no habilita una corrida; primero exige una hipótesis documental independiente y las autorizaciones separadas aplicables.

## A3 — Señal de balanceo verificable

**A3 está completada sólo en local.** El contrato auxiliar ahora incluye una prueba CPU de la dirección esperada del gradiente ante densidad concentrada y uniforme. Este resultado sólo valida la semántica algebraica de la pérdida implementada; no estima su efecto de entrenamiento, no selecciona un valor distinto de `0.01` y no habilita D1C, Dataset, Kaggle, GPU ni una corrida.

## D1C — Hipótesis auxiliar, intento V1 bloqueado y corrección V2 local

[`AETHEL_D1C_ROUTER_AUX_LOSS_PROTOCOL_2026-08-23.md`](AETHEL_D1C_ROUTER_AUX_LOSS_PROTOCOL_2026-08-23.md) registra la única variación D1C `router_aux_loss_weight: 0.01 → 0.05`, conservando el baseline D1A y criterios de descarte predefinidos. La interfaz compartida por el usuario confirmó la **Version 13** privada; la CELDA 6 verificó el release y la CELDA 7 verificó primero sus cinco puertas cerradas. Tras autorizaciones separadas se efectuó una única corrida V1 desde inicialización nueva y sólo train. El intento llegó al cierre, pero el resumen seguro rechazó el identificador `D1C` porque la Version 13 sólo admitía `D1A`/`D1B`; faltan `D1C_METRICS_SUMMARIZED` y `D1C_DIAGNOSTIC_COMPLETE`, por lo que no hay clasificación de hipótesis.

La corrección local incorpora `D1C` al contrato CLI del resumidor, cobertura de regresión, el marcador `d1c-v2-summary-cli-fix-train-only` y una **CELDA 8** estrictamente bloqueada. El usuario confirmó la creación manual de una nueva versión privada de código con el ZIP correctivo, sin número visual compartido. La CELDA 8 ya resolvió el release V2 exacto y confirmó `D1C_V2_CELL_PREPARED_NOT_EXECUTED`. No se abrió ni manipuló output/checkpoint, no se usó holdout, no hubo promoción ni serving. V2 no autoriza GPU, retry, reanudar V1, repetir D1C ni ejecutar una variante nueva: esas acciones requerirían plan y confirmaciones específicas posteriores.

Para reducir trabajo de preparación sin adelantar ejecución, se construyó el release `d1c-v3-retry-cell-train-only`. Incluye la plantilla de retry con cinco puertas cerradas, inicio nuevo, sólo train, rutas de trabajo/salida nuevas y bloqueo explícito de reanudación. El ZIP validado tiene SHA-256 `7028a42ac0246ae1b455e0c7036f5e865b5fe6b9c16331867a3ce40dc0377f06`. El usuario aportó una captura que muestra el directorio V3 en el Dataset privado de código y confirmó que añadió manualmente la **CELDA 9 V3 bloqueada** al notebook. Una primera comprobación devolvió `candidatos: ninguno` porque el input V3 no estaba actualizado; tras actualizarlo, la celda resolvió el release exacto y emitió sólo estados bloqueados. El nuevo protocolo local [`AETHEL_D1C_V3_RETRY_DECISION_PROTOCOL_2026-08-23.md`](AETHEL_D1C_V3_RETRY_DECISION_PROTOCOL_2026-08-23.md) y el contrato `aethel_d1c_v3_r1_authorization_contract.json` mantienen cada autorización independiente y en `false`.

La revisión local posterior identificó que el lanzador D1C conservaba el marcador V1 como único release permitido. El perfil local V3-R1 mantiene V1 como valor predeterminado y permite exclusivamente el release `d1c-v4-v3-r1-launcher-profile-train-only` bajo una sexta confirmación de perfil. La plantilla local de **CELDA 10** combina esa puerta con las cinco puertas de retry. El bundle V4 fue construido y validado localmente: TAR SHA-256 `7905caff0c40552b0ae6780f5991827f0106cb34b6dafa1bd51f9508db061c51` y ZIP SHA-256 `08d51374a9684340d7ffe47d48a2f9edf6eb36b0bb123b72ae56bd0f397c043a`; ambos excluyen corpus, JSONL, pesos, checkpoints y bytecode. El usuario aportó una captura que confirma **Version 16 — complete** y el estado **Success** para esa versión privada. V4 todavía no está adjuntado al notebook y CELDA 10 no se ha añadido ni ejecutado; no hubo GPU, retry, lectura de datos, outputs, checkpoints, holdout, promoción ni serving.

## Resultado esperado y límites de comunicación

Al cerrar A1–A3 habrá una preparación local más rápida y menos ambigua, no un modelo. Al cerrar A4 podría haber evidencia para fijar un candidato; al cerrar A6 podría existir un prototipo experimental verificable. En todos los casos se mantendrán separadas las expresiones **diagnóstico**, **candidato**, **prototipo verificable** y **producto**. Ninguna de ellas implica conciencia, AGI, calidad de frontera, disponibilidad comercial ni inferencia propia activa en la aplicación web.

## Referencias

[1]: ./AETHEL_E0_V8_REMEDIATION_PLAN_2026-08-23.md "Evidencia E0 V8 y puertas D0–D3"

[2]: ./d1a_v1_router_diagnostic_evidence.json "Resumen seguro D1A"

[3]: ./d1b_v1_router_diagnostic_evidence.json "Resumen seguro D1B"
