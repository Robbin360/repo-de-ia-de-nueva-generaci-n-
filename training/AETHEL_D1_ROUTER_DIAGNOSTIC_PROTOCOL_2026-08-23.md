# Protocolo propuesto D1A — diagnóstico corto del router MoE

## Estado y decisión solicitada

Este documento registra **D1A**, una ventana diagnóstica desde inicialización nueva. Sus componentes de contrato y bundle privado se probaron primero con fixtures sintéticas; el release exacto se verificó como versión privada del Dataset de código en Kaggle y D1A se ejecutó después mediante autorizaciones separadas. El documento no autoriza acciones posteriores de Kaggle, cambio del Dataset de datos, `Save Version`, `Save & Run All`, movimiento de checkpoints ni D1B.

La motivación es acotada: E0 V8 terminó 4.992 pasos, pero su telemetría registró 4.836 pasos con router no saludable y un `min_entropy` final de `0.47724199295043945`; esa señal no identifica por sí misma una causa. [1] D0 confirmó la identidad de release, el manifiesto de datos y los límites de no lectura de pesos/corpus/holdout, pero no analizó telemetría cruda por capa ni autorizó GPU. [2]

> **Resultado permitido:** telemetría de entrenamiento de una línea base corta y una recomendación sobre si conviene preparar un único cambio de configuración para una futura D1B. No produce un candidato, benchmark, evaluación holdout, checkpoint promocionable, serving, Edge o Pro.

## Diseño D1A: línea base, no candidato

D1A no cambia la arquitectura ni el comportamiento E0. Su función es observar con granularidad suficiente el inicio de la dinámica del router; por ello fija la configuración Seed ya auditada y no introduce una intervención que pueda confundir el diagnóstico.

| Control | Valor propuesto | Justificación y límite |
|---|---:|---|
| Inicialización | Nueva; sin `--resume` | No lee ni reutiliza `latest.pt` o snapshots de E0 V8. |
| Arquitectura | dim 512; 4 capas; 8/2 cabezas GQA; 8 expertos; top-2; contexto 1.024 | Replica el Seed E0 V8 para aislar la observación del router. [3] |
| Router | `router_bias_step=0.05`; `router_bias_limit=0.5` | Línea base de E0 V8; D1A no propone que estos valores sean correctos. [3] |
| Datos permitidos | Sólo `corpus/train-*.jsonl` o `train-*.jsonl.gz` | Los entrenadores existentes descubren únicamente esos patrones; los holdouts deben permanecer sin abrir. [4] |
| Semilla y estrategia | semilla 17; `single`; mundo 1 | Reproducción operacional de una sola GPU; no acredita DDP/FSDP. |
| Presupuesto | 768 pasos; batch 2; acumulación 16; 1.572.864 tokens teóricos | Ventana diagnóstica fija, menor que E0; no es un presupuesto de convergencia. |
| Fallback | PyTorch experimental, sólo si se autoriza de forma expresa | Triton estricto sigue bloqueado; el resultado permanece no promocionable. [3] |
| Evaluación | Ninguna sobre holdout | El holdout EN/ES no participa en entrenamiento, tokenización, muestreo, selección ni comparación D1A. |

El valor de 768 pasos no es una meta de calidad: es una ventana de observación predeclarada. Su cálculo de tokens es `768 × 2 × 1.024`; cualquier cambio de duración, batch, acumulación, semilla, arquitectura o hiperparámetro de router crea un protocolo nuevo y requiere revisión antes de la ejecución.

## Implementación local validada y release verificado

Se implementaron localmente los tres componentes siguientes y la prueba aislada `training/test_d1a_local_contracts.py` confirmó sus contratos sin usar el Dataset Aethel, GPU, pesos, checkpoints ni entrenamiento. La identidad `d1a-v1-router-baseline-train-only` quedó incluida en un bundle privado y se verificó como release exacto en el Dataset de código de Kaggle. La quinta celda de preparación se ejecutó primero en modo bloqueado y posteriormente D1A completó una corrida diagnóstica desde cero.

| Componente local | Responsabilidad | Límite verificable |
|---|---|---|
| `training/validate_aethel_train_only_mount.py` | Lee el manifiesto raíz y verifica sólo bytes/hashes de shards `train-*` y del tokenizador. | No abre archivos, paths, bytes, hashes ni texto de holdout; tampoco parsea texto train. No reutiliza el validador E0 completo. |
| `training/run_kaggle_d1a_router_diagnostic.sh` | En una futura ejecución, exige autorizaciones D1A, release exacto, salida nueva, preflight train-only y smoke CUDA antes del entrenamiento desde cero. | Rechaza reanudación, salida existente y release incorrecto; no invoca `evaluate_nextgen.py`, `inspect_checkpoint.py` ni rutas holdout. |
| `training/summarize_d1a_router_metrics.py` | Agrega exclusivamente `metrics_rank_0.jsonl` de entrenamiento en `router_diagnostic.json`. | Rechaza otro nombre de archivo, incluidos checkpoints; no carga `.pt`, no abre corpus ni hace red. |
| Celda Kaggle D1A | Preparada como `AETHEL_D1A_ROUTER_DIAGNOSTIC_CELL.py`; selecciona exactamente el release D1A y escribe bajo un nuevo directorio de trabajo sólo cuando sea autorizada. | Permanece con `D1A_EXECUTION_ENABLED=False`; no borra entradas/salidas existentes, no reutiliza E0 y no activa ejecución sin confirmaciones explícitas separadas. |
| Variante habilitable | Preparada localmente como `AETHEL_D1A_ROUTER_DIAGNOSTIC_EXECUTION_CELL.py` y validada con `D1A_EXECUTION_CELL_STATIC_GATES_PASSED`. | Las cuatro puertas se abrieron sólo tras confirmación final y la corrida terminó; no autorizan repetir, reanudar, promover ni iniciar D1B. |

La instrumentación del entrenador ya emite por paso `routing` con `entropy`, `max_load`, `imbalance` y `bias` por capa, junto con `router_health`, pérdida, tokens y configuración. [4] Por tanto, el resumen D1A debe **agregar lo existente** y no inventar un indicador de salud nuevo.

## Bundle local auditado

Se construyó un bundle para revisión privada con el release `d1a-v1-router-baseline-train-only`: `aethel-nextgen-source.tar.gz`, SHA-256 `6391310a5b4aa0e78644c7454a110054fc0f23c7313fb2758afaf863463ee39f`. La auditoría de lista verificó 158 entradas, el hash contra su manifiesto, los scripts D1A requeridos y la ausencia de corpus, archivos JSONL, pesos, checkpoints, bytecode, cachés y dependencias. Se añadió únicamente como nueva versión privada del Dataset de código y se verificó su marcador; el hash no acredita una promoción ni cambio del Dataset de datos.

La celda se añadió como quinta celda del notebook y se ejecutó una vez sólo en su modo bloqueado. Resolvió el montaje fuente D1A bajo versión `(10)` y el Dataset privado congelado; su salida final fue `D1A_CELL_PREPARED_NOT_EXECUTED`. La propia salida declaró que no se seleccionó GPU, no se copió código, no se leyó corpus/holdout, no se cargaron pesos y no se entrenó. Ese resultado verifica preparación de inputs, no una corrida D1A.

La variante habilitable usa un directorio de trabajo nuevo `aethel-d1a-router-baseline-run-v1`. Su sintaxis y sus bloqueos estáticos se verificaron antes de la ejecución; las cuatro puertas sólo se modificaron tras confirmación final inmediata y no autorizan repetir la corrida.

## Resultado D1A observado

El log compartido finalizó con `D1A_DIAGNOSTIC_COMPLETE` y el resumen con `D1A_METRICS_SUMMARIZED`. La evidencia resumida, derivada exclusivamente de ese log y sin copiar corpus, pesos, checkpoints ni telemetría cruda, está en `training/d1a_v1_router_diagnostic_evidence.json`.

| Campo del log | Valor observado |
|---|---:|
| Pasos / tokens finales | 768 / 1.572.864 |
| Pérdida: mínimo / media / máximo | 7,648315 / 9,259973 / 10,438221 |
| Router saludable / no saludable | 78 / 690 pasos (10,156250% / 89,843750%) |
| Máximo desequilibrio / mínima entropía | 0,187500 / 0,333333 |
| Estado final | `D1A_DIAGNOSTIC_COMPLETE` |

Las medias de entropía por capa fueron 0,526389, 0,494771, 0,449744 y 0,448883; las medias de desequilibrio fueron 0,149431, 0,157085, 0,165791 y 0,166020. La entropía mínima cayó por debajo del umbral configurado de 0,50; eso es una observación de telemetría, no una atribución causal ni una selección de hiperparámetros.

El resumen reafirma `checkpoint_loaded=false`, `raw_corpus_read=false`, `holdout_content_read=false`, `network_requests=0` y `promotion_authorized=false`. El log compartido no verifica de forma independiente la persistencia de un checkpoint o salida D1A nueva; no se moverá, cargará, inspeccionará, reanudará ni promoverá artefacto alguno. Holdout, D1B, serving, Edge y Pro siguen bloqueados.

## Preflight y límites tras la ejecución

La corrida D1A usó el preflight *train-only* en lugar del preflight E0 genérico, que recorre también holdouts para verificar el paquete. [5] El validador D1A incluido en el release exacto declara `holdout_content_read=false`, `checkpoint_loaded=false`, `network_requests=0`, el hash del tokenizador, hashes de manifiesto aplicables y el listado de shards de entrenamiento verificados. El resultado final observado es coherente con ese contrato, pero no sustituye la revisión independiente de ningún artefacto de salida.

Toda repetición D1A requeriría un nuevo preflight train-only, comprobación de Dataset sin cambios, smoke CUDA, directorio de salida nuevo y confirmaciones separadas. No hay autorización de repetición, reanudación ni cambio de configuración en este documento.

Todo artefacto D1A que el entrenador haya generado permanece privado, sin cargar, inspeccionar, mover, reanudar ni promover. La evidencia usada para el diagnóstico es el resumen de `metrics_rank_0.jsonl` reportado en el log, no una evaluación holdout ni una comparación de checkpoints.

Después de una confirmación adicional y específica, se ejecutó **Save Version** en Kaggle. La interfaz aportada por el usuario mostró **Version #3 — Successful**. La versión privada conserva el notebook y la salida de esa ejecución, pero no verifica de forma independiente el contenido binario de ningún artefacto ni autoriza descargar, cargar, mover, inspeccionar, reanudar o promover un checkpoint. El Dataset de datos permanece sin cambios.

La preparación documental de una posible revisión futura está delimitada en [`AETHEL_D1A_EVIDENCE_REVIEW_PLAN_2026-08-23.md`](AETHEL_D1A_EVIDENCE_REVIEW_PLAN_2026-08-23.md). Ese plan no inspecciona outputs; exige una confirmación nueva y específica incluso para metadatos visibles, y preserva la prohibición de tocar `.pt`, corpus, holdout, JSONL crudo o cualquier acción de ejecución.

El único diseño posterior preparado fue [`AETHEL_D1B_ROUTER_BIAS_PROTOCOL_2026-08-23.md`](AETHEL_D1B_ROUTER_BIAS_PROTOCOL_2026-08-23.md). Probó un único cambio de `router_bias_step: 0.05 → 0.01` desde inicialización nueva y sólo con train. Tras las puertas B1–B5c, D1B completó 768 pasos bajo GPU T4 ×2 y emitió `D1B_DIAGNOSTIC_COMPLETE`. El router terminó con 44 pasos saludables y 724 no saludables, por debajo de los 78 saludables de D1A; se clasifica `D1B_ROUTER_NOT_IMPROVED`, no como candidato. No hubo carga de checkpoint, lectura de corpus/holdout crudo, red ni promoción. D2, D3, holdout, selección de candidato, cambios del Dataset de datos, promoción y serving siguen bloqueados.

## Telemetría y decisión permitida

`router_diagnostic.json` deberá contener, por capa y para la corrida completa, el mínimo/media/máximo de entropía, carga máxima, desequilibrio y norma o rango del sesgo; además de los conteos de `healthy`, pérdida, tokens vistos, configuración exacta y cualquier parada por integridad. La telemetría cruda `metrics_rank_0.jsonl` debe persistirse sin alteración.

Los umbrales existentes se reportarán tal como están configurados (`min_router_entropy=0.50`, `max_router_imbalance=0.30`), no como un criterio de aceptación recién inventado. [4] D1A puede concluir únicamente una de estas opciones: **(a)** no hay telemetría íntegra y se bloquea cualquier continuación; **(b)** la línea base exige más instrumentación; o **(c)** existe evidencia suficiente para proponer exactamente un cambio de router para revisión posterior. D1A nunca elige ese cambio mediante holdout.

La corrida se detendrá ante fallo de identidad de release, preflight train-only, integridad de Dataset, smoke CUDA, escritura de telemetría, pérdida no finita o interrupción de infraestructura. Un `healthy=false` aislado o sostenido es evidencia diagnóstica, no permiso para continuar, promover o reiniciar automáticamente.

## Puertas observadas y permisos restantes

| Hito | Estado | Acción que sigue bloqueada |
|---|---|---|
| Bundle y fuente privada | Completados bajo confirmaciones separadas; release `(10)` verificado. | Modificar `aethel-nextgen-data-v1`. |
| Celda y GPU | Completadas bajo confirmaciones separadas; la corrida D1A terminó. | Repetir, cambiar configuración o usar otra celda. |
| Ejecución D1A | Completada una vez, sólo train y desde cero. | Evaluar holdout, reanudar/prometer pesos o iniciar D1B. |
| Persistencia | `Save Version` privado completado; interfaz reportó **Version #3 — Successful**. | `Save & Run All`, descarga/carga/inspección/movimiento/reanudación/promoción de checkpoint o D2/D3. |

No hay autorización implícita entre filas. Si una pantalla se cierra, una versión cambia o aparece un error, se detiene y se pide una nueva confirmación en el punto exacto.

## Referencias

[1]: ./AETHEL_E0_V8_ARTIFACT_AUDIT_2026-08-23.md "Auditoría de artefactos E0 V8"
[2]: ./AETHEL_E0_V8_D0_AUDIT_CONTRACT.md "Contrato de auditoría estática D0"
[3]: ./AETHEL_E0_V8_REMEDIATION_PLAN_2026-08-23.md "Plan de remediación E0 V8"
[4]: ../engine/train_aethel_gpu.py "Entrenador GPU Aethel"
[5]: ./run_kaggle_e0_offline_preflight.py "Preflight offline E0"
