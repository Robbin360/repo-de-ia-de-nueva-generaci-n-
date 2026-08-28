# Plan de remediación basado en evidencia — Seed E0 V8

## Propósito y límites

Este documento convierte la auditoría del experimento **Seed E0 V8** en un plan de diagnóstico reproducible. No autoriza un nuevo entrenamiento, no modifica el Dataset privado `aethel-nextgen-data-v1`, no carga `latest.pt`, no reanuda pesos y no promociona el checkpoint. Los cambios de configuración o cualquier uso de GPU requieren una autorización nueva e independiente.

> **Estado:** E0 V8 es un experimento Seed completado y persistido. No es un modelo de serving, un benchmark, un modelo Edge/Pro ni evidencia de capacidad comercial.

## Evidencia que motiva el diagnóstico

| Señal auditada | Valor observado | Interpretación permitida |
|---|---:|---|
| Pasos de entrenamiento | 4,992 | La corrida programada terminó y dejó recibo de recuperación. |
| Parámetros entrenables | 97,154,564 | Corresponden a la configuración Seed ejecutada, no a una promesa de rendimiento. |
| Pérdida de entrenamiento inicial/final | 10.422280 / 7.821860 | Hubo descenso neto; no prueba convergencia suficiente ni utilidad. |
| Mínimo de pérdida | 6.273063 en paso 3,850 | La pérdida posterior osciló; la causa no está demostrada por esta telemetría. |
| Holdout EN aislado | pérdida 7.771877; PPL 2372.921097; 256 segmentos | Resultado real de una evaluación final aislada; no es benchmark comparativo. |
| Holdout ES aislado | pérdida 10.774346; PPL 47779.196085; 256 segmentos | Resultado real de una evaluación final aislada; revela una brecha EN/ES en esta ejecución, sin atribución causal automática. |
| Router `healthy` | 156 pasos sanos; 4,836 no sanos | El router requiere diagnóstico antes de cualquier promoción o afirmación de MoE saludable. |
| Router final | `healthy=false`, entropía mínima 0.477242, desequilibrio máximo 0.163269 | La telemetría registra concentración desigual de expertos; no identifica por sí sola el hiperparámetro responsable. |
| Curiosidad | 2,048 evaluaciones `observe_only`; 0 acciones externas | La gobernanza se mantuvo pasiva y no ejecutó acciones externas autónomas. |

La fuente primaria de estos valores es `metrics_rank_0.jsonl`; las evaluaciones EN/ES proceden de sus JSON separados. La auditoría completa está en [`AETHEL_E0_V8_ARTIFACT_AUDIT_2026-08-23.md`](AETHEL_E0_V8_ARTIFACT_AUDIT_2026-08-23.md).

## Hipótesis que permanecen por probar

Las siguientes son **hipótesis de ingeniería**, no conclusiones:

1. La dinámica actual de corrección de sesgo del router puede ser demasiado agresiva, demasiado lenta o estar insuficientemente informada para el corpus y la duración E0.
2. La divergencia entre las evaluaciones EN y ES puede estar relacionada con cobertura léxica, segmentación/tokenización, secuenciación de lotes, distribución efectiva de entrenamiento o una combinación de factores no observados. El Dataset congelado no se debe alterar para probarlo.
3. La estabilización de la pérdida de entrenamiento no permite distinguir entre límite de escala, configuración de optimización, capacidad de modelo o ruido de lote sin una matriz de diagnóstico limitada.

## Puerta D0 — auditoría de configuración sin GPU

Antes de ejecutar un nuevo experimento, se debe construir un informe de configuración que enlace la identidad exacta de E0 V8, el hash del tokenizer y los contadores congelados del Dataset. Debe ser de sólo lectura y no usar holdout para selección.

La implementación D0 no abre `metrics_rank_0.jsonl`, pesos ni shards: sólo vincula la evidencia estática auditada con `package_manifest.json`. La extracción de entropía, carga máxima, desequilibrio, sesgo del router y el campo `healthy` por capa requiere un contrato separado y no se debe inventar ni usar para escoger candidatos sin revisión previa.

El contrato concreto de la implementación D0 está en [`AETHEL_E0_V8_D0_AUDIT_CONTRACT.md`](AETHEL_E0_V8_D0_AUDIT_CONTRACT.md). La preparación local validada enlaza el marcador exacto de código, evidencia estática y `package_manifest.json`; no abre métricas por capa, pesos, shards ni contenido holdout. Por eso D0 no sustituye el análisis de telemetría que correspondería preparar para D1, ni autoriza GPU, Kaggle, selección de candidatos o cambios en el Dataset.

## Puerta D1 — diagnóstico corto del router con datos de entrenamiento

Tras una autorización específica de GPU, el primer experimento posterior debe ser un diagnóstico corto y aislado, inicializado desde cero y alimentado **sólo por shards de entrenamiento**. No debe cargar ni reanudar `latest.pt`, y el holdout EN/ES permanece excluido de muestreo, tokenización, selección y ajuste.

| Control | Requisito |
|---|---|
| Arquitectura | Seed E0 actual: 4 capas, dimensión 512, GQA 8/2, 8 expertos y top-2. |
| Datos | Dataset privado congelado y validado; sólo split de entrenamiento. |
| Duración | Cadencia corta predeclarada, con guardado de telemetría por paso y sin evaluación holdout durante la selección. |
| Comparación | Una sola variable por candidato: por ejemplo, magnitud de corrección de sesgo del router. |
| Observabilidad | Persistir pérdidas, cargas/entropías por capa, `healthy`, contador de tokens y configuración exacta. |
| Parada | Bloquear el candidato si falla preflight, smoke CUDA, integridad de Dataset o la salida de telemetría. |

La selección entre diagnósticos debe basarse sólo en telemetría de entrenamiento y estabilidad operacional, no en holdout. El resultado permitido es una recomendación de configuración para una corrida posterior; no es un modelo promocionable.

La primera ventana fue D1A, documentada en [`AETHEL_D1_ROUTER_DIAGNOSTIC_PROTOCOL_2026-08-23.md`](AETHEL_D1_ROUTER_DIAGNOSTIC_PROTOCOL_2026-08-23.md): 768 pasos desde inicialización nueva, router E0 intacto, telemetría por capa y validación *train-only*. Tras autorizaciones separadas, finalizó con `D1A_DIAGNOSTIC_COMPLETE`; el router mostró 78 pasos saludables, 690 no saludables y mínima entropía 0,333333. Después de otra confirmación, **Save Version** privado en Kaggle indicó **Version #3 — Successful**. Esto preserva notebook y salida sin acreditar ni autorizar inspección, descarga, carga, movimiento, reanudación o promoción de un checkpoint. La clasificación sigue siendo diagnóstica y no permite usar holdout, checkpoints, promoción ni serving.

El único experimento siguiente documentado fue D1B, definido en [`AETHEL_D1B_ROUTER_BIAS_PROTOCOL_2026-08-23.md`](AETHEL_D1B_ROUTER_BIAS_PROTOCOL_2026-08-23.md). Probó desde inicialización nueva y con sólo train el cambio único `router_bias_step: 0.05 → 0.01`, con GPU T4 ×2 y el mismo presupuesto de 768 pasos. La salida compartida terminó en `D1B_DIAGNOSTIC_COMPLETE`, con 44 pasos saludables y 724 no saludables; el resultado `D1B_ROUTER_NOT_IMPROVED` no supera los 78/690 de D1A. La revisión segura en [`AETHEL_D1A_D1B_ROUTER_EVIDENCE_REVIEW_2026-08-23.md`](AETHEL_D1A_D1B_ROUTER_EVIDENCE_REVIEW_2026-08-23.md) descarta la hipótesis de que reducir únicamente el paso de sesgo resuelva el router en esta ventana, sin atribuir causalidad. No se cargó checkpoint, no se abrió holdout, no hubo red ni promoción. Por tanto no existe selección de candidato ni autorización para D2/D3: cualquier propuesta posterior exige un protocolo nuevo, basado en esta evidencia y sujeto a confirmaciones separadas.

El protocolo D1C está en [`AETHEL_D1C_ROUTER_AUX_LOSS_PROTOCOL_2026-08-23.md`](AETHEL_D1C_ROUTER_AUX_LOSS_PROTOCOL_2026-08-23.md). Mantiene el baseline D1A y registra la única variación explícita del peso auxiliar `0.01 → 0.05`; sus criterios de apoyo o descarte estaban fijados antes de la única corrida V1 autorizada. La **Version 13** de código y las CELDAS 6/7 verificaron el release y las puertas previas. La corrida V1 no completó su resumen seguro porque esa versión omitía `D1C` de las opciones CLI; no se clasifica, no se reanuda, no se repite automáticamente ni se inspeccionan outputs/checkpoints. El usuario confirmó que el ZIP correctivo `d1c-v2-summary-cli-fix-train-only` fue cargado como nueva versión privada de código, sin número visual compartido. La CELDA 8 ya verificó el release V2 en modo bloqueado y no autorizó GPU, nueva ejecución, Dataset, holdout, promoción ni serving.

Como preparación posterior, el marcador `d1c-v3-retry-cell-train-only` y su bundle ZIP SHA-256 `7028a42ac0246ae1b455e0c7036f5e865b5fe6b9c16331867a3ce40dc0377f06` contienen la plantilla de retry cerrada y el fix V2. Una captura aportada por el usuario muestra el directorio V3 dentro del Dataset privado de código, y el usuario confirmó que pegó una CELDA 9 V3 bloqueada en el notebook. La primera comprobación se bloqueó antes de accesos sensibles porque el input V3 no estaba actualizado; una vez actualizado, la celda resolvió el release exacto y emitió sus dos estados bloqueados. El release V4 `d1c-v4-v3-r1-launcher-profile-train-only` incorpora el perfil de lanzador V3-R1 y la CELDA 10 con seis puertas cerradas; sus bundles son TAR SHA-256 `7905caff0c40552b0ae6780f5991827f0106cb34b6dafa1bd51f9508db061c51` y ZIP SHA-256 `08d51374a9684340d7ffe47d48a2f9edf6eb36b0bb123b72ae56bd0f397c043a`. El usuario aportó una captura de **Version 16 — complete** con estado **Success** para V4 privada. No hubo selección de GPU, retry, acceso a Dataset train/holdout, outputs o checkpoints asociado a V3-R1/V4. D2, D3, holdout, promoción y serving continúan bloqueados.

## Puerta D2 — análisis bilingüe sin tocar el Dataset

La siguiente inspección local debe cuantificar, sin exponer texto del corpus, los siguientes metadatos por idioma y por shard: número de registros, longitud de tokens, proporción de truncamiento, tasa de tokens desconocidos —si aplica— y distribución de lotes durante el launch. La finalidad es identificar diferencias de preparación observables; no reescribir, mezclar, publicar ni sustituir los 22 shards.

Si se considera una estrategia de muestreo o currículo bilingüe, debe tratarse como una configuración de entrenamiento nueva, con manifiesto propio y autorización explícita. El holdout seguirá reservado exclusivamente para auditoría final del candidato ya seleccionado.

## Puerta D3 — corrida candidata y auditoría final

Sólo después de D0, D1 y D2, y mediante otra autorización explícita, puede prepararse una corrida candidata. Sus requisitos mínimos son:

1. Release de código distinto, hashable y seleccionado de forma exacta por el notebook.
2. Preflight offline, smoke CUDA de memoria líquida y telemetría del router antes del entrenamiento largo.
3. Checkpoints, recibo de recuperación, inspección estructural y manifiesto de lanzamiento persistidos.
4. Una única evaluación final separada EN/ES sobre holdout aislado, después de fijar el candidato por evidencia de entrenamiento.
5. Informe que compare hechos auditados sin llamar al candidato Edge, Pro, benchmark o producto.

## Decisiones expresamente fuera de alcance

No están autorizados por este plan: ajustar pesos manualmente, cargar el checkpoint en la web, habilitar serving, ejecutar Sueño/LoRA como promoción, cambiar el Dataset congelado, evaluar iterativamente contra holdout, afirmar autonomía de aprendizaje, ni usar las métricas E0 V8 como reclamo comercial.

### Resultado D1C V3-R1 — corrida train-only completada

El usuario compartió el resumen seguro de una única corrida D1C V3-R1 desde inicialización nueva, con 768 pasos, 1.572.864 tokens, `router_aux_loss_weight=0.05`, `holdout_content_read=false`, `checkpoint_loaded=false` y sin peticiones de red. El resumidor corrigió el bloqueo de V1 y emitió `D1C_DIAGNOSTIC_COMPLETE`.

La clasificación predefinida es **`D1C_ROUTER_NOT_IMPROVED`**: hubo 67/768 pasos saludables, frente al mínimo requerido de 117; la entropía mínima fue 0,3333333433, que no supera el umbral estricto >0,333333; el desequilibrio máximo fue 0,1875 y sí cumplió; la pérdida media fue 9,43690848, por encima del máximo permitido 9,35257273. Esta evidencia no acredita un modelo funcional o comercial y no selecciona candidato para D2/D3. No se deben abrir, descargar, mover, cargar ni deserializar outputs o checkpoints. D1D no se inicia automáticamente; cualquier hipótesis posterior requiere un protocolo nuevo.
