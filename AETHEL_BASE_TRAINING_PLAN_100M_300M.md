# Aethel Base: plan operativo de entrenamiento 100M–300M

**Estado:** plan ejecutable pendiente de autorización para consumir GPU/Kaggle  
**Fecha:** 2026-08-28  
**Autor:** Manus AI  
**Proyecto:** `aethel-platform`

## 1. Decisión ejecutiva

La ruta correcta es **entrenar primero `pilot-100m`**, no saltar directamente a 300M ni activar crecimiento dinámico. El objetivo de esta primera versión no es demostrar AGI ni una puntuación humana de IQ, sino obtener un modelo causal real que pueda conversar en español e inglés, seguir instrucciones cortas y resolver razonamiento y matemáticas básicas en tareas no vistas.

La corrida 100M será la referencia científica. Sólo se promoverá a 300M si el modelo 100M demuestra que el problema principal es capacidad representacional y no datos, tokenización, mezcla lingüística, optimización o colapso del router. De esta manera evitamos gastar sesiones de T4 en escalar un experimento mal especificado.

> **Regla de honestidad:** un checkpoint, una pérdida menor o una respuesta ocasionalmente coherente no equivalen a un modelo competente. Cada afirmación funcional debe estar respaldada por un recibo reproducible con hashes, configuración, semilla, datos y resultados de holdout.

## 2. Estado de recursos y restricciones

El repositorio ya contiene un Transformer causal con RoPE, GQA, Sparse MoE top-2, módulos NextGen, telemetría, checkpoints reanudables, tokenizador BPE y harnesses que no producen puntuaciones cuando faltan predicciones reales. También existe un corpus Edge preparado en Kaggle con diez shards de entrenamiento, un `validation.jsonl` y `tokenizer.json`.

La disponibilidad conocida es de dos Tesla T4 en Kaggle, con sesiones temporales de hasta doce horas y una cuota semanal ya agotada en la sesión heredada. Por tanto, **el plan queda diseñado y listo, pero no se inicia otra ejecución GPU automáticamente**. Será necesario disponer de una nueva ventana de cuota y autorización expresa del usuario. El primer lanzamiento debe usar una sola GPU y el modo distribuido se habilitará únicamente después de validar FSDP en dos procesos.

| Recurso o restricción | Tratamiento operativo |
|---|---|
| Dos Tesla T4 | Primera corrida en `world_size=1`; segunda GPU sólo tras prueba FSDP CUDA aprobada |
| Sesión temporal de doce horas | Checkpoint portátil cada 15–30 minutos y al menos cada 2.000–4.000 pasos de optimizador |
| Cuota Kaggle agotada | No ejecutar ahora; conservar plan y pedir autorización cuando la cuota esté disponible |
| Almacenamiento efímero | Guardar `latest.pt`, snapshots, tokenizador, manifiesto y métricas en Dataset privado/Release antes de cerrar |
| Peso grande para GitHub | No subir pesos al repositorio; usar Dataset privado, Release Asset o LFS |
| Holdout | Nunca usarlo para tokenizer, entrenamiento, replay ni selección adaptativa |
| Métricas | Sólo aceptar JSONL emitido por el runner o por evaluadores reproducibles |

## 3. Configuraciones exactas

Las cifras siguientes proceden de `engine/report_model_budget.py`; son presupuestos analíticos, no una medición de VRAM real. El preset de 300M queda ligeramente por encima de 300 millones debido a embeddings, módulos NextGen y la forma redondeada del MLP MoE.

| Variante | Vocabulario | Dimensión | Capas | Cabezas KV | Expertos / activos | Contexto | Parámetros totales | Parámetros activos aprox. | Adam estimado |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pilot-100m` | 32.000 | 512 | 4 | 2 | 8 / 2 | 1.024 | 97,16M | 40,53M | 1,09 GiB |
| `research-300m` | 32.000 | 768 | 8 | 3 | 8 / 2 | 2.048 | 344,34M | 117,84M | 3,85 GiB |
| `adaptive-research-300m` | 32.000 | 768 | 8 | 3 | 8 / 2 | 2.048 | 347,88M | 121,39M | 3,89 GiB |

El cálculo de Adam considera aproximadamente pesos BF16, gradientes BF16 y dos estados FP32 por parámetro. No incluye activaciones, buffers CUDA, fragmentación, KV-cache ni memoria adicional del runtime. En T4, la decisión final de microbatch debe salir de un **smoke test de memoria**, no de esta tabla.

### Configuración recomendada para la primera corrida

Se utilizará `pilot-100m` con `adaptive_refinement_steps=0`, cuatro capas, ocho expertos y top-2. La primera corrida debe mantener desactivados el crecimiento dinámico, los adaptadores automáticos y las modificaciones experimentales del router. Los módulos Sólido, Líquido, Sueño, memoria y curiosidad se conservarán en el contrato arquitectónico, pero el entrenamiento base debe medirlos por separado y no permitir que introduzcan cambios no auditados en el núcleo.

La razón es experimental: primero necesitamos una línea base interpretable. Si simultáneamente cambiamos routing, plasticidad, replay, capacidad y datos, no podremos atribuir una mejora o una regresión a una causa concreta.

## 4. Corpus y preparación

Se partirá del Dataset Edge ya preparado, con sus diez shards y el `prepared_manifest.json` verificado. No se debe volver a construir el corpus durante la sesión de entrenamiento. La preparación y el entrenamiento son trabajos distintos.

La mezcla objetivo para el entrenamiento base será bilingüe y explícita. Como punto inicial se propone 45% inglés general, 45% español general y 10% tareas técnicas, conversación, traducción y razonamiento verificable. Esta proporción no es una afirmación óptima: se congelará como versión `mix-v1` y se comparará contra una variante de balanceo posterior sólo si el diagnóstico por idioma lo justifica.

| Componente | Proporción inicial | Uso |
|---|---:|---|
| Texto general EN | 45% | Fluidez, sintaxis, continuidad y vocabulario |
| Texto general ES | 45% | Fluidez, concordancia, registro y vocabulario |
| Instrucción, diálogo y traducción | 5% | Seguir órdenes y transferencia EN↔ES |
| Razonamiento y matemáticas verificables | 5% | Procedimientos cortos, aritmética y problemas básicos |

Los documentos deben deduplicarse por contenido normalizado, eliminarse si están vacíos o corruptos y conservar su procedencia. El holdout debe dividirse por familias de problema y no sólo al azar. Las familias de razonamiento y matemáticas deben contener plantillas o números no presentes literalmente en el entrenamiento.

El tokenizador BPE se entrena únicamente con el split de entrenamiento. Su hash se registra en cada checkpoint y cualquier reanudación con otro tokenizador se rechaza. Si el corpus real disponible no alcanza el volumen de la mezcla, se reduce el presupuesto de tokens; **no se duplica artificialmente el mismo texto para aparentar más conocimiento**.

## 5. Currículo de entrenamiento

El currículo se ejecuta por etapas y cada etapa termina con un checkpoint y una evaluación de validación. La progresión busca que el modelo aprenda primero la distribución lingüística, luego la interacción y finalmente procedimientos de razonamiento, evitando que ejemplos de instrucciones escasos dominen un núcleo que aún no conoce bien EN y ES.

| Etapa | Tokens objetivo 100M | Mezcla dominante | Criterio de avance |
|---|---:|---|---|
| A — Lenguaje bilingüe | 600M–800M | 50/50 EN/ES general | Pérdida descendente, sin divergencia y diferencia EN/ES controlada |
| B — Interacción | 150M–250M | Diálogo, instrucciones, resumen y traducción | Responde en el idioma pedido y conserva contexto corto |
| C — Procedimientos | 100M–200M | Clasificación, contradicciones, composición y planificación | Mejora en familias no vistas, no sólo en plantillas conocidas |
| D — Matemáticas | 50M–100M | Aritmética, fracciones, porcentajes, potencias y logaritmos simples | Resultado correcto ante verificador externo |
| E — Consolidación | 100M–200M | Replay estratificado de A–D | Sin regresión relevante en EN, ES o tareas anteriores |

El rango total inicial es aproximadamente **1,0–1,55 mil millones de tokens** para 100M. En Kaggle no se intenta completarlo en una sola sesión; se divide en sesiones reanudables. Para 300M, el objetivo inicial será **2–4 mil millones de tokens**, pero sólo después de que 100M pase las puertas de estabilidad y calidad.

### Orden de ejecución por sesiones

La primera sesión debe ser de calibración y arranque: 500 pasos de optimizador, medición de memoria, throughput, pérdida por idioma y escritura de un checkpoint completo. Si el smoke test falla, se ajusta microbatch o longitud; no se cambia la arquitectura en mitad de la corrida.

Las siguientes sesiones continúan el mismo contrato global. Cada una carga `latest.pt`, `tokenizer.json` y el manifiesto exacto; comprueba la configuración, el hash del tokenizador, el horizonte global, la precisión, la semilla y el estado RNG. Después realiza entrenamiento hasta el límite temporal, ejecuta validación ligera y exporta los artefactos antes de salir.

## 6. Reanudación segura y cálculo de pasos

El runner debe conservar el modelo, el optimizador, el scheduler, el scaler si se utiliza, el paso global, la semilla/RNG, el estado runtime de NextGen y el contrato de reanudación. El contrato existente `aethel-training-resume/v2` debe tratarse como obligatorio.

La cantidad de pasos no se fija copiando un número histórico. Se calcula así:

```text
tokens_por_actualización = world_size × microbatch × seq_len × grad_accum
actualizaciones_objetivo = ceil(tokens_objetivo / tokens_por_actualización)
```

Como ejemplo operativo, con una GPU, `microbatch=2`, `seq_len=1024` y `grad_accum=8`, se obtienen 16.384 tokens por actualización. Con dos procesos equivalentes, serían 32.768. Estos son ejemplos de cálculo, no valores garantizados: el microbatch real depende de la VRAM observada.

Cada snapshot debe incluir al menos `latest.pt`, un snapshot inmutable numerado, `tokenizer.json`, `prepared_manifest.json`, configuración JSON, `metrics_rank_0.jsonl`, resumen de validación, hashes SHA-256 y un recibo de entorno. Se conservan los tres últimos snapshots y uno de cada hito de etapa.

La reanudación se rechaza si cambia cualquiera de estos elementos: arquitectura, vocabulario, tokenizador, longitud de contexto, precisión, estrategia distribuida, tamaño global de batch, horizonte total, semilla contractual o manifiesto de datos. Para cambiar el horizonte, se crea una nueva corrida derivada con un nuevo identificador y se documenta la relación con el padre; no se sobrescribe la evidencia anterior.

## 7. Evaluación reproducible

La evaluación se divide en pérdida de lenguaje y pruebas funcionales. La pérdida es útil para observar aprendizaje, pero no decide por sí sola si el modelo conversa o razona. La batería debe ejecutarse en un proceso separado, con el holdout protegido y con una versión de evaluador congelada.

| Grupo | Prueba | Métrica principal | Desglose obligatorio |
|---|---|---|---|
| Lenguaje | Holdout causal | Pérdida y perplejidad | EN, ES y total |
| Conversación | Instrucciones no vistas | Exactitud de criterio | EN, ES, idioma solicitado |
| Transferencia | Traducción y cambio de idioma | Exactitud y preservación semántica | EN→ES, ES→EN |
| Razonamiento | Reglas, contradicciones y planificación | Porcentaje correcto | Familia, dificultad y plantilla |
| Matemáticas | Aritmética y álgebra elemental | Resultado verificable | Operación y tipo de error |
| Retención | Replay de etapas anteriores | Variación contra checkpoint padre | Cada idioma y capacidad |
| MoE | Salud del router | Cobertura, concentración, overflow y entropía | Por capa y por etapa |
| Eficiencia | Ejecución fija | Tokens/s, latencia p50/p95 y memoria | Mismo hardware y secuencia |

Para el piloto se recomienda un conjunto interno de al menos 200 casos por familia funcional, balanceado por idioma y con plantillas retenidas fuera del entrenamiento. Los casos no deben reutilizar respuestas del corpus. Los resultados se publican sólo cuando el evaluador pueda reconstruir exactamente qué ejemplos fueron usados.

## 8. Puertas de calidad y decisión

Los umbrales siguientes son **criterios de promoción propuestos**, no resultados actuales. Deben congelarse en un archivo de evaluación antes de seleccionar el checkpoint ganador. Una media global nunca puede compensar un colapso de un idioma.

| Puerta | Mínimo propuesto para promover 100M | Acción si falla |
|---|---|---|
| Integridad | Cero NaN, checkpoint reanudable y hashes completos | Rechazar corrida o reparar runner antes de continuar |
| Lenguaje | Holdout estable y ninguna degradación EN/ES superior al 25% frente al otro idioma | Revisar mezcla, tokenizer o datos |
| Conversación | ≥70% de criterios en EN y ≥70% en ES en el conjunto piloto | Continuar etapa B o ajustar datos; no escalar |
| Transferencia | ≥60% en cada dirección EN↔ES | Reforzar traducción y control de idioma |
| Razonamiento | ≥55% en cada familia y mejora frente a baseline sin entrenamiento | Revisar currículo y evitar memorizar plantillas |
| Matemáticas | ≥70% en operaciones básicas y ≥50% en logaritmos simples | Mantener herramienta/verificador y ampliar ejemplos curados |
| Retención | Pérdida funcional no empeora más de 10% en replay | Aumentar replay o descartar el candidato |
| MoE | Sin overflow sostenido ni concentración extrema por capa | Ajustar router mediante experimento aislado |
| Eficiencia | Medición reproducible, sin declarar ventaja no observada | Conservar baseline y no afirmar ultra-eficiencia |

La promoción a 300M requiere además que una ablación muestre que el 100M está limitado por capacidad y no por una deficiencia corregible de datos u optimización. Si el 100M falla conversación EN/ES pero mejora al corregir la mezcla, se corrige el currículo y se vuelve a entrenar 100M. Si pasa lenguaje pero falla razonamiento, se mejora la etapa C/D antes de aumentar parámetros.

## 9. Qué queda activado y qué queda bloqueado

Durante el entrenamiento base quedan activos RoPE, GQA, MoE top-2, telemetría, checkpoints, memoria de trabajo necesaria para el forward y los contratos de estabilidad. La Roca puede actuar como ruta estable; El Líquido, Sueño, curiosidad y memoria episódica deben registrar estados y experimentos, pero no crear parámetros ni promover cambios durante una conversación.

Quedan bloqueados el crecimiento automático de expertos, la creación de parámetros en caliente, la promoción automática de adaptadores, el acceso del replay al holdout y cualquier afirmación de inteligencia general. Una nueva capacidad, como logaritmos, se incorpora primero como adaptador o herramienta versionada, se prueba contra un conjunto nuevo y sólo se integra al núcleo si mejora la tarea sin regresión EN/ES.

## 10. Procedimiento Kaggle que se ejecutará cuando exista cuota

Primero se crea una sesión nueva con el Dataset de código `aethel-direct-train-source-v1`, el Dataset de datos `aethel-edge-corpus-v1` y, para reanudación, el Dataset privado de artefactos de la sesión anterior. Se ejecutan únicamente las celdas de inventario, preflight, entrenamiento y empaquetado del cuaderno nuevo; no se mezclan con el cuaderno histórico D1D/D1E.

Después se verifica que hay exactamente un `prepared_manifest.json`, un tokenizer compatible, diez shards de entrenamiento y un holdout. Se comprueba CUDA, el nombre de la GPU, la precisión soportada, la memoria disponible y que no existe una salida residual con el mismo identificador. Si cualquiera de esas comprobaciones falla, la sesión termina sin iniciar entrenamiento.

A continuación se ejecuta el smoke test de 500 pasos. Si la memoria y la pérdida son válidas, se continúa con la etapa A hasta el límite de la sesión. El runner debe guardar snapshots periódicos y realizar una exportación final antes de que Kaggle cierre el proceso. Al terminar, se inspecciona el recibo: paso global, tokens vistos, pérdida, métricas por idioma, salud MoE, hash de checkpoint, hash de tokenizer y hash del manifiesto.

En la siguiente sesión se monta el Dataset de artefactos, se selecciona el snapshot más reciente que pase la validación, se ejecuta la comprobación de reanudación y se continúa con el mismo contrato. El botón de guardado de versión de Kaggle preserva el entorno/notebook, pero **no sustituye** la exportación explícita de los pesos y del estado del optimizador. La evidencia importante debe estar en un Dataset o Release que pueda montarse en una sesión posterior.

## 11. Resultado esperado al completar la primera campaña

Si todas las puertas se superan, tendremos un **checkpoint real de Aethel Base 100M**, su tokenizador, corpus versionado, estado completo de optimización, métricas de entrenamiento y un informe de evaluación EN/ES. Será un modelo Edge experimental que podrá cargarse para generación controlada y conversación, pero su calidad se describirá exactamente según las métricas observadas.

Si alguna puerta falla, también obtendremos un resultado útil: una causa de rechazo reproducible. En ese caso no se anuncia el modelo como funcionalmente competente ni se aumenta la escala por intuición. Se modifica una sola variable —datos, mezcla, optimización, routing o currículo— y se crea una nueva versión derivada.

## 12. Decisión final recomendada

La próxima ejecución autorizada debe ser **100M, `pilot-100m`, 1,0–1,55B tokens acumulados, sesiones reanudables y evaluación por etapas**. No se debe empezar por 300M, no se debe añadir crecimiento dinámico todavía y no se deben publicar pesos sin un recibo de evaluación. Después de una base 100M competente, el 300M será una expansión razonable; antes de eso, sería principalmente más coste y más superficie de errores.

### Referencias internas

1. [`AETHEL_BASE_CAPABILITY_SPEC.md`](./AETHEL_BASE_CAPABILITY_SPEC.md): contrato de competencia bilingüe, razonamiento, matemáticas, retención y crecimiento gobernado.
2. [`engine/report_model_budget.py`](./engine/report_model_budget.py): presets, conteo analítico de parámetros y estimación transparente del estado Adam.
3. [`engine/aethel_resume.py`](./engine/aethel_resume.py): contrato de reanudación `aethel-training-resume/v2` y validación de payload completo.
4. [`engine/evaluate_nextgen.py`](./engine/evaluate_nextgen.py): evaluación real de pérdida y perplejidad sobre holdout BPE.
5. [`engine/evaluate_benchmarks.py`](./engine/evaluate_benchmarks.py): agregación de predicciones reales para benchmarks, sin puntuaciones inventadas.
6. [`AETHEL_ROUTER_STABILITY_GATE.md`](./AETHEL_ROUTER_STABILITY_GATE.md): separación entre entropía, asignación top-k, cobertura, concentración y overflow.
