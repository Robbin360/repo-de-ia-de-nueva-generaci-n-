# Aethel 100T — Especificación de escala distribuida

**Estado:** propuesta de investigación, no modelo entrenado  
**Autor:** Manus AI  
**Fecha:** 2026-08-28  
**Relación con Aethel Pro:** diseño de capacidad extrema; no sustituye la ruta Pro ni Edge

> Esta especificación responde a “100 millones de millones de parámetros”, interpretado como aproximadamente **100T parámetros**. Si se quería decir 100 mil millones, la arquitectura y el presupuesto cambian por varios órdenes de magnitud.

## 1. Tesis de diseño

Un modelo de 100T sólo tiene sentido como **capacidad dispersa y distribuida**. Un Transformer denso de 100T obligaría a ejecutar todos los parámetros en cada token y haría prohibitivo el cómputo. La propuesta, por tanto, separa tres conceptos: capacidad total almacenada, parámetros activos por token y memoria de estado distribuida.

La meta no es afirmar que 100T sea automáticamente más inteligente. La hipótesis es que una gran colección de expertos especializados, combinada con routing estable, recuperación verificable, deliberación condicional y destilación, puede ofrecer más capacidad útil por unidad de cómputo que un modelo denso de igual tamaño. Esa hipótesis sólo se aceptará después de comparaciones controladas.

## 2. Configuración de referencia

| Componente | Valor propuesto | Razón | Riesgo principal |
|---|---:|---|---|
| Parámetros totales | 100.0356T | Capacidad MoE objetivo | No implica calidad ni conocimiento útil |
| Capas | 80 | Profundidad para composición | Optimización y pipeline complejos |
| Dimensión oculta | 16.384 | Ancho suficiente para expertos grandes | Activaciones y comunicación elevadas |
| Atención | 128 cabezas / 8 KV heads | GQA para reducir KV-cache | Calidad dependiente de la relación Q/KV |
| Dimensión por cabeza | 128 | `16.384 / 128` | Fijada por la geometría de atención |
| Expertos por capa | 776 | Ajuste al total aproximado de 100T | Routing y almacenamiento masivo |
| Expertos activos | Top-2 | Cómputo condicional | Colisiones y desbalance |
| FFN por experto | 32.768 | Bloque de gran capacidad | Coste activo de unos 304B por token |
| Vocabulario | 131.072 | Cobertura multilingüe y multimodal futura | Embeddings de 2,15B parámetros |
| Contexto de referencia | 32.768 tokens | Documentos largos | KV-cache y atención caros |
| Precisión de entrenamiento | BF16 + estados FP32 | Estabilidad y memoria razonable | Requiere hardware especializado |
| Posición | RoPE o variante validada | Compatibilidad conceptual con Edge/Pro | Extensión larga debe medirse |

El número de expertos se obtiene de forma deliberada: con 80 capas, cada experto FFN contiene aproximadamente `3 × 16.384 × 32.768 = 1.610.612.736` parámetros. Con 776 expertos por capa, más atención, routers, normas y embeddings, el total calculado es **100.035.639.902.208 parámetros**.

## 3. Conteo reproducible

El cálculo usado es:

```text
head_dim = d / heads
embedding = vocab × d
attention_layer = d² + 2 × d × (kv_heads × head_dim) + d²
expert = 3 × d × ff
router_layer = d × experts
layer_total = attention_layer + experts × expert + router_layer + 2 × d
total = embedding + layers × layer_total
active_token = layers × (attention_layer + active_experts × expert + router_layer + 2 × d)
```

Para la configuración de referencia:

| Magnitud | Resultado |
|---|---:|
| Embeddings | 2.147.483.648 parámetros |
| Atención por capa | 570.425.344 parámetros |
| Un experto FFN por capa | 1.610.612.736 parámetros |
| Router por capa | 12.713.984 parámetros |
| Total | 100.035.639.902.208 parámetros |
| Cómputo parametrizado por token, sin embeddings | 304.351.805.440 parámetros activos equivalentes |

El script reproducible está en `/home/ubuntu/calc_aethel_100t.py`. El valor “parámetros activos equivalentes” no significa que todos esos pesos residan en una sola GPU; describe el trabajo de las capas seleccionadas más la atención y el router, antes de optimizaciones de kernel.

## 4. Presupuesto de memoria

Las cifras siguientes usan unidades binarias, con `1 GiB = 2^30 bytes` y `1 TiB = 2^40 bytes`. Son **pisos teóricos**: no incluyen activaciones, fragmentación, buffers de comunicación, duplicación, metadata del router, compiladores ni margen operativo.

### 4.1 Inferencia

| Elemento | Fórmula | Estimación |
|---|---|---:|
| Pesos BF16 | `100.0356T × 2 bytes` | 200,07 TB decimales; **181,96 TiB** |
| Pesos FP32 | `100.0356T × 4 bytes` | 400,14 TB decimales; **363,93 TiB** |
| KV-cache, una secuencia | `2 × layers × context × kv_heads × head_dim × 2` | **10 GiB** BF16 |
| KV-cache, 1.024 secuencias | anterior × 1.024 | **10.240 GiB** |

El mínimo aritmético para almacenar sólo los pesos BF16 en GPU de 80 GiB es `181,96 TiB / 0,078125 TiB = 2.329,1 GPU`; el cálculo por dispositivo debe considerar replicación, balance y buffers. Ninguna de estas cifras representa un servidor de inferencia listo.

### 4.2 Entrenamiento completo con AdamW

La estimación estática utilizada es:

```text
BF16 pesos       = 2 bytes por parámetro
BF16 gradientes  = 2 bytes por parámetro
FP32 master      = 4 bytes por parámetro
Adam m + v       = 8 bytes por parámetro
Total estático   = 16 bytes por parámetro
```

| Estado | Estimación |
|---|---:|
| Pesos BF16 | **181,96 TiB** |
| Gradientes BF16 | **181,96 TiB** |
| Copia master FP32 | **363,93 TiB** |
| Estados `m` y `v` de Adam FP32 | **727,86 TiB** |
| Estado estático total | **1.455,71 TiB** |
| GPUs de 80 GiB, suelo estático | **18.634** |
| GPUs de 128 GiB, suelo estático | **11.646** |

El suelo de GPU no es un tamaño de clúster recomendado. Para trabajar se necesitaría margen adicional para activaciones, recomputación, comunicación, checkpoints, imbalance de expertos y fallos. Con ZeRO/FSDP se reducen redundancias por GPU, pero no desaparece la cantidad total de información que el clúster debe almacenar y actualizar [1].

### 4.3 Cómputo de entrenamiento

Como aproximación de orden de magnitud, se usa `6 × parámetros activos × tokens`, aunque la cifra real cambia con la implementación, el routing, el backward, el checkpointing y las pérdidas auxiliares. Con 2 × 10^15 tokens y unos 304,35B parámetros activos equivalentes por token:

```text
FLOPs aproximados = 6 × 304.351.805.440 × 2.000.000.000.000
                  = 3,6522 × 10^24 FLOPs
```

Esto es una hipótesis de presupuesto, no una promesa de tiempo. La duración depende del hardware, la eficiencia de utilización, la red, el balance de expertos, la tasa de tokens/s y el porcentaje de tokens descartados o repetidos.

## 5. Topología distribuida propuesta

Un clúster de este tamaño requeriría varias formas de paralelismo simultáneamente. La referencia de Megatron-LM muestra que el paralelismo tensorial puede combinarse con pipeline y que el escalado real depende de la colocación de comunicación y operaciones; su demostración publicada fue a una escala muy inferior a 100T [2].

| Dimensión | Propuesta inicial | Responsabilidad |
|---|---:|---|
| Data parallel | 8–32 réplicas lógicas | Procesar lotes distintos |
| Tensor parallel | 16–64 | Dividir proyecciones y atención |
| Pipeline parallel | 8–20 etapas | Repartir las 80 capas |
| Expert parallel | 776 expertos distribuidos | Alojar y ejecutar expertos |
| Sequence/context parallel | 2–8 | Repartir secuencias largas |
| ZeRO/FSDP | Stage 3 o equivalente | Particionar estados y parámetros |

Estos números no se multiplican ciegamente para obtener el cluster final. Deben elegirse mediante un planificador que respete la topología física: NVLink/NVSwitch dentro del nodo, enlaces de alta velocidad entre nodos y grupos de expertos colocados para minimizar all-to-all. El routing global de tokens es probablemente el mayor riesgo de red.

### Flujo por token

1. El embedding y la atención producen el estado oculto.
2. El router calcula logits de expertos con ruido sólo durante entrenamiento y una política de capacidad explícita.
3. Los dos expertos seleccionados se agrupan por destino y se ejecuta un `all-to-all` de tokens.
4. Los expertos procesan los tokens con kernels fusionados.
5. Se realiza el `all-to-all` inverso y se combinan las salidas con pesos normalizados.
6. Un balanceador registra carga, overflow, latencia, pérdida auxiliar y tokens por experto.
7. Sólo después de verificar integridad se continúa al siguiente bloque de pipeline.

El router no debe enviar información a una memoria externa ni modificar pesos durante inferencia. La memoria y el aprendizaje en línea deben permanecer fuera del camino de actualización automática hasta contar con un protocolo de evaluación y reversión.

## 6. Cómo aumentar inteligencia útil sin ejecutar 100T en cada token

La propuesta fuera de la caja no es “activar más expertos siempre”, sino usar un **presupuesto cognitivo variable**:

| Ruta | Cuándo | Acción | Puerta de seguridad |
|---|---|---|---|
| Swift | Consulta fácil y baja incertidumbre | Capas tempranas, pocos expertos, sin recuperación amplia | No degradar exactitud frente a baseline |
| Standard | Consulta normal | Profundidad completa y top-2 | Latencia y pérdida de validación |
| Deliberate | Matemática, planificación o conflicto | Más pasos, herramientas y expertos especializados | Verificador independiente |
| Verify | Resultado incierto o de alto impacto | Recalcular, contrastar fuentes y detectar contradicciones | Rechazar respuesta si no pasa umbral |

La memoria externa debería guardar hechos, episodios y trazas resumidas con fuente, fecha, hash y nivel de confianza. La memoria no debe servir para introducir gradientes silenciosos en el modelo base. Las adaptaciones de dominio pueden entrenarse como módulos pequeños y promoverse sólo si superan un conjunto de regresión.

## 7. Ruta de construcción por escalas

Un 100T no debe ser el primer experimento. La progresión reduce riesgo y permite comprobar que cada propiedad sobrevive al aumento de escala.

| Fase | Tamaño orientativo | Objetivo | Condición para avanzar |
|---|---:|---|---|
| Edge | ~97M | Contrato de reanudación, EN/ES, telemetría | Evaluación aislada reproducible |
| Micro-Pro | 300–700M | Routing, memoria y rutas adaptativas | Mejora contra baseline con coste controlado |
| Pro | ~3,8B | Integración de módulos y entrenamiento distribuido pequeño | Calidad, estabilidad y throughput medidos |
| Pro-MoE | 20–100B | Expert parallel y recuperación | Balance y comunicación aceptables |
| Scale-1T | 0,5–2T | Validar ZeRO/3D parallel | Checkpoints distribuidos recuperables |
| 10T pilot | 5–10T | Probar topología y red de expertos | Coste por token y fallos dentro de presupuesto |
| 100T research | ~100T | Experimento de capacidad extrema | Sólo después de las puertas anteriores |

Una versión intermedia bien entrenada puede ser más útil que una versión 100T mal balanceada. La decisión racional es maximizar calidad validada por dólar, segundo y vatio, no maximizar el contador de parámetros.

## 8. Ablations obligatorias

Cada mecanismo debe compararse contra el mismo baseline y el mismo conjunto de evaluación. No se debe cambiar simultáneamente tamaño, datos, tokenizer y routing, porque entonces no se sabría qué produjo una mejora.

| Ablation | Comparación | Métricas |
|---|---|---|
| Dense vs MoE | Mismo presupuesto de tokens y FLOPs | Pérdida, calidad, tokens/s, balance |
| Top-1 vs Top-2 | Mismo número total de expertos | Calidad, overflow, all-to-all |
| GQA vs MLA | Misma calidad objetivo y contexto | KV-cache, latencia, pérdida |
| Profundidad fija vs adaptativa | Mismo modelo y umbral calibrado | FLOPs, calidad, errores difíciles |
| Sin memoria vs RAG verificable | Misma consulta y corpus | Exactitud factual, citas, latencia |
| Adam particionado vs baseline | Misma semilla y schedule | VRAM, recuperación, convergencia |
| BF16 vs cuantización mixta | Capas sensibles protegidas | Pérdida, estabilidad, throughput |
| Sin verificador vs Verify | Mismo generador | Matemáticas, contradicciones, rechazo correcto |

## 9. Criterios de rechazo

La escala se rechaza o se reduce si el MoE obtiene peor calidad que un baseline denso con el mismo presupuesto activo; si el router concentra tráfico en pocos expertos; si el all-to-all domina el tiempo; si los checkpoints no pueden recuperarse; si la memoria externa aumenta alucinaciones; si la ruta rápida deforma la calidad; o si las mejoras desaparecen fuera del conjunto de desarrollo.

También se rechaza cualquier afirmación de “superinteligencia”, razonamiento general, bilingüismo nativo o eficiencia extrema basada sólo en el tamaño nominal. Esas afirmaciones requieren evaluación independiente, conjuntos no vistos, análisis de errores y comparación con baselines.

## 10. Recomendación ejecutiva

**Sí es posible diseñar una arquitectura de aproximadamente 100T con MoE y entrenamiento distribuido en teoría.** No es un siguiente paso operativo para el hardware actual de Aethel: la memoria estática estimada ronda 1.455,71 TiB y el suelo aritmético supera 18.000 GPU de 80 GiB antes de activaciones y overhead. La ruta correcta es construir primero el runtime distribuido y validar escalas de 1B, 10B, 100B y 1T.

La idea más potente no es almacenar 100T por prestigio. Es tratar el modelo como un sistema con **capacidad enorme, cómputo escaso, memoria verificable y un presupuesto adaptativo**. Si las ablaciones no muestran una mejora clara por unidad de coste, el diseño debe reducirse.

## Referencias

[1]: https://arxiv.org/abs/1910.02054 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"

[2]: https://arxiv.org/abs/1909.08053 "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"

[3]: https://jmlr.org/papers/v23/21-0998.html "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"

[4]: https://arxiv.org/abs/2305.13245 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"

[5]: https://arxiv.org/abs/2405.04434 "DeepSeek-V2: A Strong Mixture-of-Experts Language Model"

[6]: https://aclanthology.org/2024.acl-long.681/ "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
