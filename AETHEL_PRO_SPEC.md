# Aethel Pro — especificación técnica experimental

**Versión:** Pro v0.1 — diseño previo a GPU
**Estado:** propuesta arquitectónica no validada
**Autor:** Manus AI
**Fecha:** 2026-08-28
**Relación con Edge:** Pro comparte contratos e ideas del motor Aethel, pero no reutiliza automáticamente sus pesos ni demuestra sus capacidades.

> **Advertencia de alcance.** Este documento define una configuración razonada para construir y evaluar Aethel Pro. Los tamaños, presupuestos y comportamientos descritos son hipótesis de ingeniería. No constituyen evidencia de inteligencia, razonamiento, bilingüismo nativo, eficiencia relativa ni calidad de generación. Sólo un entrenamiento reproducible y una evaluación independiente pueden validar esas afirmaciones.

## 1. Objetivo de Pro

Aethel Pro es la variante de mayor capacidad de la familia Aethel. Su propósito es estudiar si una arquitectura causal con atención eficiente, mezcla dispersa de expertos y módulos cognitivos acotados puede mejorar la competencia lingüística, matemática básica, programación, recuperación de contexto y razonamiento estructurado frente a una línea base densa comparable.

El objetivo no es hacer que una etiqueta arquitectónica implique consciencia, voluntad, inteligencia humana o una puntuación de IQ. El sistema sólo podrá describirse como competente en una capacidad cuando exista una prueba definida, datos retenidos, una ejecución reproducible y un resultado registrado.

| Variante | Propósito | Escala de diseño | Estado actual |
|---|---|---:|---|
| Edge | Iteración pequeña, barata y portable | ~97 M entrenables observados en su perfil | Checkpoint preservado; evaluación Edge pendiente por falta de cuota GPU |
| Pro v0.1 | Experimento de capacidad y eficiencia con MoE | 3,816,136,704 parámetros totales calculados | Especificación; no entrenado |
| Pro dense baseline | Control experimental | Igual dimensión y capas, sin expertos dispersos | Debe construirse antes de afirmar ventaja de MoE |

## 2. Principios de diseño

Pro debe conservar una separación estricta entre el modelo autoregresivo y los servicios auxiliares. El Transformer calcula la distribución del siguiente token. La memoria, la curiosidad, el sueño/replay y la neuromodulación producen señales y registros controlados; no deben modificar pesos durante inferencia ni admitir datos al entrenamiento sin procedencia, aprobación y separación del holdout.

La eficiencia se tratará como una métrica, no como una promesa. GQA reduce el número de cabezas de clave/valor frente a MHA, mientras que MoE permite que cada token active una fracción de los parámetros expertos. Ambos mecanismos añaden complejidad de routing, comunicación y balanceo que debe medirse en hardware real [1] [2].

La configuración usa RoPE para incorporar posición relativa mediante rotaciones en las representaciones de atención [3]. Usa BF16 cuando el dispositivo lo soporte, con acumulaciones y operaciones sensibles conservando la precisión necesaria. En PyTorch, la ruta de precisión mixta debe utilizar el contrato vigente de AMP y validarse con pruebas numéricas y de estabilidad [4].

## 3. Configuración base Pro v0.1

| Parámetro | Valor propuesto | Justificación y límite |
|---|---:|---|
| `vocab_size` | 32.768 | Compatibilidad inicial con el BPE bilingüe existente; debe fijarse por hash antes de entrenar |
| `hidden_size` (`d`) | 1.536 | Dimensión divisible entre 16 cabezas y 4 cabezas KV |
| `num_layers` | 24 | Profundidad suficiente para un experimento Pro sin saltar inmediatamente a una escala no operable |
| `num_attention_heads` | 16 | Atención multi-cabeza con `head_dim = 96` |
| `num_key_value_heads` | 4 | GQA con cuatro grupos de consultas por cabeza KV |
| `max_seq_len` inicial | 4.096 | Reduce el riesgo de memoria; 8.192 queda como segunda configuración evaluable |
| `num_experts` | 8 | Capacidad total dispersa por capa |
| `active_experts` | 2 | Routing top-2 por token |
| `ffn_hidden_size` | 4.096 | Tamaño de cada proyección experta; debe medirse contra una variante densa |
| Expert activation | SwiGLU | Tres matrices principales por experto: gate, up y down |
| Normalización | RMSNorm o equivalente estable | Debe mantenerse idéntica entre baseline y Pro |
| Posición | RoPE | Aplicada a Q y K; base y escalado deben fijarse en el checkpoint |
| Atención | Causal GQA | Sin acceso a tokens futuros |
| Router | lineal `d → experts`, top-2 | Capacidad, overflow, balanceo y ruido deben quedar registrados |
| Precision de entrenamiento | BF16 + estados FP32 | Requiere GPU compatible; no se asume sólo por disponer de CUDA |
| Optimizador | AdamW | Estados `m` y `v` en FP32 en la primera implementación |
| Contexto de despliegue inicial | 4.096 | 8.192 requiere una nueva medición de KV-cache y throughput |
| Cabeza de salida | Tied embeddings | Comparte matriz de entrada salvo que una prueba justifique lo contrario |

### 3.1. Flujo de un bloque

Un bloque Pro seguirá el siguiente flujo lógico, sujeto a confirmación mediante implementación y pruebas:

```text
x
 ├─ RMSNorm
 ├─ Q, K, V
 ├─ RoPE(Q, K)
 ├─ atención causal GQA
 ├─ proyección de salida + residual
 ├─ RMSNorm
 ├─ router top-2
 ├─ dispatch a dos expertos SwiGLU
 ├─ combinación ponderada + residual
 └─ salida del bloque
```

La capacidad de expertos será limitada por token y por lote. El lanzador debe registrar tokens descartados o derivados por overflow, distribución por experto, entropía del router, carga máxima, carga media y porcentaje de tokens en los dos expertos más frecuentes. Un router que selecciona dos expertos casi siempre no demuestra especialización útil aunque la pérdida baje.

### 3.2. Módulos cognitivos Aethel

| Módulo | Función propuesta | Regla de seguridad |
|---|---|---|
| La Roca | Parámetros, contratos y reglas estables | Inmutable durante observación e inferencia; cambios sólo mediante nueva versión |
| El Líquido | Estado adaptativo y propuestas de actualización | Versionado, con TTL, procedencia y reversión |
| Memoria episódica | Eventos recientes recuperables | No se presenta como conocimiento verificado; límite de capacidad y trazabilidad |
| Memoria semántica | Índice de hechos o documentos con citas | Sólo incorpora fuentes aprobadas y separadas del holdout |
| Espacio de trabajo | Variables intermedias y planes estructurados | No expone ni trata cadenas internas como hechos garantizados |
| Curiosidad | Prioriza incertidumbre, novedad y contradicción | No puede autoautorizar datos, objetivos ni entrenamiento |
| Sueño/replay | Selecciona experiencias para consolidación offline | Requiere curación, aprobación, hash de procedencia y evaluación de regresión |
| Neuromodulación | Señales de prioridad, plasticidad o cautela | No cambia pesos por sí sola en serving |

Estos módulos no se cuentan como parámetros del Transformer salvo que una futura implementación los integre explícitamente dentro de la red. Su memoria persistente y su consumo de CPU/RAM deben medirse aparte.

## 4. Conteo exacto de parámetros de la configuración base

El conteo siguiente asume matrices lineales sin bias, SwiGLU con tres matrices por experto, vocabulario compartido entre entrada y salida, y dos RMSNorm por capa. Los valores son deterministas para la configuración de la sección 3.

### 4.1. Variables derivadas

```text
d = 1.536
L = 24
H = 16
H_kv = 4
E = 8
K = 2
d_ff = 4.096
V = 32.768
head_dim = d / H = 96
```

### 4.2. Embeddings

Con salida atada a los embeddings, la matriz se cuenta una sola vez:

```text
P_embedding = V × d
             = 32.768 × 1.536
             = 50.331.648
```

Si se decide no atar la cabeza de salida, deben añadirse otros `V × d = 50.331.648` parámetros, y el total subiría a **3.866.468.352**.

### 4.3. Atención GQA por capa

Las matrices Q y O tienen dimensión `d × d`. K y V producen sólo `H_kv × head_dim` canales cada una:

```text
P_attention = P_Q + P_K + P_V + P_O
            = d² + 2 × d × (H_kv × head_dim) + d²
            = 1.536² + 2 × 1.536 × (4 × 96) + 1.536²
            = 5.898.240
```

Esta cifra cuenta parámetros, no FLOP. El ahorro de GQA en memoria de KV-cache se produce porque se almacenan cuatro grupos KV en vez de dieciséis cabezas KV completas; el throughput real depende de kernel, lote, secuencia, hardware y ancho de banda.

### 4.4. Expertos SwiGLU por capa

Cada experto utiliza tres matrices principales —gate, up y down— con coste aproximado `3 × d × d_ff`. Los ocho expertos residentes en una capa son:

```text
P_experts_per_layer = E × 3 × d × d_ff
                    = 8 × 3 × 1.536 × 4.096
                    = 150.994.944
```

Los dos expertos activos por token representan, para el cálculo de parámetros tocados por el camino experto, lo siguiente:

```text
P_active_experts_per_layer = K × 3 × d × d_ff
                            = 2 × 3 × 1.536 × 4.096
                            = 37.748.736
```

Los ocho expertos siguen residentes en memoria aunque cada token sólo use dos. Por eso **parámetros totales** y **parámetros activos** no son equivalentes a memoria ocupada.

### 4.5. Router y normalización

```text
P_router_per_layer = d × E = 1.536 × 8 = 12.288
P_norm_per_layer   = 2 × d = 3.072
```

### 4.6. Total

```text
P_layer_total = P_attention + P_experts_per_layer
              + P_router_per_layer + P_norm_per_layer
              = 156.908.544

P_total = P_embedding + L × P_layer_total
        = 50.331.648 + 24 × 156.908.544
        = 3.816.136.704 parámetros
```

El camino activo aproximado por token, excluyendo el embedding de entrada y la cabeza de salida atada, es:

```text
P_active_per_token ≈ L × (
    P_attention
  + P_active_experts_per_layer
  + P_router_per_layer
  + P_norm_per_layer
)
= 1.047.896.064
```

Este número es una medida contable de matrices seleccionadas, no una afirmación de que Pro tenga exactamente 1,048 mil millones de parámetros efectivos ni una medida directa de coste. El router, las secuencias y el kernel determinan el trabajo real.

## 5. Cálculos de VRAM

### 5.1. Convenciones

Las cifras binarias usan `1 GiB = 2^30 bytes` y `1 MiB = 2^20 bytes`. Un valor de VRAM no es una garantía de ejecución: se debe reservar espacio para allocator, kernels, comunicaciones, fragmentación, activaciones, workspace CUDA, dataloader y picos de memoria.

Para los cálculos estáticos se usa `P = 3.816.136.704`:

| Elemento | Tipo asumido | Fórmula | Tamaño calculado |
|---|---|---:|---:|
| Pesos | BF16, 2 bytes | `P × 2` | 7,108 GiB |
| Gradientes | BF16, 2 bytes | `P × 2` | 7,108 GiB |
| Primer estado AdamW | FP32, 4 bytes | `P × 4` | 14,216 GiB |
| Segundo estado AdamW | FP32, 4 bytes | `P × 4` | 14,216 GiB |
| Copia maestra | FP32, 4 bytes | `P × 4` | 14,216 GiB |
| Estado estático de entrenamiento | suma anterior | `2P + 2P + 8P + 4P` bytes | **56,865 GiB** |

El estado estático de entrenamiento no incluye activaciones, buffers temporales, fragmentación ni comunicación. Por tanto, una GPU de 48 GiB no puede contener esta configuración sin sharding, offload, optimizador más compacto u otra reducción de memoria. Dos GPU de 16 GiB tampoco son suficientes para la ruta AdamW no particionada; disponer de dos T4 no convierte automáticamente el entrenamiento en viable.

### 5.2. Inferencia con pesos BF16

```text
VRAM_pesos_BF16 = P × 2 / 2^30
                = 7,108 GiB
```

Una estimación de base para serving de una sola réplica es, por tanto, **7,1 GiB sólo para los pesos**. Debe añadirse memoria de runtime y KV-cache. En FP32, los pesos ocuparían aproximadamente **14,216 GiB**; esa modalidad no es el objetivo inicial de serving.

### 5.3. KV-cache GQA

Para una secuencia con longitud `S`, dos estados —K y V—, `L` capas, `H_kv` cabezas KV, `head_dim` canales y BF16:

```text
VRAM_KV = 2 × L × S × H_kv × head_dim × 2 bytes
        = 2 × 24 × S × 4 × 96 × 2
```

| Contexto por secuencia | KV-cache BF16 por secuencia | Ocho secuencias simultáneas |
|---:|---:|---:|
| 2.048 | 72 MiB | 0,563 GiB |
| 4.096 | **144 MiB** | **1,125 GiB** |
| 8.192 | 288 MiB | 2,250 GiB |

Estos valores son para una secuencia completa en la configuración base y no incluyen estados de paginación, metadata, batching dinámico ni buffers del motor. Una implementación paged KV-cache puede cambiar la fragmentación, no la cantidad lógica de claves y valores que deben conservarse.

### 5.4. Escenarios de inferencia

| Escenario | Pesos | KV-cache | Presupuesto mínimo antes de runtime | Recomendación |
|---|---:|---:|---:|---|
| 1 × 4.096, BF16 | 7,108 GiB | 0,141 GiB | 7,249 GiB | GPU de 16 GiB con margen para runtime; validar |
| 8 × 4.096, BF16 | 7,108 GiB | 1,125 GiB | 8,233 GiB | Requiere medir batching y memoria residual |
| 1 × 8.192, BF16 | 7,108 GiB | 0,281 GiB | 7,389 GiB | Validar antes de prometer contexto largo |
| 1 × 4.096, FP32 | 14,216 GiB | 0,281 GiB en FP32 | 14,497 GiB | No es la ruta eficiente objetivo |

El margen operativo real debe dejar espacio adicional. La tabla no debe utilizarse para afirmar que una GPU concreta ejecutará Pro hasta completar un smoke test con el kernel, el backend y la versión exacta de PyTorch.

### 5.5. Escenarios de entrenamiento distribuido

La ruta AdamW FP32 completa requiere aproximadamente **56,9 GiB de estado estático agregado** antes de activaciones. Con FSDP o un esquema equivalente de partición completa, una aproximación ideal divide esos estados entre `N` dispositivos:

```text
VRAM_estática_por_GPU ≈ 56,865 GiB / N
```

| Dispositivos | Límite ideal agregado por GPU | Lectura operativa |
|---:|---:|---|
| 1 × 48 GiB | 56,865 GiB | No cabe antes de activaciones |
| 2 × 16 GiB | 28,433 GiB | No cabe aun con partición ideal completa |
| 2 × 48 GiB | 28,433 GiB | Puede ser viable con FSDP, checkpointing y margen cuidadoso |
| 4 × 24 GiB | 14,216 GiB | Más razonable con partición y activación controlada |
| 4 × 48 GiB | 14,216 GiB | Mejor margen para secuencia y comunicación |
| 8 × 24 GiB | 7,108 GiB | Ruta más cómoda, no garantía de throughput |

La división ideal no considera duplicación temporal durante all-gather, buffers de comunicación, activaciones ni expertos remotos. En MoE, el expert parallelism puede reducir memoria local de expertos, pero añade tráfico entre dispositivos. La aceptación debe medir memoria máxima por GPU y no sólo el promedio.

### 5.6. Activaciones y longitud de secuencia

El cálculo exacto de activaciones depende del kernel, microbatch, checkpointing, atención implementada, layout y si se conserva o recomputa cada tensor. Por ello esta especificación no inventa una cifra única de activaciones. El runner debe registrar `max_memory_allocated`, `max_memory_reserved`, duración de pasos, tokens por segundo, overflow del router y tamaño de microbatch.

Como presupuesto orientativo, en una GPU de 80 GiB quedarían unos **23,135 GiB** después del estado estático agregado no particionado de 56,865 GiB, antes de reservar memoria adicional. En una GPU de 48 GiB el presupuesto sería negativo: la ruta no cabe sin partición u optimizaciones de estado. Estos saldos no sustituyen una medición.

## 6. Plan de entrenamiento

La primera corrida Pro no debe intentar aprender todos los dominios a la vez. Se propone un currículo por etapas, con un manifiesto congelado y un holdout que nunca se mezcla con entrenamiento:

| Etapa | Contenido | Puerta de salida |
|---|---|---|
| P0 — contrato | Smoke CPU, conteo, carga, tokenizer y un batch | Checkpoint y forward deterministas |
| P1 — estabilidad | Mezcla EN/ES equilibrada, secuencias cortas y router monitorizado | Sin NaN, gradientes finitos, router sin colapso |
| P2 — lenguaje | Corpus bilingüe con procedencia y deduplicación | Pérdida de validación EN/ES y brecha registrada |
| P3 — fundamentos | Matemáticas básicas, instrucciones y programación con fuentes permitidas | Exactitud en conjuntos retenidos, sin usar el holdout |
| P4 — capacidad | Contexto mayor, currículo y mezcla controlada | Comparación contra baseline denso |
| P5 — adaptación | SFT o adaptación separada, si se autoriza | Regresión y retención antes de promoción |

El objetivo inicial propuesto es de **30–80 mil millones de tokens vistos**, sujeto al presupuesto de hardware y a la calidad del corpus. Este rango es un plan de trabajo, no una regla universal ni una garantía de calidad. Cada sesión debe guardar pesos, AdamW, scheduler, scaler, RNG, configuración, tokenizer, hashes del manifiesto, paso global y recibo de integridad.

El checkpoint no debe considerarse utilizable sólo porque se haya escrito `latest.pt`. Antes de promoción se exige carga estricta, generación controlada, pérdida por idioma, pruebas de regresión y verificación de que los pesos no mutan durante evaluación.

## 7. Evaluación mínima de Pro

| Área | Medición mínima | Criterio de honestidad |
|---|---|---|
| Lenguaje EN/ES | Pérdida y perplejidad en holdout separado | Informar cada idioma y el intervalo/protocolo |
| Generación | Prompts fijos, temperatura y seed documentadas | No seleccionar sólo ejemplos favorables |
| Matemáticas básicas | Exactitud en problemas retenidos y parser determinista | Separar memorización de generalización |
| Razonamiento | Tareas reproducibles con respuestas verificables | No inferir razonamiento interno a partir de prosa |
| Router MoE | Entropía, carga, overflow, expertos activos y tokens/s | Comparar contra baseline y registrar coste |
| Eficiencia | VRAM máxima, latencia p50/p95 y tokens/s | Nunca afirmar ventaja relativa sin control comparable |
| Memoria | Recuperación con procedencia y prueba de no contaminación | Separar memoria auxiliar del conocimiento del LM |
| Estabilidad | Reanudación, seeds, regresión y pérdida por etapa | Conservar recibos y fallos, no sólo el mejor run |

MMLU, HumanEval, GSM8K u otras baterías sólo deben mostrar puntuaciones cuando el dataset, el harness, la versión y las predicciones estén disponibles y auditados. El dashboard de Aethel debe dejar estos campos vacíos o marcarlos como no disponibles mientras falte evidencia.

## 8. Eficiencia: hipótesis y comparación

Las hipótesis que justifican Pro son: (1) GQA puede reducir el coste de KV-cache frente a MHA con la misma dimensión; (2) MoE puede aumentar la capacidad total sin activar todos los expertos por token; (3) checkpointing de activaciones y kernels fused pueden reducir memoria; y (4) una ruta Triton/CUDA validada puede cambiar el rendimiento de atención y dispatch. Ninguna hipótesis autoriza a afirmar ultra-eficiencia.

La comparación mínima debe usar el mismo tokenizer, corpus, secuencia, hardware, número de tokens, calidad de kernel y criterio de convergencia. Se deben ejecutar al menos:

| Control | Cambio único |
|---|---|
| Dense baseline | Sustituir los ocho expertos por un FFN denso con presupuesto comparable |
| MHA control | Sustituir GQA por 16 cabezas KV para medir KV-cache y calidad |
| Sin módulos cognitivos | Desactivar memoria, sueño y señales auxiliares en el camino del LM |
| Pro completo | Activar únicamente los componentes cuya integración haya pasado contratos |

Los indicadores principales serán tokens/s, milisegundos por token, VRAM pico, energía o coste si está disponible, pérdida a tokens vistos y calidad por benchmark. Menor VRAM no implica menor coste total si aumenta la comunicación o el número de tokens requerido para converger.

## 9. Riesgos y mitigaciones

| Riesgo | Señal observable | Mitigación |
|---|---|---|
| Colapso del router | Entropía baja, carga concentrada, expertos inactivos | Capacidad, balanceo, jitter controlado y ablations; no seleccionar una corrección después de mirar sólo el resultado favorable |
| OOM | Picos de memoria o proceso terminado | FSDP, activation checkpointing, microbatch menor, secuencia menor, optimizer sharding |
| Inestabilidad BF16 | NaN, overflow o pérdida divergente | Escalería adecuada, clipping, acumulación, comprobación de hardware y fallback explícito |
| Sobreajuste | Holdout empeora mientras train mejora | Deduplicación, currículo, regularización y evaluación retenida |
| Contaminación del holdout | Fuentes o hashes coinciden | Manifiesto independiente, filtros y auditoría de procedencia |
| Coste de MoE | Comunicación domina el paso | Expert parallelism medido, agrupación de tokens y baseline denso |
| Memoria auxiliar no trazable | Recuperaciones sin fuente o replay automático | Citas, TTL, aprobación y snapshots reversibles |
| Exageración de capacidades | Claims sin benchmark | Dashboard y documentación con estados `no evaluado` o `no disponible` |

## 10. Artefactos que debe producir una implementación futura

La implementación Pro debe generar, como mínimo, un archivo de configuración inmutable, un manifiesto de datos, tokenizer y hash, contrato de checkpoint, recibos de sesión, métricas compactas por ventana, métricas de router, registro de VRAM, evaluación EN/ES, generaciones controladas y comparación contra baseline. El paquete debe poder reanudarse en otra sesión sin asumir que `/kaggle/working` persiste.

Los pesos grandes no deben entrar en el historial Git normal. Deben conservarse en almacenamiento de artefactos privado hasta decidir explícitamente su publicación. Si se publica un checkpoint, el mecanismo debe admitir el tamaño, conservar su SHA-256 y documentar exactamente qué contiene, qué no contiene y si está evaluado.

## 11. Criterios de promoción

Pro sólo podrá llamarse **checkpoint experimental entrenado** cuando exista un checkpoint cargable y un recibo de entrenamiento válido. Sólo podrá llamarse **modelo evaluado** cuando se ejecute la evaluación aislada con resultados reproducibles. Sólo podrá presentarse una mejora de eficiencia cuando exista un baseline comparable. Sólo podrá describirse una capacidad bilingüe, matemática o de razonamiento con resultados que la midan explícitamente.

La promoción a un servicio no debe ocurrir por el número de parámetros. Requiere integridad del checkpoint, seguridad del runtime, latencia y memoria dentro del objetivo, evaluación de regresión, procedencia de datos, controles de privacidad y una política clara para memoria y aprendizaje continuo.

## 12. Resumen ejecutivo

Aethel Pro v0.1 queda definido como un Transformer causal de **24 capas**, dimensión **1.536**, **16 cabezas de consulta**, **4 cabezas KV**, **8 expertos SwiGLU** con routing **top-2**, contexto inicial **4.096** y vocabulario BPE de **32.768**. Bajo los supuestos indicados, tiene **3.816.136.704 parámetros totales**, de los cuales el camino experto activo por token cuenta aproximadamente **1.047.896.064** matrices seleccionadas, sin que esto reduzca la memoria residente de todos los expertos.

La memoria de pesos BF16 calculada es **7,108 GiB**; el estado estático de entrenamiento BF16 + AdamW FP32 + copia maestra es aproximadamente **56,865 GiB**, sin activaciones ni overhead. La configuración no debe intentarse en una GPU de 48 GiB sin partición u optimización de estados, y dos T4 de 16 GiB no son suficientes para la ruta AdamW no particionada. La siguiente decisión técnica debe ser un smoke test CPU y un diseño de baseline, no una afirmación de calidad.

## Referencias

[1]: https://arxiv.org/abs/2305.13245 "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints"

[2]: https://jmlr.org/papers/v23/21-0998.html "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"

[3]: https://arxiv.org/abs/2104.09864 "RoFormer: Enhanced Transformer with Rotary Position Embedding"

[4]: https://pytorch.org/docs/stable/amp.html "PyTorch Automatic Mixed Precision package"
