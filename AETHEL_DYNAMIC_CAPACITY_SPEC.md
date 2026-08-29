# Aethel Dynamic Capacity Specification

**Estado:** propuesta experimental para revisión arquitectónica  
**Autor:** Manus AI  
**Fecha:** 2026-08-27  
**Relación con Aethel Pro:** extensión conceptual; no reemplaza `AETHEL_PRO_SPEC.md`

> **Resumen ejecutivo.** Aethel puede diseñarse para ampliar su capacidad durante el aprendizaje, pero no conviene crear parámetros arbitrariamente en cada conversación. La estrategia técnicamente más sólida es mantener un tronco estable, añadir módulos o expertos en versiones discretas y utilizar una memoria externa que pueda crecer continuamente. Cada expansión debe superar un protocolo de replay, holdout, regresión y coste antes de incorporarse.

## 1. Escala correcta

La meta discutida es **100B**, donde `B` significa *billion* en notación inglesa. En español corresponde a **100.000 millones de parámetros**. No debe traducirse como 100 billones en escala larga española.

La primera distinción es entre **capacidad total** y **capacidad activa por token**. Un MoE puede contener 100B de parámetros totales y activar sólo una fracción para cada token. Eso reduce el cómputo medio, pero no elimina el coste de almacenar, distribuir, actualizar y mantener esos expertos.

| Magnitud | Definición | Consecuencia |
|---|---|---|
| Parámetros totales | Todos los pesos existentes, incluidos expertos no seleccionados | Determina almacenamiento y parte de la memoria distribuida |
| Parámetros activos | Pesos usados por un token o secuencia | Determina principalmente FLOPs y tráfico de activación |
| Memoria externa | Documentos, vectores, episodios y herramientas fuera de los pesos | Puede crecer sin modificar el modelo base |
| Capacidad de adaptación | Adaptadores, nuevos expertos o bloques entrenables | Permite aprendizaje por versiones, con riesgo de deriva |

## 2. ¿Puede crear parámetros al aprender?

**Sí, a nivel de arquitectura**, mediante expansión estructural. Existen precedentes de redes que expanden unidades o columnas ante nuevas tareas. Dynamically Expandable Networks propone decidir capacidad, hacer expansión selectiva y dividir unidades para reducir deriva semántica [1]. Progressive Neural Networks conserva columnas anteriores y añade nuevas con conexiones laterales [2]. Firefly Neural Architecture Descent estudia crecimiento de anchura y profundidad mediante selección estructural guiada [3]. También se ha estudiado crecer y podar redes con objetivos conjuntos de precisión y esparsidad [4].

Lo que no está demostrado por esos trabajos es que un LLM grande pueda aprender indefinidamente, en línea y sin supervisión, creando parámetros útiles sin degradar sus capacidades anteriores. Por tanto, Aethel debe usar **crecimiento gobernado**, no mutación libre.

El aprendizaje de un dato nuevo puede resolverse en cuatro niveles:

| Nivel | Qué cambia | Velocidad | Riesgo | Uso recomendado |
|---|---|---:|---:|---|
| Memoria episódica | Se guarda un episodio con fuente, fecha y confianza | Inmediata | Recuperación incorrecta o datos obsoletos | Hechos nuevos y experiencia reciente |
| Índice semántico | Se añade un documento o embedding verificable | Minutos | Retrieval irrelevante o duplicado | Conocimiento consultable |
| Adaptador | Se entrenan pocos pesos sobre el tronco congelado | Horas | Olvido localizado, sobreajuste | Dominio o tarea recurrente |
| Experto/bloque nuevo | Se crean parámetros y se aprende un módulo | Días o más | Router inestable y regresión global | Capacidad nueva que supera el presupuesto de adaptación |

La recomendación es que una consulta individual sólo pueda escribir en memoria provisional. La creación de parámetros debe ejecutarse por **lotes de aprendizaje versionados**, con un proceso de aprobación automático y humano cuando el impacto sea alto.

## 3. Arquitectura propuesta: Frozen Trunk + Growing Modules

Aethel 100B debería ser un sistema modular compuesto por un **trunk** relativamente estable y un conjunto de módulos ampliables.

### 3.1 Tronco base

El tronco contiene embeddings, bloques Transformer compartidos, normalización y una cabeza de salida. Se congela durante las expansiones normales. Para una variante MoE de 100B, el tronco puede aportar aproximadamente 10–20B de parámetros y los expertos el resto. La cifra exacta debe fijarse mediante un cálculo de arquitectura, no por redondeo comercial.

El tronco no debe reescribirse cada vez que llega información nueva. Esto conserva una referencia estable para comparar versiones, facilita rollback y reduce el riesgo de olvidar conocimiento general.

### 3.2 Banco de expertos ampliable

Cada capa MoE mantiene expertos versionados. Un controlador puede añadir un experto cuando la capacidad existente no explica una familia persistente de ejemplos. La expansión propuesta es:

1. Congelar el checkpoint aprobado `V_n` y registrar su hash.
2. Medir la demanda: pérdida residual por dominio, entropía del router, saturación de capacidad, frecuencia de recuperación fallida y repetición de errores.
3. Crear un experto nuevo a partir de una copia de un experto compatible, con una perturbación pequeña y reproducible o una inicialización especializada.
4. Inicializar el router con una masa de probabilidad baja para el experto nuevo, evitando que capture todo el tráfico.
5. Entrenar únicamente el experto nuevo, el router de transición y, si hace falta, adaptadores cercanos.
6. Mezclar replay general, datos nuevos y holdout no visto.
7. Comparar `V_n` contra `V_{n+1}` en calidad, latencia, VRAM, estabilidad y regresión por idioma.
8. Promover el módulo sólo si supera los umbrales; de lo contrario, archivarlo y revertirlo.

Este procedimiento crea parámetros en **momentos controlados**, no durante el forward de cada usuario. La ruta de inferencia debe ser determinista respecto a una versión publicada.

### 3.3 Memoria externa verificable

La memoria debe separarse en episódica y semántica. La episódica conserva eventos o interacciones con procedencia; la semántica conserva fragmentos deduplicados, embeddings y relaciones. Cada entrada debe tener fuente, fecha de ingestión, idioma, hash del contenido, nivel de confianza y política de caducidad.

RAG demuestra la utilidad de combinar memoria paramétrica con memoria no paramétrica para tareas intensivas en conocimiento [5]. En Aethel, la recuperación no debe convertirse en una autoridad ciega: el modelo debe distinguir evidencia recuperada, conocimiento paramétrico y deducción propia.

Una memoria creciente resuelve mejor el problema de **mantener más información** que añadir parámetros indiscriminadamente. Los parámetros deberían reservarse para regularidades, procedimientos y representaciones que se usan repetidamente y cuya incorporación justifique el coste de reentrenamiento.

## 4. Controlador de expansión

El controlador `CapacityManager` no aprende directamente del usuario. Produce una propuesta de expansión con evidencia y puntuación.

### 4.1 Señales

| Señal | Interpretación | Acción posible |
|---|---|---|
| Pérdida residual persistente por dominio | El modelo no absorbe una familia de patrones | Nuevo adaptador o experto |
| Alta entropía y baja separación del router | Los expertos no se especializan | Rebalanceo antes de expandir |
| Expertos saturados | Hay poca capacidad efectiva en una ruta | Aumentar capacidad o cambiar dispatch |
| Recuperación frecuente y consistente | La memoria externa cubre un hueco estable | Adaptador o destilación selectiva |
| Errores repetidos pese a evidencia correcta | Problema de procedimiento, no sólo de conocimiento | Herramienta, verificador o entrenamiento de tarea |
| Regresión en replay | La propuesta produce olvido | Rechazar o volver a entrenar |
| Aumento de latencia desproporcionado | El módulo no es eficiente | Podar, fusionar o no promover |

### 4.2 Regla de expansión

Una propuesta puede pasar a entrenamiento si se cumple, durante varias ventanas, una condición compuesta:

```text
expand =
    persistent_residual > threshold_residual
    AND retrieval_failure_rate > threshold_retrieval OR repeated_error_rate > threshold_error
    AND router_capacity_not_recoverable
    AND expected_gain_per_byte > minimum_gain
    AND budget_available
```

La expresión anterior es una política de diseño, no una métrica ya calibrada. Los umbrales deben estimarse en un conjunto de validación separado y congelarse antes de comparar variantes.

## 5. Cómo mejorar el razonamiento sin confiar sólo en parámetros

Más parámetros pueden aumentar capacidad, pero **no garantizan razonamiento**. El razonamiento útil requiere que el sistema pueda descomponer, verificar y corregir.

La ruta propuesta es un circuito de cómputo condicional:

| Ruta | Activación | Operación |
|---|---|---|
| Swift | Baja incertidumbre y tarea conocida | Una pasada corta con pocos expertos |
| Standard | Consulta normal | Inferencia completa del presupuesto estándar |
| Deliberate | Ambigüedad, matemáticas o planificación | Generación de candidatos y pasos intermedios internos |
| Verify | Resultado sensible o contradictorio | Comprobador, herramienta o segunda pasada independiente |

LayerSkip estudia layer dropout, pérdidas de salida temprana y auto-especulación, donde capas tempranas proponen y capas posteriores verifican [6]. En Aethel se debe probar como **early exit condicionado**, no asumir que abandonar capas siempre mantiene la calidad.

Para matemáticas básicas, la mejora más segura puede venir de un verificador determinista o una herramienta aritmética. Para hechos cambiantes, la recuperación con fuentes es más adecuada. Para razonamiento abstracto, se debe comparar deliberación con una baseline de la misma cantidad de FLOPs.

## 6. Estrategia de entrenamiento continuo

El ciclo de aprendizaje recomendado tiene seis zonas.

| Zona | Estado | Qué se permite |
|---|---|---|
| Inbox | Datos nuevos no aprobados | Validación, deduplicación y clasificación |
| Memory | Conocimiento consultable | Indexación con procedencia |
| Adaptation | Entrenamiento aislado | Adaptadores o experto candidato |
| Replay | Conservación de capacidades | Muestras EN/ES, matemáticas y casos difíciles |
| Evaluation | Decisión | Métricas y comparación contra checkpoint anterior |
| Promotion | Publicación interna | Crear versión inmutable y registro de cambios |

La actualización debe mantener una mezcla de datos antiguos y nuevos. El holdout debe permanecer fuera del entrenamiento. Se deben registrar hashes del corpus, tokenizador, configuración, checkpoint anterior y semillas.

La política de promoción mínima es:

```text
promote(V_next) iff
    quality_next >= quality_prev - allowed_regression
    AND bilingual_next >= bilingual_prev - bilingual_regression
    AND math_next >= math_prev - math_regression
    AND safety_next >= safety_prev
    AND p95_latency_next <= latency_budget
    AND memory_next <= memory_budget
    AND provenance_complete
```

Ningún módulo se promueve por reducir la pérdida de entrenamiento únicamente.

## 7. Ruta práctica desde Aethel actual hacia 100B

Un objetivo 100B no debe ser el primer experimento. La progresión recomendada es:

| Fase | Capacidad total orientativa | Objetivo técnico | Puerta de avance |
|---|---:|---|---|
| Edge | ~100M | Verificar datos, checkpoint, EN/ES y evaluación aislada | Holdout reproducible y contrato de reanudación |
| Small Pro | 1–3B | Validar MoE, router, memoria y early exit | Ganancia frente a dense con coste medido |
| Mid Pro | 7–13B | Validar expertos ampliables y aprendizaje continuo | Expansión sin regresión significativa |
| Large Pro | 30–70B | Validar paralelismo y especialización | Escala lineal razonable y estabilidad |
| Aethel-100B | 100B total | Operación distribuida y crecimiento versionado | Sólo después de pasar todas las puertas anteriores |

En esta ruta, el primer objetivo inteligente no es “llegar a 100B”, sino demostrar que un módulo nuevo aporta mejora por byte, por FLOP y por unidad de latencia. Si no lo demuestra en 1–13B, escalarlo a 100B sólo multiplica el coste del error.

## 8. Coste conceptual de 100B

Para 100B parámetros, una aproximación de memoria es:

| Elemento | Aproximación BF16/FP32 | Tamaño decimal aproximado |
|---|---:|---:|
| Pesos BF16 | 2 bytes por parámetro | 200 GB |
| Gradientes BF16 | 2 bytes por parámetro | 200 GB |
| Copia maestra FP32 | 4 bytes por parámetro | 400 GB |
| Momento Adam FP32 | 4 bytes por parámetro | 400 GB |
| Varianza Adam FP32 | 4 bytes por parámetro | 400 GB |
| Estado estático total | 16 bytes por parámetro | 1,6 TB |

La tabla no incluye activaciones, buffers de comunicación, fragmentación, checkpoints, KV-cache ni réplicas. Con GPUs de 80 GB, el límite teórico de pesos solos sería de 2.500 GPU, mientras que el estado estático de entrenamiento exigiría al menos 20.000 GPU equivalentes antes del overhead. Esas cifras son límites de almacenamiento, no una estimación de un clúster operativo.

El paralelismo tendría que combinar datos, tensor, pipeline, expertos y particionado de estados, siguiendo ideas de Megatron-LM y ZeRO [7] [8]. La comunicación entre expertos puede convertirse en el cuello de botella dominante. El sistema necesitaría interconexión de alta velocidad, tolerancia a fallos, checkpoints particionados y observabilidad distribuida.

## 9. Lo que sí y lo que no promete este diseño

**Sí permite diseñar:** un tronco de 100B, expertos que se añaden por versiones, memoria externa creciente, adaptadores especializados, rutas de cómputo variable y un protocolo de promoción.

**No demuestra todavía:** que Aethel razone como un genio, que sea bilingüe de forma nativa, que aprenda sin olvidar, que cree parámetros útiles automáticamente o que 100B sea más eficiente que un modelo pequeño bien entrenado.

La idea central es que Aethel tenga una **capacidad elástica gobernada**: memoria que puede crecer con bajo riesgo, módulos que pueden crecer con evaluación y pesos troncales que cambian lentamente. El modelo no debe aprender todo mediante gradiente ni guardar todo en parámetros.

## Referencias

[1]: https://arxiv.org/abs/1708.01547 "Lifelong Learning with Dynamically Expandable Networks"
[2]: https://arxiv.org/abs/1606.04671 "Progressive Neural Networks"
[3]: https://proceedings.neurips.cc/paper_files/paper/2020/hash/fdbe012e2e11314b96402b32c0df26b7-Abstract.html "Firefly Neural Architecture Descent"
[4]: https://arxiv.org/abs/2007.15353 "Growing Efficient Deep Networks by Structured Continuous Sparsification"
[5]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[6]: https://aclanthology.org/2024.acl-long.681/ "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
[7]: https://arxiv.org/abs/1909.08053 "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism"
[8]: https://arxiv.org/abs/1910.02054 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"
