# Aethel: hoja de ruta para más capacidad por unidad de cómputo

**Autor:** Manus AI  
**Estado:** propuesta experimental, no validada en GPU  
**Fecha:** 2026-08-28  
**Relación:** complementa `AETHEL_PRO_SPEC.md`; no sustituye la evaluación del checkpoint Edge.

## 1. Tesis de diseño

La meta no debe ser construir el modelo con más parámetros, sino maximizar la **capacidad útil por token, joule, byte de memoria y segundo de latencia**. Un sistema más inteligente por unidad de coste debería gastar poco en tareas fáciles, recuperar conocimiento fuera de los pesos cuando sea conveniente y escalar su cómputo sólo cuando la dificultad o la incertidumbre lo justifiquen.

La regla operativa será:

> **No se incorpora una técnica por ser sofisticada. Se incorpora sólo si mejora una métrica objetivo sin romper calidad, estabilidad, reproducibilidad o coste.**

Aethel mantendrá tres planos separados. El **núcleo paramétrico** aprenderá representaciones y transformaciones lingüísticas; el **plano de memoria** manejará información recuperable y adaptable sin reentrenar todo el modelo; y el **plano de control** decidirá cuánto cómputo, qué expertos y qué herramientas se permiten en cada entrada. Esta separación evita forzar a los pesos a almacenar todo el conocimiento y evita activar todos los módulos en cada token.

## 2. Diagnóstico del punto de partida

El motor actual ya contiene RoPE, GQA, MoE top-2, BF16, caché KV, módulos de memoria y contratos de Triton. La implementación, sin embargo, todavía bloquea el modo Triton estricto cuando no existe un kernel validado para prefill causal o dispatch/combine completo de MoE. Por eso, **la primera oportunidad no es añadir más módulos**, sino medir y reemplazar los caminos de referencia PyTorch con kernels correctos, manteniendo una prueba de equivalencia numérica.

| Área | Situación observada | Riesgo | Métrica que decide |
|---|---|---|---|
| Atención | GQA y KV-cache presentes; prefill Triton estricto aún no validado | El modelo puede ser eficiente en decode pero lento en prefill | tokens/s de prefill, latencia p50/p95 y error numérico |
| MoE | Top-2, balanceo y jitter de selección presentes; dispatch/combine de referencia | El coste real puede acercarse al de expertos densos por overhead de Python | FLOPs activos, tokens/s, ocupación, carga máxima por experto |
| Profundidad | Las capas se ejecutan todas | Se desperdicia cómputo en tokens fáciles | calidad a 25/50/75/100% de profundidad y coste/token |
| Memoria | Hay módulos cognitivos; la recuperación verificable debe aislarse | Actualizaciones no controladas o contaminación de evaluación | exactitud con/sin memoria, trazabilidad y regresión |
| Precisión | BF16 de entrenamiento; cuantización aún debe medirse | Ahorrar VRAM puede degradar router, norm o logits | pérdida, benchmarks y latencia por formato |
| Datos | Corpus Edge bilingüe preparado; calidad Pro aún no probada | Más tokens de baja calidad pueden empeorar la muestra eficiente | pérdida por idioma, deduplicación, contaminación y calidad por fuente |

## 3. Arquitectura propuesta: el sistema de presupuesto adaptativo

Aethel Pro debería tener un **controlador de presupuesto** antes de ejecutar el camino completo. Este controlador recibe señales baratas: longitud, entropía de los logits preliminares, idioma estimado, incertidumbre del router, presencia de patrones matemáticos/código y necesidad de recuperación. Produce un presupuesto discreto, no una decisión libre sin límites:

| Modo | Profundidad | Expertos | Memoria/herramientas | Uso |
|---|---:|---:|---|---|
| `swift` | 25–40% | 1 experto o ruta densa pequeña | sin recuperación | saludo, clasificación y continuación fácil |
| `standard` | 60–75% | top-2 | recuperación opcional | conversación y conocimiento común |
| `deliberate` | 100% | top-2 o top-2 con segundo pase | workspace/verificador | matemáticas, contradicciones y tareas difíciles |
| `verify` | variable | presupuesto residual | comprobador aislado | comprobar una respuesta antes de mostrarla |

El controlador debe estar **calibrado por tarea**, no sólo por confianza token a token. Una confianza alta no garantiza una respuesta correcta en matemáticas o hechos recientes. La decisión de escalar debe poder basarse en una prueba barata, como un verificador de formato, una consulta de recuperación o una segunda muestra concordante.

## 4. Siete palancas de eficiencia e inteligencia

### 4.1 Profundidad adaptativa con salida temprana y auto-especulación

Entrenar salidas auxiliares compartidas en varias profundidades permitiría que las entradas sencillas salgan antes. Para entradas difíciles, las capas tempranas pueden proponer varios tokens y las capas restantes verificarlos, siguiendo la idea de LayerSkip [1]. La hipótesis es que una parte significativa de los tokens de conversación no necesita la profundidad completa.

La primera versión debe usar tres salidas —40%, 70% y 100%— y una política conservadora: si la salida temprana no supera un umbral calibrado en el conjunto de desarrollo de la misma tarea, continuar. La ganancia se acepta sólo si el modelo mantiene la calidad dentro de un margen prefijado y reduce la media ponderada de FLOPs.

### 4.2 Atención latente como segunda generación de GQA

GQA reduce el número de cabezas K/V y ya forma parte de Aethel. La siguiente ablation es una atención latente, inspirada en MLA, que comprime el estado K/V en un vector latente antes de almacenarlo [2]. No debe sustituirse GQA de inmediato: se deben comparar `MHA`, `GQA-8`, `GQA-4` y `MLA` con el mismo número de capas, datos y semillas.

La hipótesis es especialmente relevante para contextos largos y dispositivos Edge. El criterio no será sólo la VRAM: una reducción de caché que aumente demasiado el error o el tiempo de compresión no es una mejora.

### 4.3 MoE con especialistas útiles, no con expertos decorativos

El MoE debe ser un mecanismo de **capacidad condicional**. Switch Transformers muestra la lógica de activar distintos parámetros por entrada, pero también identifica comunicación, complejidad y estabilidad como obstáculos [3]. Para Aethel se propone:

1. Mantener top-2 como baseline y medir top-1, top-2 y top-2 con capacidad limitada.
2. Añadir una pérdida de carga sólo si no distorsiona la señal semántica del router.
3. Registrar carga por idioma, tarea y longitud, no sólo la media global.
4. Usar grouped GEMM/dispatch fusionado sólo después de demostrar equivalencia con la referencia PyTorch.
5. Penalizar colas y expertos saturados, no forzar uniformidad perfecta: algunos especialistas pueden ser legítimamente más demandados.

Una mejora fuera de la caja sería un **router jerárquico**: un selector barato decide una familia de expertos y un segundo selector elige dentro de ella. Esto podría reducir comparaciones cuando el número de expertos crezca, pero se descarta si empeora balance, latencia o transferencia entre dispositivos.

### 4.4 Memoria externa verificable y pesos especializados

El conocimiento cambiante no debería exigir actualizar todos los pesos. Aethel puede usar recuperación híbrida: índice semántico para localizar candidatos, filtro léxico para nombres/números y un reranker pequeño para seleccionar evidencia. RAG combina generación paramétrica con recuperación externa [4]; su utilidad debe medirse con preguntas con evidencia, preguntas sin evidencia y preguntas adversariales.

La memoria episódica no debe escribir directamente sobre el núcleo durante la inferencia. Cada entrada debe incluir origen, fecha, hash del contenido, confianza y política de caducidad. Las memorias contradictorias deben conservarse como conflicto, no resolverse silenciosamente.

Para adaptación personal o de dominio, la primera opción debe ser un módulo delta pequeño —por ejemplo LoRA/adapters— cargable y descargable. El núcleo congelado reduce riesgo de olvido catastrófico y permite mantener varias personalidades o dominios sin duplicar Pro completo.

### 4.5 Dos sistemas: respuesta rápida y deliberación escalable

En lugar de obligar a Pro a razonar profundamente siempre, se propone un **circuito dual**. Un camino Swift responde rápidamente. Un camino Deliberate recibe sólo casos escalados y genera un estado de trabajo interno no expuesto automáticamente. Un verificador independiente revisa consistencia, cálculos y citas.

Esto no crea razonamiento por decreto. La hipótesis sólo se considera válida si `deliberate + verify` mejora exactitud en tareas difíciles con un coste medio aceptable y sin aumentar alucinaciones en tareas fáciles. Las trazas de razonamiento no deben publicarse como si fueran prueba de proceso interno fiable.

### 4.6 Destilación de capacidad hacia Edge

Pro puede actuar como maestro para mejorar Edge, pero la destilación debe transferir conductas evaluadas, no autoridad ciega. Se propone destilar logits, respuestas verificadas, selección de profundidad y decisiones de recuperación. Las respuestas sintéticas deben pasar filtros de exactitud y diversidad; nunca deben confundirse con evidencia humana.

La prueba decisiva es un conjunto bloqueado que compare Edge original, Edge destilado y Pro. Si Edge gana en coste pero pierde demasiado en bilingüismo, matemáticas o robustez, se conserva como variante diferente y no se declara reemplazo.

### 4.7 Cuantización mixta guiada por sensibilidad

No todos los tensores merecen la misma precisión. Se propone mantener router, embeddings, RMSNorm, acumulaciones y logits en mayor precisión, y cuantizar lineales menos sensibles a INT8; INT4 sólo debe evaluarse después. SmoothQuant es una referencia de cuantización post-entrenamiento a 8 bits [5], pero sus resultados no predicen los de Aethel.

La asignación debe ser por capa y módulo, usando un conjunto de calibración bilingüe que incluya números. Se acepta una configuración sólo si reduce memoria y mantiene la pérdida, exactitud y estabilidad del router dentro de márgenes medidos.

## 5. La capa de sistemas: CUDA, Triton y C++ sin convertirlos en fetiches

CUDA, Triton o C++ no vuelven eficiente a un modelo por sí mismos. La secuencia correcta es: perfilado, baseline correcto, kernel aislado, comparación numérica, prueba de estrés y sólo después integración. El orden recomendado es:

| Prioridad | Kernel o cambio | Razón |
|---:|---|---|
| 1 | fused RMSNorm + proyección cuando sea seguro | eliminar lecturas/escrituras intermedias |
| 2 | dispatch/combine MoE agrupado | reducir overhead y aprovechar GEMM agrupado |
| 3 | prefill causal Triton validado | atacar la ruta actualmente bloqueada en modo estricto |
| 4 | KV-cache paginada y atención decode | mejorar sesiones largas y concurrencia |
| 5 | CUDA Graphs para formas estables | reducir overhead de lanzamiento en decode |
| 6 | kernels específicos C++/CUDA sólo donde el perfil lo justifique | controlar casos que Triton no cubra bien |

Cada kernel debe tener una prueba de equivalencia con tolerancias explícitas, una prueba de determinismo cuando corresponda y una ruta de fallback sólo en desarrollo. El modo de producción no debe anunciar Triton si internamente usa un bucle PyTorch.

## 6. Entrenamiento: más calidad por token, no sólo más tokens

La eficiencia de datos debe atacar la calidad por token. El corpus Pro debería aplicar deduplicación aproximada, filtros de idioma, detección de texto corrupto, mezcla equilibrada EN/ES, validación separada por fuente y un pequeño conjunto de matemáticas con trazas verificables. La mezcla debe ser una variable de experimento, no una intuición fija.

Se recomienda un currículo en cuatro etapas: lenguaje bilingüe limpio; conocimiento y documentos; matemáticas/código con verificación; y destilación/adaptación. Cada etapa debe conservar un checkpoint completo y un recibo de datos. No se reanuda una corrida cuyo manifiesto o tokenizer no coincida.

## 7. Ablations y criterios de rechazo

El experimento principal debe comenzar con un baseline pequeño, reproducible y barato. Cada ablation cambia una sola variable o una combinación declarada, usa la misma semilla y se evalúa en un holdout bloqueado.

| Experimento | Comparación | Éxito mínimo propuesto |
|---|---|---|
| `depth-adaptive-v1` | profundidad fija vs salida temprana | ≥20% menos FLOPs medios sin pérdida >0,5% relativa en tareas fáciles |
| `mla-vs-gqa-v1` | GQA-8/GQA-4/MLA | ≥30% menos KV-cache sin degradación significativa en calidad |
| `moe-dispatch-v1` | referencia PyTorch vs kernel fusionado | ≥1,5× tokens/s de MoE con error numérico dentro de tolerancia |
| `retrieval-gate-v1` | sin RAG vs RAG sólo cuando hace falta | mayor exactitud factual sin degradar preguntas cerradas |
| `dual-path-v1` | siempre profundo vs escalado | menor coste medio con mejora en el subconjunto difícil |
| `mixed-int8-v1` | BF16 vs cuantización por sensibilidad | ≥35% menos memoria con pérdida y router estables |
| `edge-distill-v1` | Edge original vs destilado | mejor calidad/coste en al menos dos familias de tareas |

Los umbrales son **criterios de diseño**, no resultados actuales. Se deben ajustar después de establecer la varianza de las métricas. Un experimento se rechaza si sólo mejora una métrica sintética mientras empeora calidad, reproducibilidad, seguridad o coste total.

## 8. Orden de ejecución recomendado

Primero se debe medir el baseline Edge/Pro en CPU y en una GPU disponible, incluyendo prefill, decode, memoria, pérdida y router. Después se implementan profundidad adaptativa y recuperación con módulos aislados, porque ofrecen valor potencial sin reescribir todo el núcleo. En paralelo se puede construir el harness de kernels, pero no se debe activar el modo estricto hasta validar prefill y dispatch.

La segunda ola compara MLA contra GQA y prueba cuantización mixta. La tercera ola aborda router jerárquico, circuito dual y destilación. El entrenamiento largo sólo tiene sentido después de que el pipeline de evaluación pueda distinguir una mejora real de una variación de semilla.

## 9. Qué significaría “más en menos”

Aethel podría considerarse **más eficiente**, de forma limitada y específica, si entrega una calidad determinada con menos FLOPs, VRAM, latencia o energía que su baseline bajo el mismo protocolo. Podría considerarse **más capaz**, también de forma limitada, si mejora pruebas bloqueadas de comprensión EN/ES, matemáticas, recuperación factual y robustez.

No se debe convertir estos resultados en afirmaciones de AGI, conciencia, IQ humano, bilingüismo nativo o razonamiento general. La arquitectura puede ser prometedora y seguir siendo experimental. La ventaja real será una gráfica reproducible de calidad contra coste, no una etiqueta grandiosa.

## Referencias

[1]: https://aclanthology.org/2024.acl-long.681/ "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
[2]: https://arxiv.org/abs/2405.04434 "DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model"
[3]: https://jmlr.org/papers/v23/21-0998.html "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity"
[4]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[5]: https://arxiv.org/abs/2211.10438 "SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models"
