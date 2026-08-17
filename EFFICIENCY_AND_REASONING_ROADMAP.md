# Aethel NextGen — Eficiencia, parámetros y razonamiento verificable

**Autor:** Manus AI  
**Estado:** diseño técnico y plan de experimentación; no constituye una afirmación de inteligencia humana o rendimiento de frontera.

## 1. Lenguajes y responsabilidades

| Lenguaje, runtime o formato | Evidencia del repositorio original | Responsabilidad alineada para Aethel |
|---|---|---|
| **Python + PyTorch** | README: investigación y entrenamiento; archivos de modelo, entrenamiento y `triton_kernels/`. | Núcleo, tokenización, entrenamiento, evaluación, memoria y exportación. |
| **Triton + CUDA** | `triton_kernels/fused_swiglu.py` contiene un kernel SwiGLU fusionado con fallback PyTorch. | Aceleración GPU opcional de operaciones críticas y futura actualización eficiente de El Líquido. |
| **Rust + Candle** | README y `rust_engine/` documentan el motor de inferencia de producción. | Runtime de inferencia exportado, multihilo y sin depender del intérprete Python. |
| **TypeScript / Node.js** | Pipeline de datos, evaluaciones, API y la interfaz original. | Ecosistema, evaluaciones, gateway y dashboard React/tRPC. |
| **TSX / React** | `src/App.tsx` y componentes de interfaz. | Chat, telemetría y controles; no ejecuta el cómputo del modelo. |
| **C++** | La interfaz original menciona una futura conexión nativa C++/Rust, pero esta clonación no incluye fuentes `.cpp`. | Posible capa nativa futura; no se presenta como motor implementado. |
| **C#** | No se encontraron archivos `.cs` ni una referencia textual verificable en la clonación auditada. | Se registra como tecnología a confirmar si existe otra rama o documento del repositorio que la especifique. |
| **SQL / Drizzle, Bash, JSONL y Markdown** | Capas añadidas por la plataforma de laboratorio actual. | Persistencia, lanzadores reproducibles, manifiestos, métricas y documentación. |

## 2. Principio de diseño

La eficiencia no se obtiene solo reduciendo parámetros. Aethel debe minimizar el producto de **tokens de entrenamiento × FLOPs por token × coste de memoria**, manteniendo una evaluación real de calidad. Las leyes de escalado de Chinchilla motivan entrenar modelos proporcionados al volumen de datos en vez de aumentar parámetros con datos insuficientes.[1]

La metáfora humana sirve únicamente como inspiración funcional. Las memorias episódica, semántica y de trabajo son módulos de software con estados acotados; no demuestran conciencia, comprensión humana ni aprendizaje autónomo general.

## 3. Parámetros: qué conservar, qué activar y qué adaptar

| Decisión | Beneficio esperado | Riesgo o condición | Estado en Aethel |
|---|---|---|---|
| **MoE disperso top-2** | Más capacidad total sin activar todos los expertos en cada token. | Requiere balanceo, capacidad de expertos y medición de tokens descartados. | Implementado; se registran carga, entropía y desbalance. |
| **GQA y KV-cache** | Reduce memoria de claves/valores durante generación. | Debe preservarse el contexto causal entre pasos. | Implementado; la materialización del prefill fue corregida y probada. |
| **Pesos de embeddings atados** | Elimina una matriz de salida duplicada. | Sólo es apropiado para el vocabulario compartido de entrada/salida. | Implementado. |
| **Adaptación de bajo rango** | Reduce parámetros entrenables durante adaptación de dominio. | No reemplaza el preentrenamiento; el rango y objetivos deben medirse. | Implementada de forma opcional en atención y expertos MoE; requiere comparación de calidad contra ajuste completo. |
| **Cuantización para ajuste fino** | Reduce memoria de la base durante adaptación. | Debe comprobarse estabilidad y calidad en los pesos de Aethel. | Investigación; no activada todavía. |

La arquitectura MoE se apoya en activación selectiva, cuyo potencial de escalado eficiente se ha estudiado en GLaM y en configuraciones de expertos más especializados.[2] [3] Para la adaptación de un modelo ya entrenado, LoRA y QLoRA son opciones de investigación más razonables que reentrenar todos los pesos para cada memoria o dominio.[7] [8]

## 4. Razonamiento de menor coste

El protocolo actual de Aethel es **recuperación → integración → predicción**. La recuperación vectorial limita el contexto inyectado a recuerdos relevantes; el Espacio de Trabajo Global mezcla explícitamente la ruta estable, la líquida y los recuerdos. Esto evita reinyectar una transcripción completa de la historia como si fuese memoria semántica.

| Mecanismo | Qué hace | Qué no se puede afirmar |
|---|---|---|
| **Memoria de trabajo** | Conserva un estado GRU por sesión. | No equivale a memoria humana ni garantiza razonamiento largo. |
| **Memoria episódica** | Recupera estados y tokens por similitud coseno ponderada. | No verifica la veracidad del texto que originó un recuerdo. |
| **Memoria semántica** | Fusiona prototipos vectoriales similares y registra observaciones. | No crea conceptos interpretables ni conocimiento validado por sí solo. |
| **Ciclo de Sueño** | Repite secuencias diversas y prioritarias durante entrenamiento. | No aprende fuera de la política y corpus autorizados. |
| **Trazabilidad** | Registra fuentes, similitud, selección y pesos de integración. | No revela ni pretende reproducir cadena de pensamiento interna. |

## 5. Mejora validada en este checkpoint

La caché KV de la atención ahora se materializa durante el **prefill** aunque no existan claves/valores previos. En la generación token a token, cada capa recibe el contexto acumulado en lugar de descartar el pasado. La prueba `engine/test_kv_cache.py` compara los logits de prefill y los logits de decodificación con los de una pasada causal completa; la equivalencia se comprobó con tolerancia numérica.

Esta corrección preserva el contexto autoregresivo y evita recomputar claves/valores previos en los pasos de generación. El microbenchmark reproducible `engine/benchmark_kv_cache.py`, ejecutado localmente en CPU con un prompt de 48 tokens y 48 tokens de decodificación, midió **517.54 tokens/s con caché** frente a **267.42 tokens/s sin caché**, equivalente a un factor de **1.935×** en esa configuración. El resultado es una línea base local, no una promesa de rendimiento en GPU ni una medición de VRAM.

## 6. Hoja de ruta priorizada

| Prioridad | Experimento | Métrica de aceptación | Decisión posterior |
|---|---|---|---|
| **P0 — completada** | KV-cache causal en prefill y decode. | Equivalencia de logits y longitud de caché creciente. | Mantener implementación. |
| **P1** | Baseline por familia: tokens/s, VRAM, pérdida, salud del router y acierto de memoria. | Reporte reproducible por GPU y corpus. | Tener una línea base antes de optimizar. |
| **P2** | Ajustar tamaño y granularidad de expertos MoE bajo presupuesto de FLOPs fijo. | Mejor pérdida/eficiencia, sin colapso del router. | Elegir top-1/top-2 y número de expertos por evidencia. |
| **P3** | Adaptadores LoRA en proyecciones de atención y/o expertos. | Calidad de adaptación frente a número de parámetros entrenables. | Adoptar sólo si supera o iguala ajuste completo dentro del presupuesto. |
| **P4** | Cabezas de predicción múltiple o decodificación especulativa. | Tokens/s y tasa de aceptación con igualdad de salida o calidad medida. | Activar sólo en inferencia si el modelo auxiliar justifica su coste. |
| **P5** | Bloques selectivos de espacio de estados para contexto largo. | Pérdida y coste por longitud de contexto. | Mantener Transformer/GQA si el híbrido no supera el baseline. |

Los bloques de espacio de estados selectivos son una hipótesis para secuencias largas, no un reemplazo automático del Transformer.[4] La predicción de varios tokens y la decodificación especulativa ofrecen rutas diferentes para eficiencia; ambas requieren medición de calidad, latencia y memoria bajo el mismo prompt y hardware.[5] [6]

## 7. Criterios científicos antes de afirmar una mejora

Toda mejora deberá informar corpus y licencia, semillas, configuración, parámetros totales y activos, FLOPs o proxy reproducible, longitud de contexto, precisión, hardware, latencia, VRAM, pérdida y resultados de evaluación. No se publicará una puntuación de razonamiento, inteligencia o benchmark si no existe una ejecución que produzca predicciones y el harness correspondiente.

## Referencias

[1] Hoffmann et al. (2022), [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556).

[2] Du et al. (2022), [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](http://proceedings.mlr.press/v162/du22c.html).

[3] Dai et al. (2024), [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066).

[4] Gu y Dao (2023), [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752).

[5] Gloeckle et al. (2024), [Better & Faster Large Language Models via Multi-token Prediction](https://arxiv.org/abs/2404.19737).

[6] Leviathan, Kalman y Matias (2023), [Fast Inference from Transformers via Speculative Decoding](https://proceedings.mlr.press/v202/leviathan23a.html).

[7] Hu et al. (2021), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).

[8] Dettmers et al. (2023), [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
