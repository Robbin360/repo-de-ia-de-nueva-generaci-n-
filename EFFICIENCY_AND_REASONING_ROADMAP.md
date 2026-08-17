# Aethel NextGen — Eficiencia, parámetros y razonamiento verificable

**Autor:** Manus AI  
**Estado:** diseño técnico y plan de experimentación; no constituye una afirmación de inteligencia humana o rendimiento de frontera.

## 1. Lenguajes y responsabilidades

| Lenguaje o formato | Uso en Aethel | Razón de ingeniería |
|---|---|---|
| **Python** | Núcleo PyTorch, tokenización, entrenamiento, evaluación, exportación de artefactos y pruebas del motor. | Es el ecosistema de investigación y entrenamiento de aprendizaje profundo del proyecto. |
| **TypeScript** | API tRPC, contratos de datos, cálculo analítico de presupuestos y pruebas del servidor. | Mantiene tipos coherentes entre servidor y cliente. |
| **TSX / React** | Dashboard, chat, telemetría y controles de entrenamiento. | Permite presentar al usuario capacidades y límites sin inventar datos. |
| **SQL / Drizzle** | Historial persistente del chat y datos estructurados del dashboard. | Separa metadatos consultables de artefactos de modelo. |
| **Bash** | Lanzadores reproducibles para Kaggle y GPU persistente. | Estandariza el entorno y reduce errores operativos. |
| **CSS / HTML** | Sistema visual y accesibilidad del dashboard. | No participa en el cómputo del modelo. |
| **JSON / JSONL / Markdown** | Manifiestos de corpus, trazas de memoria, métricas y documentación. | Dejan auditable el ciclo de datos y las memorias sin almacenar afirmaciones textuales falsas. |

## 2. Principio de diseño

La eficiencia no se obtiene solo reduciendo parámetros. Aethel debe minimizar el producto de **tokens de entrenamiento × FLOPs por token × coste de memoria**, manteniendo una evaluación real de calidad. Las leyes de escalado de Chinchilla motivan entrenar modelos proporcionados al volumen de datos en vez de aumentar parámetros con datos insuficientes.[1]

La metáfora humana sirve únicamente como inspiración funcional. Las memorias episódica, semántica y de trabajo son módulos de software con estados acotados; no demuestran conciencia, comprensión humana ni aprendizaje autónomo general.

## 3. Parámetros: qué conservar, qué activar y qué adaptar

| Decisión | Beneficio esperado | Riesgo o condición | Estado en Aethel |
|---|---|---|---|
| **MoE disperso top-2** | Más capacidad total sin activar todos los expertos en cada token. | Requiere balanceo, capacidad de expertos y medición de tokens descartados. | Implementado; se registran carga, entropía y desbalance. |
| **GQA y KV-cache** | Reduce memoria de claves/valores durante generación. | Debe preservarse el contexto causal entre pasos. | Implementado; la materialización del prefill fue corregida y probada. |
| **Pesos de embeddings atados** | Elimina una matriz de salida duplicada. | Sólo es apropiado para el vocabulario compartido de entrada/salida. | Implementado. |
| **Adaptación de bajo rango** | Reduce parámetros entrenables durante adaptación de dominio. | No reemplaza el preentrenamiento; el rango y objetivos deben medirse. | Siguiente experimento propuesto. |
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
