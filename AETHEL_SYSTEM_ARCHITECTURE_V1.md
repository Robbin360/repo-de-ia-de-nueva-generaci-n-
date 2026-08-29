# Aethel: arquitectura integral y principio de funcionamiento

**Versión:** 1.0  
**Estado:** arquitectura implementada experimentalmente; capacidades de producción y claims de inteligencia general no demostrados  
**Autor:** Manus AI  
**Fecha:** 2026-08-29

## 1. Propósito

Aethel es una plataforma experimental para construir un modelo causal bilingüe EN/ES con una arquitectura que combina un Transformer eficiente, rutas cognitivas auditables y aprendizaje posterior gobernado. La idea central no es añadir complejidad por estética, sino separar tres problemas que normalmente se mezclan: **representar lenguaje**, **mantener estado útil** y **adquirir habilidades nuevas sin destruir las anteriores**.

La arquitectura se llama bio-inspirada porque utiliza metáforas funcionales —Sólido, Líquido, Sueño, curiosidad y espacio de trabajo— para organizar mecanismos computacionales concretos. No implica consciencia, emociones, voluntad ni vida biológica.

> **Contrato de evidencia:** todo lo que esté marcado como implementado debe poder localizarse en código, ejecutarse en una prueba o aparecer en un recibo reproducible. Todo lo que esté marcado como hipótesis debe tener una ablación, una métrica y un criterio de rechazo antes de promoverse.

## 2. La idea revolucionaria, expresada de forma comprobable

La innovación propuesta por Aethel no es una única capa mágica. Es una **arquitectura de capacidad separable**: un núcleo lingüístico estable, un estado plástico de corto plazo, memorias externas con procedencia, cálculo adaptativo bajo presupuesto y módulos de expansión que sólo se convierten en parámetros después de pasar evaluación de regresión.

Este diseño intenta maximizar calidad por parámetro, FLOP y byte. En vez de obligar a todo el modelo a recordar cada hecho, separa conocimiento estable, experiencias recuperables y habilidades especializadas. En vez de ejecutar el máximo razonamiento en cada token, propone reservar refinamiento adicional para estados que un router de dificultad seleccione. En vez de modificar pesos durante una conversación, convierte el aprendizaje en un artefacto versionado que puede aceptarse o descartarse.

Estas ideas son **programas de investigación**, no ventajas demostradas. El repositorio contiene contratos y sondas para medirlas; una implementación experimental no prueba que superen a un Transformer denso.

## 3. Capas del sistema

| Capa | Responsabilidad | Implementación actual | Evidencia requerida para promoverla |
|---|---|---|---|
| Interfaz | Chat, telemetría, configuración y transparencia | React, TypeScript, Tailwind, tRPC y Express | Pruebas de UI y estado real del runtime |
| Orquestación | Lanzar, detener, inspeccionar y exportar trabajos | Runner PyTorch, contratos de entrenamiento y exportación | Recibo de proceso y artefactos persistentes |
| Núcleo lingüístico | Predicción causal de tokens | Transformer con RoPE, GQA y MoE top-2 | Pérdida, holdout y generación controlada |
| Estado cognitivo | Integrar estado estable, plástico y recuperado | La Roca, El Líquido, memoria, workspace y GRU | Contratos CPU/CUDA y trazabilidad |
| Consolidación | Seleccionar experiencias sin mutar el núcleo automáticamente | Sueño, replay y preflight de admisión | No colisión con holdout y aprobación explícita |
| Aceleración | Reducir memoria y coste por token | BF16/FP16, KV-cache, SwiGLU fusionado, Triton experimental | Paridad numérica y benchmark en GPU objetivo |
| Expansión | Añadir habilidades o capacidad con control | LoRA y refinamiento adaptativo opt-in | Mejora nueva menos regresión y versión reproducible |

## 4. Flujo de inferencia

En cada entrada tokenizada, el modelo calcula una observación compacta promediando los embeddings. La memoria de trabajo actualiza esa observación mediante una GRU y la combina con tres rutas: la representación estable de **La Roca**, la traza plástica de **El Líquido** y los estados recuperados de memoria episódica/semántica.

El **Espacio de Trabajo Global** calcula pesos para esas tres fuentes y produce el contexto que se inyecta al Transformer. Si el refinamiento adaptativo está habilitado, un predictor de dificultad selecciona algunos estados y aplica pasos GRU adicionales; la selección, el porcentaje de estados refinados y los pasos efectivos quedan registrados como telemetría.

El Transformer ejecuta atención causal con RoPE y GQA. En el bloque feed-forward, el router selecciona los dos expertos activos por token entre ocho candidatos. La salida de los expertos se combina, se calculan las pérdidas auxiliares del router y se emiten métricas de cobertura, concentración, entropía y overflow cuando están disponibles.

Después de la predicción se calculan prioridad neuromoduladora, sorpresa y señales de curiosidad. La curiosidad sólo puede producir `observe_only`, `retrieve_local`, `ask_clarification` o `propose_replay`; no puede ejecutar acciones externas, crear optimizadores ni iniciar entrenamiento.

```text
entrada EN/ES
    ↓
tokenizador BPE versionado
    ↓
embeddings → memoria de trabajo
    ↓
La Roca ─┐
El Líquido ├─→ Espacio de Trabajo Global ─→ refinamiento opcional
episódico/semántico ┘                         ↓
                                      Transformer causal
                                  RoPE + GQA + MoE top-2
                                                ↓
                                      logits y telemetría
                                                ↓
                  curiosidad / observación / candidato de replay
```

## 5. El núcleo lingüístico

El núcleo `AethelModel` es causal: cada posición sólo puede utilizar información disponible a su izquierda. RoPE codifica posición mediante rotaciones en las consultas y claves. GQA comparte cabezas de clave/valor para reducir memoria de KV-cache respecto a atención multi-cabeza completamente independiente. MoE mantiene varios expertos feed-forward, pero activa sólo top-2 por token.

El MoE es una apuesta de eficiencia condicionada. El número total de parámetros puede ser grande mientras el número activo por token es menor, pero el routing añade coste de selección, dispatch, combinación y posibles desequilibrios. Por eso Aethel registra salud del router y no declara eficiencia sólo por contar parámetros activos.

| Mecanismo | Beneficio esperado | Riesgo | Control |
|---|---|---|---|
| RoPE | Mejor señal posicional relativa | Extrapolación fuera de contexto | Evaluar longitudes no vistas |
| GQA | Menor KV-cache | Menor capacidad por cabeza KV | Comparar calidad/latencia con baseline |
| MoE top-2 | Más capacidad total con cómputo activo reducido | Colapso, overflow y comunicación | Cobertura, concentración, entropía y overflow |
| SwiGLU | No linealidad expresiva | Coste de tres proyecciones | Medir tokens/s y memoria |
| BF16/FP16 | Menor memoria y mayor throughput | Inestabilidad o pérdida de precisión | Smoke test, NaN gate y scaler |

## 6. Los módulos cognitivos

### 6.1 La Roca

La Roca es la ruta estable: una proyección aprendida y un ancla no entrenable que representan identidad y conocimiento base. Su función es evitar que todo el estado del sistema sea plástico. No es una memoria perfecta ni una garantía contra el olvido; es una superficie estable que debe cambiar lentamente mediante releases evaluados.

### 6.2 El Líquido

El Líquido mantiene una traza Hebbiana con decaimiento. `observe()` normaliza el estado, aplica saliencia limitada a `[0,1]`, incrementa una versión y puede escribir `liquid_versions.jsonl`. La traza se actualiza en una frontera explícita y no debe confundirse con entrenamiento del núcleo.

El diseño de crecimiento dinámico usa El Líquido como detector de patrones persistentes, no como permiso para escribir parámetros. Una señal líquida puede proponer que una habilidad merece un adaptador; sólo el pipeline offline, con datos y holdout, puede aprobarlo.

### 6.3 Memoria episódica

La memoria episódica guarda estados, tokens y saliencia en JSONL. La recuperación usa similitud coseno y combina los registros más cercanos mediante pesos softmax. Conserva procedencia y capacidad acotada. Es adecuada para experiencias y contexto recuperable, no para afirmar que el núcleo aprendió esos hechos.

### 6.4 Memoria semántica

La memoria semántica mantiene prototipos vectoriales. Si un estado nuevo supera el umbral de similitud, se fusiona con el prototipo; de lo contrario, crea un registro hasta alcanzar su capacidad. No genera etiquetas ni texto por sí misma. Su utilidad debe medirse con y sin recuperación, manteniendo el mismo checkpoint.

### 6.5 Memoria de trabajo y Espacio de Trabajo Global

La memoria de trabajo es una GRU que se reinicia explícitamente por sesión. El workspace recibe las salidas de Roca, Líquido y memoria recuperada, produce pesos interpretables y genera el contexto top-down que alimenta el Transformer. Los pesos del workspace son telemetría de mezcla, no una explicación causal completa de la respuesta.

### 6.6 Neuromodulación y curiosidad

La neuromodulación calcula prioridad y sorpresa a partir de la observación y, cuando existe, la pérdida. El controlador de curiosidad combina incertidumbre, novedad, contradicción, progreso esperado, riesgo y coste. Bloquea señales de alto riesgo y exige progreso demostrable antes de proponer replay.

El protocolo es deliberadamente conservador: cuando hay incertidumbre sin evidencia de que el error sea aprendible, Aethel observa, recupera memoria o pide aclaración. No debe convertir novedad en entrenamiento automático.

### 6.7 Ciclo de Sueño

Sueño es un sistema de consolidación y replay. Recibe estados y tokens, calcula una firma aproximada de secuencia, conserva diversidad y prioriza registros salientes. `sample_pairs()` devuelve pares autoregresivos reales del buffer. El preflight de Sueño impide usar holdout, iniciar entrenamiento o promover candidatos sin aprobación.

## 7. Entrenamiento y reanudación

El entrenamiento base usa corpus con procedencia, tokenizador BPE derivado sólo del split de entrenamiento, holdout separado, precisión mixta y checkpoints portátiles. Cada checkpoint debe incluir pesos, optimizador, scheduler, scaler si aplica, paso global, RNG, estado runtime NextGen, contrato de configuración, hashes del tokenizador y del manifiesto.

El contrato `aethel-training-resume/v2` bloquea la reanudación si cambia la arquitectura, longitud, precisión, estrategia distribuida, batch global, horizonte, tokenizer, semilla contractual o datos. La razón es científica: una continuación con otro contrato es una nueva corrida, no una reanudación fiel.

La campaña recomendada comienza con `pilot-100m`: 97,16M parámetros totales, 40,53M activos aproximados, 4 capas, dimensión 512, contexto 1024, 8 expertos y top-2. El modelo 300M tiene 344,34M parámetros totales con el preset actual y sólo debe intentarse después de una puerta de calidad 100M.

## 8. Evolución controlada de capacidad

Aethel distingue cuatro tipos de “aprendizaje”:

| Tipo | Cambia pesos del núcleo | Persistencia | Ejemplo |
|---|---:|---|---|
| Contexto | No | Una sesión | Memoria de trabajo |
| Memoria externa | No | JSONL/vectorial | Episodio recuperable |
| Adaptador | No, si la base queda congelada | Release separado | `math-log-v1` con LoRA |
| Expansión del núcleo | Sí | Nuevo checkpoint | Expertos o bloques promovidos |

El flujo para una capacidad nueva es: detectar errores repetidos, agruparlos, demostrar que el patrón persiste, curar datos con procedencia, entrenar un candidato aislado, medir generalización y regresión, y promover sólo si el beneficio neto supera el coste. La creación de parámetros en caliente durante conversación queda prohibida porque impide reproducibilidad y facilita contaminación de datos.

Una expansión de expertos puede ser útil cuando una habilidad es frecuente, especializada y no cabe en un adaptador. Antes de añadirla se debe demostrar que el adaptador no basta, que el router puede distribuir carga y que el nuevo checkpoint retiene EN/ES. El tamaño nuevo debe estar versionado y tener un camino de rollback.

## 9. Ultra-eficiencia: programa de investigación

La eficiencia de Aethel se mide en el mismo hardware, longitud, batch y precisión. Las métricas mínimas son tokens/s, latencia p50/p95, memoria máxima, pérdida por token y calidad funcional. Un mecanismo sólo se conserva si reduce coste sin degradar calidad más allá del umbral predefinido.

Las principales hipótesis son: MoE reduce cómputo activo; GQA reduce memoria de inferencia; KV-cache reduce recomputación autoregresiva; refinamiento adaptativo concentra cálculo en casos difíciles; LoRA reduce parámetros entrenables; Triton puede reducir overhead de atención y dispatch. Ninguna debe declararse ganadora sin un benchmark de ablación.

La ruta de aceleración es Python/PyTorch para investigación y entrenamiento, CUDA/Triton para kernels validados, Rust/Candle como posible runtime de inferencia y TypeScript/Node.js para interfaz y orquestación. C++ y C# permanecen opcionales mientras no exista una ruta implementada y medida.

## 10. Seguridad científica y límites

Aethel no debe presentar respuestas del LLM de plataforma como respuestas del checkpoint propio. El dashboard declara cuándo sólo existe una configuración teórica, cuándo no hay checkpoint y cuándo el motor no está conectado. Los benchmarks permanecen vacíos hasta cargar predicciones reales.

No se afirma consciencia, AGI, IQ 300, razonamiento general fuerte ni bilingüismo nativo sin pruebas. La métrica principal de progreso será un conjunto reproducible de resultados por idioma, familia de tarea, coste y regresión. El fracaso de una hipótesis también es un resultado válido si queda documentado.

## 11. Mapa del repositorio

| Área | Archivos principales |
|---|---|
| Modelo causal | `engine/aethel_model.py`, `engine/aethel_nextgen.py` |
| Entrenamiento | `engine/train_aethel_gpu.py`, `engine/train_nextgen.py`, `engine/train_real.py` |
| Reanudación | `engine/aethel_resume.py`, `engine/test_resume_contract.py` |
| Datos | `engine/prepare_bilingual_corpus.py`, `training/aethel_edge_v1.manifest.json` |
| Presupuesto | `engine/report_model_budget.py` |
| Router | `engine/router_health.py`, `engine/router_assignment_health.py`, `engine/router_auxiliary.py` |
| Memoria/Sueño | `engine/sleep_*.py`, `engine/aethel_nextgen.py` |
| Evaluación | `engine/evaluate_nextgen.py`, `engine/evaluate_benchmarks.py` |
| GPU | `engine/triton_bridge.py`, `engine/test_triton_gpu.py` |
| Plan base | `AETHEL_BASE_CAPABILITY_SPEC.md`, `AETHEL_BASE_TRAINING_PLAN_100M_300M.md` |
| Evidencia | `training/`, `research/`, `todo.md` |

## Referencias

[1]: ./AETHEL_BASE_CAPABILITY_SPEC.md "Contrato de capacidad base"
[2]: ./AETHEL_BASE_TRAINING_PLAN_100M_300M.md "Plan de entrenamiento 100M–300M"
[3]: ./AETHEL_PRO_SPEC.md "Especificación hipotética Aethel Pro"
[4]: ./training/AETHEL_COGNITIVE_EXPERIMENT_CONTRACT_V1.md "Contrato de experimentos cognitivos"
[5]: ./training/AETHEL_TRITON_CUDA_ACCEPTANCE_MATRIX_V1.md "Matriz de aceptación CUDA/Triton"
[6]: ./training/AETHEL_TRAINING_RESUME_CONTRACT_V1.md "Contrato de reanudación"
[7]: https://arxiv.org/abs/1706.03762 "Attention Is All You Need"
[8]: https://arxiv.org/abs/2104.09864 "RoFormer: Enhanced Transformer with Rotary Position Embedding"
[9]: https://arxiv.org/abs/2006.16668 "GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding"
