# Fuentes primarias para eficiencia de Aethel

Estas referencias se usarán sólo para fundamentar mecanismos conocidos; no prueban que Aethel obtenga las mismas ventajas.

1. **GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints** — https://arxiv.org/abs/2305.13245. Describe Grouped-Query Attention, donde varios grupos de consultas comparten cabezas de clave/valor, con implicaciones para coste de decodificación y KV-cache.
2. **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** — https://jmlr.org/papers/v23/21-0998.html. Presenta una familia de modelos dispersos con routing MoE y parámetros seleccionados por entrada; sus resultados no se transfieren automáticamente a Aethel.
3. **RoFormer: Enhanced Transformer with Rotary Position Embedding** — https://arxiv.org/abs/2104.09864. Fuente primaria del mecanismo RoPE usado como referencia posicional en la especificación.
4. **PyTorch Automatic Mixed Precision package** — https://pytorch.org/docs/stable/amp.html. Documentación oficial para la ruta BF16/FP16 con autocast y GradScaler cuando corresponda.

Nota: la propuesta de eficiencia debe comparar cada cambio contra un baseline comparable y registrar VRAM, tokens/s, latencia, pérdida, calidad y coste. Ninguna referencia autoriza afirmar ultra-eficiencia o inteligencia superior sin experimentos de Aethel.

## Hallazgos verificados en las páginas fuente

La página de GQA indica que MQA usa una sola cabeza K/V, que GQA generaliza ese esquema con un número intermedio de cabezas K/V y que su evaluación reporta calidad cercana a MHA con velocidad comparable a MQA en el protocolo de los autores. Esto sustenta GQA como hipótesis de reducción de KV-cache, no como garantía para Aethel. Fuente: https://arxiv.org/abs/2305.13245.

La página de Switch Transformers explica que MoE selecciona parámetros distintos por entrada y que el modelo queda dispersamente activado, pero también enumera complejidad, comunicación y estabilidad como obstáculos. El resumen reporta resultados de velocidad en su propio entorno experimental; no deben transferirse directamente a Aethel. Fuente: https://jmlr.org/papers/v23/21-0998.html.

## Hallazgos adicionales verificados

DeepSeek-V2 describe Multi-head Latent Attention (MLA) como una forma de comprimir el KV-cache en un vector latente y combina esa idea con MoE disperso. La fuente también deja claro que se trata de resultados de un modelo y escala concretos; para Aethel, MLA debe ser una ablation posterior frente a GQA, no un reemplazo asumido. Fuente: https://arxiv.org/abs/2405.04434.

LayerSkip describe entrenamiento con layer dropout y pérdidas de early exit, seguido de self-speculative decoding: capas tempranas proponen y capas restantes verifican/corrigen. Su página reporta speedups en tareas y modelos concretos; para Aethel sólo justifica experimentar con profundidad adaptativa y medir calidad/latencia por tarea. Fuente: https://aclanthology.org/2024.acl-long.681/.

## Escalado distribuido 100T — hallazgos adicionales

Megatron-LM describe el paralelismo tensorial dentro de las capas como complementario al paralelismo de pipeline y reporta entrenamiento de modelos de 8,3B con 512 GPU, mostrando que la comunicación y la colocación de operaciones son parte central del diseño, no un detalle posterior. Fuente: https://arxiv.org/abs/1909.08053.

ZeRO plantea eliminar redundancias de parámetros, gradientes y estados del optimizador mediante particionado entre dispositivos, con el objetivo de escalar el tamaño del modelo proporcionalmente al número de dispositivos. El artículo reporta más de 100B parámetros con 400 GPU, pero esto no constituye evidencia de que 100T sea viable con recursos pequeños. Fuente: https://arxiv.org/abs/1910.02054.

Estas fuentes respaldan una conclusión limitada: un Aethel de 100T exigiría paralelismo combinado —datos, tensorial, pipeline, experto y secuencia—, particionado de estados y una red de muy alta capacidad; no bastaría con activar FSDP en el código actual.

## Capacidad ampliable y aprendizaje continuo

- **Dynamically Expandable Networks (DEN)**, Yoon et al., propone decidir capacidad durante una secuencia de tareas, hacer expansión selectiva y dividir unidades para reducir deriva semántica; es evidencia en aprendizaje continuo, no una demostración de crecimiento autónomo de un LLM de 100B [https://arxiv.org/abs/1708.01547].
- **Firefly Neural Architecture Descent**, Wu et al., propone crecimiento progresivo de anchura/profundidad mediante una vecindad funcional y selección greedy basada en aproximación de Taylor; sus resultados apoyan estudiar crecimiento estructural controlado, no crear parámetros ilimitados durante conversación [https://proceedings.neurips.cc/paper_files/paper/2020/hash/fdbe012e2e11314b96402b32c0df26b7-Abstract.html].
- **Growing Efficient Deep Networks by Structured Continuous Sparsification**, Yuan et al., combina objetivos de precisión y esparsidad para crecer y podar redes; reporta ahorros en tareas de visión, por lo que debe tratarse como evidencia de método transferible, no como garantía para Transformers grandes [https://arxiv.org/abs/2007.15353].
- **Progressive Neural Networks**, Rusu et al., usa columnas nuevas para tareas nuevas y conexiones laterales, preservando columnas anteriores; evita olvido a costa de memoria creciente y no equivale a editar libremente los pesos troncales [https://arxiv.org/abs/1606.04671].
- **RAG**, Lewis et al., combina memoria paramétrica con memoria no paramétrica externa; es la opción más directa para ampliar conocimiento sin aumentar todos los parámetros, pero exige recuperación, procedencia y evaluación contra alucinaciones [https://arxiv.org/abs/2005.11401].

Conclusión de investigación: Aethel puede crecer mediante módulos o expertos nuevos con un controlador de expansión, pero la política segura debe ser externa al gradiente principal: detectar saturación, congelar una versión, añadir un bloque inicializado, entrenarlo con replay y holdout, comparar regresión y promoverlo sólo tras aprobación. La memoria externa puede crecer continuamente; los parámetros sólo deberían crecer por versiones discretas y auditadas. Añadir parámetros aumenta capacidad potencial, no garantiza retención ni razonamiento.

## Evidencia añadida — capacidad base pequeña y aprendizaje continuo

- Eldan y Li, *TinyStories: How Small Can Language Models Be and Still Speak Coherent English?*, arXiv:2305.07759, https://arxiv.org/abs/2305.07759. La fuente reporta que modelos muy pequeños pueden producir texto fluido y consistente en un dominio controlado con datos sintéticos cuidadosamente restringidos, y que el resultado no debe extrapolarse automáticamente a lenguaje general, bilingüismo amplio o razonamiento general.
- El acceso directo a OpenReview para el mismo trabajo quedó bloqueado por verificación del navegador; se usó la versión arXiv como fuente primaria accesible.
- La búsqueda también localizó una encuesta reciente sobre aprendizaje continuo de LLM y trabajos sobre olvido catastrófico; la conclusión de diseño es que la nueva capacidad debe evaluarse junto con retención de habilidades anteriores, no sólo con la tarea nueva.

Implicación para Aethel: un objetivo base de 100M debe especificar el dominio, la mezcla EN/ES y las pruebas de generalización. “Hablar fluido” en un conjunto controlado no equivale a competencia lingüística general; el razonamiento debe probarse con ejercicios no vistos y verificadores independientes.
