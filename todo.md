# Project TODO

- [x] Dashboard principal tipo laboratorio Aethel con sidebar: Chat, Arquitectura, Entrenador, Benchmarks y Estado del Motor
- [x] Estética dark cyberpunk con paleta índigo, cian y violeta
- [x] Tipografía monospace para métricas y fondo animado con partículas
- [x] Chat interactivo conectado al LLM real con personalidad Aethel V3 y contexto MoE, RoPE y GQA
- [x] Historial de conversaciones persistido en base de datos con timestamps y metadatos de sesión
- [x] Visualizador interactivo y animado de los cinco pilares: La Roca, El Líquido, Ciclo de Sueño, Neuromodulación y Espacio de Trabajo Global
- [x] Panel de estado del motor con cargas MoE, tokens por segundo, pérdida, VRAM y KV-Cache
- [x] Entrenador PyTorch real con dim, capas, expertos y learning rate configurables
- [x] Curva de pérdida animada usando exclusivamente métricas emitidas por el proceso PyTorch real
- [x] Tabla comparativa de benchmarks con GPT-4, Llama y Mixtral sin puntuaciones inventadas
- [x] Gráficas de barras preparadas para MMLU, HumanEval y GSM8K; permanecen vacías sin resultados verificables
- [x] Pruebas Vitest para estado del motor y ausencia de benchmarks sintéticos
- [x] Verificación visual responsive y corrección de errores
- [x] Checkpoint final del proyecto funcional en preview; publicación queda pendiente de imagen Python/PyTorch

## Historial

- [x] Identificación del repositorio Aethel distinto de katalog-ai
- [x] Inicialización del proyecto web aethel-platform
- [x] Integración de requisitos ampliados del dashboard solicitados por el usuario — preview y runtime productivo documentado con Python/PyTorch

## Notas de implementación

- El chat usa el LLM integrado del proyecto mediante el helper oficial del servidor; no se usan respuestas sintéticas.
- Las métricas del motor provienen únicamente del proceso PyTorch activo; sin proceso, el estado es NOT_CONNECTED y los valores son nulos.
- Los benchmarks muestran los nombres solicitados, pero no puntuaciones hasta disponer de resultados verificables de ejecuciones reales.
- La persistencia usará la base de datos del proyecto y seguirá el flujo schema -> migración SQL -> verificación.


- [x] Eliminar el endpoint de estado del motor basado en valores matemáticos simulados y sustituirlo por telemetría real o estado NO DISPONIBLE
- [x] Eliminar la curva de entrenamiento generada artificialmente y conectar el entrenador a un proceso PyTorch real con progreso del job
- [x] Eliminar benchmarks inventados y dejarlos no disponibles hasta cargar resultados verificables
- [x] Eliminar cualquier texto de UI que presente proyecciones o simulaciones como resultados reales
- [x] Implementar un adaptador real entre el servidor web y el motor Aethel/PyTorch sin respuestas sintéticas
- [x] Añadir pruebas que fallen si aparecen métricas simuladas, respuestas sintéticas o estados ONLINE sin proceso real
- [x] Documentar dependencias y requisitos reales para ejecutar el motor, incluyendo PyTorch, tokenizer/pesos cuando se incorporen y GPU opcional

## Nueva generación Aethel

- [x] Definir una arquitectura experimental bio-inspirada con memoria, plasticidad, MoE, RoPE/GQA y eficiencia medible
- [x] Separar capacidades verificables de aspiraciones de AGI y documentar límites de escala, datos y hardware
- [x] Implementar memoria episódica con recuperación real, persistencia JSONL y capacidad acotada; memoria semántica queda como extensión siguiente
- [x] Implementar aprendizaje continuo seguro mediante observación explícita, regularización contra olvido y evaluación de regresión
- [x] Implementar entrenamiento reproducible con corpus real, checkpoints y métricas JSONL persistidas
- [x] Preparar evaluación real de lenguaje sobre holdout y mantener MMLU, HumanEval y GSM8K sin scores inventados hasta implementar sus harnesses
- [x] Añadir controles del entrenamiento NextGen y telemetría del proceso al dashboard Aethel
- [x] Validar el modelo en CPU y documentar que competir con frontier requiere hardware y corpus de escala superior
- [x] Ejecutar pruebas de arquitectura, entrenamiento, memoria y ausencia de simulaciones
- [x] Dejar preparado el entrenamiento reanudable; la ejecución 24/7 requiere activar hosting reservado o una máquina persistente con recursos adecuados

## Alineación con arquitectura documentada del repositorio

- [x] Auditar AETHEL_FASE2_FRONTERA_V2.py, aethel_pytorch_architecture.py, train_aethel_v3.py y README para mapear La Roca, El Líquido, MoE, ultra-eficiencia y demás módulos
- [x] Comparar Aethel NextGen con el mapa arquitectónico original y registrar desviaciones en ARCHITECTURE_ALIGNMENT.md
- [x] Implementar La Roca como ruta estable y protegida del núcleo
- [x] Implementar El Líquido como plasticidad adaptativa explícita y versionada en liquid_versions.jsonl
- [x] Integrar Ciclo de Sueño, Neuromodulación y Espacio de Trabajo Global como módulos funcionales y medibles
- [x] Mantener MoE, RoPE, GQA, KV-Cache y restricciones de eficiencia en la arquitectura corregida
- [x] Reentrenar y evaluar el modelo alineado con datos reales; holdout posterior a la reescritura ejecutado
- [x] Actualizar telemetría y documentación con el mapa arquitectónico definitivo

- [x] Corregir el broadcast de memoria recuperada cuando el batch tiene más de una secuencia y repetir la evaluación alineada

## Entrenamiento continuo con GPU y corpus masivo

- [x] Diseñar la configuración escalable del modelo Aethel conforme a límites de cómputo y evaluación verificable
- [ ] Comparar opciones de infraestructura con GPU persistente y seleccionar una tras autorización de coste y cuenta
- [x] Preparar un manifiesto reproducible de corpus abierto, licencia, deduplicación, filtrado y partición de evaluación
- [x] Sustituir el tokenizador por bytes por un tokenizador BPE entrenable y versionado
- [x] Añadir reanudación exacta, checkpoints reanudables, empaquetado externo y logs de entrenamiento
- [x] Añadir soporte de precisión mixta y entrenamiento distribuido para GPU
- [x] Crear harnesses verificables de MMLU, HumanEval y GSM8K que esperan datasets y predicciones reales antes de producir métricas
- [ ] Iniciar una corrida continua solo después de autorizar proveedor, cuenta y presupuesto
- [x] Corregir la acumulación de gradiente del runner GPU y validarla con una corrida mínima local

## Primera corrida gratuita temporal

- [x] Comparar opciones gratuitas reales de GPU y sesiones temporales para la primera corrida de Aethel
- [x] Preparar el lanzamiento reproducible de Aethel para Kaggle Notebooks con GPU
- [x] Mantener checkpoints, manifiesto de datos y telemetría fuera del almacenamiento efímero
- [x] Documentar la transición desde la sesión gratuita a la GPU persistente autorizada
- [x] Implementar exportación automática de checkpoints, manifiesto y métricas a un destino persistente verificable
- [x] Probar que el runner confirma la escritura de artefactos persistentes antes de finalizar una sesión temporal
- [x] Ejecutar un runner local equivalente que confirme la persistencia de checkpoint, manifiesto y métricas antes de salir
- [x] Probar el bloqueo cuando falta AETHEL_KAGGLE_DATASET y la ruta de exportación exitosa mediante un CLI Kaggle simulado
- [x] Ejecutar un flujo completo equivalente que entrene un modelo mínimo y exporte sus artefactos persistentes antes de salir

## Mejora de calidad y escalado futuro

- [x] Auditar riesgos técnicos de la arquitectura Aethel a escala y priorizar mejoras sustentadas en evidencia
- [x] Añadir mecanismos de estabilidad de expertos, balanceo de carga y prevención de colapso MoE
- [x] Añadir control de consolidación, replay estratificado y métricas de retención para El Líquido y el Ciclo de Sueño
- [x] Diseñar un currículo de datos multilingüe, técnico y de razonamiento con etapas y criterios de avance
- [x] Definir una receta de escalado con presupuestos de parámetros, tokens, memoria y evaluación
- [x] Verificar que el cálculo analítico incluye todos los parámetros del núcleo NextGen sin instanciar presets grandes
- [x] Ejecutar un experimento real pequeño que compruebe las mejoras propuestas sin simulaciones
- [x] Detectar automáticamente FP16 o BF16 según la capacidad real de la GPU gratuita
- [x] Documentar capacidades esperables, límites y condiciones para aproximarse a calidad frontier
- [x] Corregir y ejecutar la prueba de estabilidad de routing MoE y replay estratificado
- [ ] Validar FSDP antes de intentar la familia Aethel `scale-1b` en varias GPU (requiere al menos 2 GPU CUDA reales)
- [x] Corregir y ejecutar la prueba de rechazo FSDP sin varias GPU CUDA
- [ ] Ejecutar una prueba reproducible de FSDP en al menos dos procesos/GPU que valide entrenamiento, checkpoint rango 0 y reanudación (bloqueada hasta disponer de GPU CUDA)
- [x] Añadir una prueba automatizada de la rama FSDP que registre estado y recuperación distribuida; su ejecución queda bloqueada hasta disponer de GPU CUDA
- [x] Documentar presupuestos explícitos de tokens por familia y sus puertas de evaluación
- [x] Guardar un reporte reproducible de entrenamiento y evaluación con pérdida, salud del router y pérdida de replay

## Entrenamiento continuo con GPU y corpus masivo

- [x] Diseñar la configuración escalable del modelo Aethel conforme a los límites de cómputo y evaluación verificable
- [ ] Comparar opciones de infraestructura con GPU persistente y seleccionar una tras autorización de coste y cuenta
- [x] Preparar un manifiesto reproducible de corpus abierto, licencia, deduplicación, filtrado y partición de evaluación
- [x] Sustituir el tokenizador por bytes por un tokenizador BPE entrenable y versionado
- [x] Añadir reanudación exacta, checkpoints reanudables, empaquetado externo y logs de entrenamiento
- [x] Añadir soporte de entrenamiento distribuido y precisión mixta para GPU
- [x] Crear scripts verificables de MMLU, HumanEval y GSM8K que requieren datasets y predicciones reales
- [ ] Iniciar una corrida continua únicamente después de que el usuario autorice el proveedor, la cuenta y el presupuesto

## Memoria, razonamiento y conversación de Aethel

- [x] Auditar las memorias existentes y definir capacidades verificables de memoria de trabajo, episódica, semántica y consolidación
- [x] Reforzar el núcleo NextGen con recuperación de memoria relevante y trazabilidad explícita de su uso
- [x] Incorporar un flujo de razonamiento estructurado y comprobable, sin presentar texto de razonamiento interno como hechos
- [x] Exponer en el chat las capacidades, límites, configuración activa y presupuestos de parámetros medidos de Aethel
- [x] Añadir pruebas del motor y del chat para la memoria, razonamiento estructurado y respuestas de especificación

## Eficiencia computacional e inteligencia verificable

- [x] Auditar los lenguajes de programación de Aethel y documentar la responsabilidad de cada uno
- [x] Investigar mecanismos de reducción de cómputo, memoria y parámetros con fuentes primarias
- [x] Diseñar una hoja de ruta de arquitectura para mejorar eficiencia y razonamiento sin afirmar inteligencia humana demostrada
- [x] Implementar y medir al menos una optimización segura de eficiencia en el núcleo o entrenamiento
- [x] Documentar límites, métricas y requisitos para comparar las mejoras en experimentos reales
- [x] Exponer en la ficha técnica del chat el modo cognitivo activo y la configuración verificable del motor cuando exista un proceso real
- [x] Añadir una prueba automatizada del flujo de ficha técnica en `chat.send` sin invocar una salida LLM no verificable
- [x] Medir latencia o memoria de la caché KV frente a la ruta sin caché con un protocolo reproducible
