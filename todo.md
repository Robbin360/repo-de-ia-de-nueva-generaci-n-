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
