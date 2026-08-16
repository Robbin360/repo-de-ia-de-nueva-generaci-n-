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
- [ ] Integración de requisitos ampliados del dashboard solicitados por el usuario — implementada en preview; requiere imagen de producción con Python/PyTorch para completarse en vivo

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
