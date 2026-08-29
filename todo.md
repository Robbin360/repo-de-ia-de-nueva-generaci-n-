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
- [x] Incorporar adaptadores de bajo rango opcionales en proyecciones de atención y MoE sin alterar la base por defecto
- [x] Exponer configuración y parámetros entrenables de la adaptación en el runner y los checkpoints
- [x] Probar el gradiente de adaptadores y medir la reducción de parámetros entrenables frente al ajuste completo

## Corrección de auditoría tecnológica del repositorio original

- [x] Localizar referencias a Triton, Rust, C#, Python y demás lenguajes o runtimes dentro del repositorio original
- [x] Clasificar cada tecnología como implementada, prevista, opcional o de infraestructura según evidencia textual y archivos fuente
- [x] Corregir la hoja de ruta y la ficha técnica para que no omitan tecnologías documentadas por el repositorio
- [x] Actualizar la ficha técnica del chat con Python/PyTorch, Triton/CUDA, Rust/Candle, TypeScript/Node.js y el estado de C++/C#
- [x] Verificar por lectura directa la hoja de ruta y la ficha técnica corregidas antes de guardar un checkpoint

## Arquitectura objetivo políglota: interfaz, silicio e inferencia persistente

- [x] Documentar TypeScript/Node.js y Tailwind/React como interfaz y capa de conexión de Aethel
- [x] Restringir Python/PyTorch en la especificación a laboratorio temporal de topología, entrenamiento y horneado de pesos
- [ ] Implementar y validar la ruta Triton obligatoria para atención causal y routing/dispatch MoE en GPU; actualmente sólo SwiGLU está integrado
- [x] Añadir una referencia CPU explícita de dispatch/combina MoE y pruebas de equivalencia para definir el contrato del futuro kernel Triton
- [x] Añadir una referencia CPU determinista de capacidad MoE para especificar asignaciones aceptadas y overflow antes del kernel Triton
- [x] Añadir una referencia CPU de prefill causal con pruebas de equivalencia SDPA antes del futuro kernel Triton por bloques
- [x] Implementar el kernel Triton experimental de prefill causal y dejarlo bloqueado hasta validación numérica CUDA
- [x] Definir la matriz de aceptación CUDA para equivalencia, gradientes, memoria, rendimiento y límites de los kernels Triton experimentales
- [x] Auditar y corregir etiquetas del dashboard que puedan confundir configuraciones calculadas con entrenamiento o métricas reales
- [x] Corregir la etiqueta global de runtime que aún puede implicar que Aethel está ejecutando un modelo propio entrenado
- [x] Añadir una prueba de texto que verifique que la interfaz declara configuración teórica, ausencia de checkpoint y bloqueo de entrenamiento
- [x] Persistir y verificar la matriz CUDA, enlazarla con la auditoría Triton y mantener los bloqueos estrictos hasta evidencia GPU
- [x] Implementar un ejecutor CUDA de aceptación que registre entorno, paridad y límites de los kernels experimentales sin habilitarlos automáticamente
- [x] Auditar las rutas actuales de atención causal y dispatch/combina MoE frente a los contratos Triton antes de completar kernels GPU
- [x] Bloquear explícitamente el prefill CUDA cuando `require_triton` esté activo hasta disponer de un kernel causal validado en GPU
- [x] Bloquear explícitamente el dispatch/combina MoE CUDA cuando `require_triton` esté activo hasta disponer de un kernel Triton validado
- [x] Documentar la capacidad de la infraestructura actual y los escenarios de escalamiento de cómputo, memoria, red y datos requeridos para Aethel
- [x] Documentar la topología operativa de Aethel: recursos para Sólido, Líquido, curiosidad, memoria, Sueño y servicios persistentes sin confundirlos con entrenamiento
- [x] Definir una topología de ejecución concurrente que separe el bucle autoregresivo del modelo de los servicios líquidos, memoria, curiosidad y Sueño
- [x] Definir un producto comercial inicial para Aethel, su propuesta de valor, operación, límites y criterios de lanzamiento
- [x] Definir una ficha técnica escalonada de Aethel con parámetros, expertos, capas, contexto, memoria y requisitos de inferencia por variante
- [x] Documentar el routing top-2 de Aethel Pro, incluyendo balanceo, capacidad, dispatch, combinación y contrato Triton de producción
- [x] Definir la ruta de Aethel desde entrenamiento autorizado hasta operación comercial persistente, con infraestructura, seguridad y criterios de promoción
- [ ] Entrenar y evaluar un modelo propio Aethel Edge con el Dataset trazable sólo cuando exista GPU autorizada, sin usar extracción de conocimiento interno ni resultados simulados
- [ ] Evaluar y documentar una ruta local reproducible con GPU para entrenar Aethel Seed y Edge sin depender de sesiones Kaggle
- [x] Implementar un inspector local que genere evidencia de GPU/CUDA, Dataset, almacenamiento y bloqueos antes de Seed o Edge
- [x] Documentar el runbook local de GPU para Aethel Seed, incluyendo directorios montados, preflight, evidencia, reanudación y límites de Edge
- [x] Comparar plataformas gratuitas vigentes de GPU para el piloto Aethel, incluyendo límites de sesión, almacenamiento y persistencia de checkpoints
- [x] Definir el runbook operativo de Aethel Seed para Kaggle con preflight offline, checkpoints atómicos, evaluación holdout y reanudación verificada
- [x] Alinear el lanzador Aethel Seed con `aethel-knowledge-corpus-v1-package` y probar que rechaza rutas, manifiestos o tokenizadores incompatibles antes de usar GPU
- [x] Preparar kernels Triton de atención causal de decodificación y selección top-2 del router con equivalencia CPU y preflight CUDA explícito
- [x] Diseñar el contrato de inferencia local token a token para un runtime Mojo
- [ ] Desplegar y validar el servicio Rust persistente en un host autorizado con supervisión real, healthcheck y restauración automática de snapshot
- [x] Añadir una plantilla de despliegue supervisado para el servicio Rust y una prueba de configuración sin declarar una instancia 24/7 activa
- [x] Implementar y probar un transporte Unix-socket local para el protocolo JSONL de memoria Rust
- [x] Implementar una capa local de memoria vectorial/RAG citable en Rust y probar recuperación end-to-end; el índice sigue siendo acotado y local
- [x] Mantener documentado que el servicio Rust actual está preparado localmente, pero no activo 24/7 hasta el despliegue autorizado
- [x] Auditar e implementar las partes posibles sin GPU CUDA ni runtime Mojo instalado, manteniendo marcadas las dependencias de hardware
- [x] Crear un núcleo Rust comprobable para registros de memoria, consolidación y recuperación trazable sin afirmar un RAG distribuido terminado
- [x] Definir un contrato de artefactos y de KV-cache para que un runtime Mojo de inferencia local pueda validarse contra PyTorch
- [x] Añadir pruebas reproducibles de los contratos Rust y Triton disponibles en este entorno
- [x] Exponer en la ficha técnica del chat el estado verificable de Python, Triton, Rust y Mojo sin declarar servicios no desplegados
- [x] Definir API JSONL, snapshot atómico, recuperación y protocolo de supervisión del servicio Rust; la ejecución persistente queda pendiente antes de declararlo 24/7
- [x] Evitar que el saludo del chat muestre un marcador de parámetros mientras la ficha técnica calculada se está cargando
- [x] Crear un preflight único de GPU que ejecute comprobaciones Triton y FSDP reales sin convertir un host sin CUDA en éxito

## Ingeniería final previa a GPU y entrenamiento real

- [x] Definir una variante Aethel experimental de eficiencia y razonamiento con hipótesis, coste y criterios de rechazo medibles
- [x] Implementar módulos cognitivos o de eficiencia que puedan validarse sin CUDA, preservando compatibilidad con checkpoints
- [x] Añadir comparativas reproducibles contra la configuración base para pérdida, estabilidad de router, memoria y coste de cómputo
- [x] Revisar el manifiesto de corpus, la receta de entrenamiento y las puertas de evaluación antes de usar una cuenta GPU
- [x] Preparar un checklist de autorización explícita que separe acceso de cuenta, selección de GPU, límite de gasto e inicio de corrida

## Primera validación con cuota gratuita

- [x] Verificar opciones vigentes de cuota gratuita con GPU CUDA para una validación inicial de Aethel
- [x] Comparar límites, persistencia de artefactos y compatibilidad con el preflight Triton/FSDP sin crear recursos de pago
- [ ] Preparar la opción gratuita seleccionada y pedir confirmación antes de iniciar cualquier sesión
- [x] Verificar y preparar el cuaderno de Kaggle, el manifiesto aprobado y los artefactos de exportación sin habilitar una sesión GPU
- [x] Integrar el validador aprobado de corpus y evaluación en el lanzador Kaggle antes de instalar dependencias o usar GPU
- [x] Comprobar el acceso de la cuenta Kaggle en el navegador sin crear notebook ni consumir cuota
- [x] Añadir una validación automática de preparación que rechace manifiestos no aprobados o rutas de evaluación incompletas antes de lanzar GPU
- [x] Integrar el validador de preparación en el launcher GPU para abortar antes de cualquier descarga o entrenamiento si falla
- [x] Exigir revisiones inmutables aprobadas y entradas de evaluación configuradas antes de permitir un lanzamiento GPU
- [x] Comprobar accesibilidad real de holdout, tokenizador y referencias de benchmark en el modo de lanzamiento aprobado
- [x] Probar el rechazo de rutas de evaluación inexistentes y el bloqueo del launcher antes de instalar o descargar
- [x] Añadir un inspector de compatibilidad de checkpoints que informe claves y formas antes de cualquier carga estricta en Kaggle
- [ ] Configurar el cuaderno Kaggle para guardar una versión comprometida mediante Save & Run All, preservando ejecución al cerrar la página
- [ ] Pedir confirmación final específica antes de iniciar la versión comprometida que consume cuota gratuita
- [x] Implementar una variante de refinamiento adaptativo con pasos máximos, umbral de dificultad y telemetría de cómputo efectiva
- [x] Comparar la variante adaptativa contra el baseline con una prueba reproducible de pérdida y coste de refinamiento
- [x] Exponer la variante ARC en la ficha técnica del chat con su presupuesto calculado, telemetría y límites experimentales
- [x] Añadir una prueba reproducible de carga y reanudación de checkpoints para baseline y ARC, incluyendo activar/desactivar ARC
- [x] Extender la comparación ARC-baseline con medición real de RAM/VRAM según el entorno y persistirla en el reporte experimental
- [x] Sustituir el flujo heredado de Aethel V3 en Kaggle por scripts autónomos de preparación, empaquetado de fuentes y lanzamiento de Aethel NextGen sin reutilizar checkpoints históricos incompatibles
- [x] Investigar y fijar un conjunto de fuentes reales bilingües, enciclopédicas y de razonamiento con licencia, revisión inmutable, procedencia y límites para un piloto Aethel NextGen
- [x] Mejorar la receta de entrenamiento NextGen para mezcla de datos, currículo, estabilidad MoE, evaluación bilingüe y registros reproducibles
- [ ] Construir y validar un Dataset privado `aethel-nextgen-data` sin datos sintéticos ni checkpoints V3
- [x] Crear un cuaderno privado Kaggle independiente para Aethel NextGen con entradas separadas, GPU T4 x2 y borrador no ejecutado
- [ ] Ejecutar Save Version → Save & Run All solo tras validar los controles de datos, evaluación y persistencia del cuaderno NextGen autorizado
- [x] Adaptar el flujo de datos bilingües real para construirlo dentro de Kaggle con límites de memoria, reintentos, hashes y evaluación retenida
- [x] Hacer que el bootstrap del cuaderno detecte de forma segura el paquete de fuentes comprimido aunque Kaggle preserve un nombre remoto no descriptivo
- [x] Documentar objetivos, criterios de éxito y límites verificables del piloto NextGen sin presentarlo como modelo de frontera final
- [x] Reforzar checkpointing, exportación y reanudación automática para interrupciones o el límite de 12 horas de Kaggle
- [x] Limitar la retención de snapshots portátiles y documentar la reanudación explícita entre sesiones sin depender de directorios efímeros

- [x] Diagnosticar la versión comprometida #1: falló antes del entrenamiento porque `/kaggle/input/aethel-nextgen-source` montó cero archivos `.gz`
- [ ] Corregir la versión del Dataset adjunta y verificar que el paquete `.gz` esté visible en `/kaggle/input/aethel-nextgen-source`
- [ ] Crear una nueva versión comprometida sólo después de pasar el bootstrap de entrada y conservar el registro del resultado real
- [x] Robustecer el preparador bilingüe ante HTTP 429/502 con Retry-After, backoff progresivo y persistencia reanudable de descargas parciales
- [x] Añadir pruebas deterministas de reintentos, Retry-After y reanudación de descargas parciales
- [x] Regenerar el bundle de fuentes y la celda de Kaggle para el siguiente Save & Run All
- [x] Añadir Open License Corpus `pd_books` con revisión fijada y umbral inglés trazable para superar 14.000 documentos
- [x] Validar que la nueva fuente no sustituye silenciosamente las fuentes inglesas existentes
- [x] Hacer que la celda de Kaggle priorice por contenido el bundle que incluye olc-pd-books-en frente a bundles antiguos montados simultáneamente
- [x] Resolver HTTP 502 recurrente de OLC o sustituirlo por una fuente inglesa real más estable sin relajar el mínimo de 14.000
- [ ] Ejecutar una puerta remota que confirme los conteos inglés/español antes de iniciar entrenamiento
- [x] Subir el bundle regenerado como nueva versión del Dataset privado `aethel-nextgen-source`
- [ ] Confirmar en Kaggle que la entrada montada contiene el bundle con `project-gutenberg-en` y ejecutar Save Version → Save & Run All
- [x] Subir el bundle validado con Project Gutenberg como nueva versión del Dataset privado de Kaggle y verificar su contenido
- [x] Corregir el selector V9: debe priorizar explícitamente el bundle con `project-gutenberg-en` sobre copias que sólo contienen OLC
- [x] Regenerar y publicar una nueva versión del Dataset con el selector corregido
- [x] Verificar en Kaggle que se selecciona la versión nueva antes de ejecutar el launcher
- [x] Corregir la celda del cuaderno que aún selecciona `pKXqovDfRmpcogEs` en vez del bundle Gutenberg
- [ ] Repetir la preparación sin interrumpirla y confirmar la puerta real de conteos bilingües
- [x] Añadir y validar una ruta inglesa de respaldo con procedencia verificable ante HTTP 502 simultáneos de OLC y Project Gutenberg
- [ ] Pegar la celda V11 en Kaggle, confirmar que selecciona el bundle con respaldo Wikimedia y ejecutar la puerta bilingüe
- [x] Tratar `IncompleteRead` de Hugging Face como fallo reintentable y aislar fuentes auxiliares para que no aborten la preparación
- [ ] Separar el piloto en preparación de corpus preempaquetado, validación de conteos y entrenamiento GPU, sin descargas de red durante la corrida
- [ ] Ejecutar una prueba corta de entrenamiento real exclusivamente después de montar un corpus bilingüe validado localmente en Kaggle
- [x] Construir un Dataset bilingüe trazable de lenguaje, matemáticas, ciencia, ingeniería y programación con fuentes reales aprobadas
- [x] Validar procedencia, licencias, idioma, deduplicación, balance de dominios y separación de holdout antes de cualquier entrenamiento
- [x] Empaquetar y versionar localmente el Dataset de datos independiente del bundle de código de Aethel
- [ ] Publicar el paquete validado como Dataset privado de Kaggle `aethel-nextgen-data-v1` sin iniciar una sesión GPU
- [ ] Guiar la creación y verificación del Dataset privado de Kaggle con los 22 shards congelados, sin activar GPU ni entrenamiento
- [ ] Verificar la sesión Kaggle del usuario en navegador y preparar el formulario privado del Dataset antes de solicitar confirmación de creación
- [ ] Vincular la sesión personal de navegador del usuario a Kaggle antes de realizar acciones en su cuenta
- [ ] Usar exclusivamente My Browser para Kaggle y detenerse ante falta de conexión, login o CAPTCHA sin cambiar a navegador aislado
- [ ] Confirmar que una navegación de prueba abre una pestaña en My Browser del usuario antes de acceder a Kaggle
- [ ] Intentar una única navegación mediante My Browser y capturar cualquier tarjeta o aviso de conexión sin continuar en navegador aislado
- [ ] Guiar y verificar la vinculación de My Browser con este chat antes de volver a navegar
- [x] Crear un documento integral de continuidad de Aethel para reanudar el proyecto en un nuevo chat
- [x] Actualizar el documento de continuidad con los contratos Triton, runbook local y correcciones de transparencia posteriores
- [ ] Confirmar que Browser Operator expuso una sesión My Browser utilizable antes de acceder a Kaggle
- [ ] Seleccionar y registrar el acelerador Kaggle disponible para Aethel Seed según VRAM, número de GPU y compatibilidad CUDA/Triton
- [ ] Verificar el hito de modelo funcional Seed mediante checkpoint entrenado, generación token a token, evaluación holdout bilingüe y restauración reproducible
- [x] Mantener local el paquete de Dataset validado y suspender su publicación en Kaggle por decisión del usuario
- [x] Auditar el estado actual de La Roca, El Líquido, Ciclo de Sueño, neuromodulación, memoria y espacio de trabajo global
- [x] Especificar el ciclo operativo y los contratos de seguridad entre los módulos sólido, líquido y sueño
- [x] Definir métricas experimentales y puertas de validación para la arquitectura cognitiva antes de entrenar
- [x] Añadir una prueba CPU de contratos cognitivos: La Roca no muta durante observación, El Líquido queda versionado y la traza no expone razonamiento interno
- [x] Documentar la autonomía de aprendizaje acotada de Aethel, sus objetivos de competencia y los límites frente a un cerebro humano
- [x] Especificar un controlador de curiosidad funcional que priorice incertidumbre, novedad, contradicción y lagunas sin autoasignarse objetivos ilimitados
- [x] Persistir propuestas de curiosidad en El Líquido con TTL y procedencia, sin admisión automática al replay de Sueño
- [x] Implementar manifiesto inmutable de La Roca y rama candidata LoRA aislada para Sueño, con prueba CPU de integridad y reversión
- [x] Medir progreso longitudinal de aprendizaje en El Líquido para que la curiosidad priorice reducción de incertidumbre en vez de ruido impredecible
- [x] Implementar una puerta de admisión de replay que exija curación, procedencia, aprobación explícita y separación de holdout sin iniciar ajuste
- [x] Exigir un registro de aprobación independiente y vinculada al hash de procedencia antes de admitir un evento al replay de Sueño
- [x] Implementar preflight de Sueño que vincule La Roca, Dataset congelado y replay aprobado, rechazando inconsistencias y holdout antes de ajuste
- [x] Implementar una máquina de estados auditable para Sueño que impida saltos de cuarentena, ejecución y promoción sin autoridad explícita
- [x] Integrar el reporte de preflight con la máquina de estados para permitir únicamente la transición de cuarentena a preflight válido
- [x] Revisar y completar el archivo de continuidad para incluir toda la historia relevante, decisiones, evidencias, bloqueos y estado actual del proyecto
- [x] Verificar que los cambios implementados de frontend y motor estén sincronizados en el repositorio GitHub correcto
- [x] Verificar la compilación y las pruebas del runtime Rust versionado de Aethel
- [x] Actualizar en GitHub la documentación de estado y la matriz de lenguajes, responsabilidades y componentes de Aethel
- [x] Contrastar la auditoría técnica aportada, aclarar artefactos no certificados y reforzar el estado verificable en la documentación de Aethel
- [x] Inventariar los archivos locales de Aethel, su relación con GitHub y las acciones permitidas para preservarlos
- [x] Revisar si la auditoría del otro chat comprendió la visión, el contenido y las restricciones verificables de Aethel
- [x] Redactar directivas corregidas para el otro chat sobre los cinco pilares, la meta bilingüe, respaldos locales y Kaggle/My Browser
- [x] Auditar y sincronizar únicamente archivos locales aptos para la preparación de Dataset privado y E0, manteniendo datos y pesos fuera de GitHub
- [x] Verificar y documentar que GitHub contiene los insumos seguros para que otro chat continúe Dataset y E0 sin acciones de Kaggle desde esta sesión
- [x] Retirar del repositorio GitHub los artefactos compilados Rust regenerables y excluirlos de futuras sincronizaciones
- [x] Inventariar y sincronizar en GitHub todos los scripts, manifiestos, validadores y metadatos seguros locales necesarios para reproducir el proceso Aethel
- [x] Redactar la directiva de continuidad para Dataset privado y E0, con objetivos, parámetros, límites y ruta Seed–Edge–Pro
- [x] Crear y verificar un archivo local privado de transferencia del Dataset congelado para el entorno que opera Kaggle
- [x] Guiar la subida manual del Dataset privado y la preparación del preflight E0 cuando My Browser no esté disponible
- [x] Crear y verificar un ZIP privado con estructura plana del Dataset listo para extracción y carga manual
- [x] Verificar visualmente el Dataset privado `aethel-nextgen-data-v1` recién creado antes de preparar el preflight E0
- [x] Generar y verificar localmente el bundle privado de código Aethel para el preflight E0, sin subirlo a Kaggle
- [x] Alinear el manifiesto del bundle de código con el Dataset privado `aethel-nextgen-data-v1`
- [x] Explicar y guiar la creación manual del Dataset privado de código E0, separado del Dataset de conocimiento
- [x] Adaptar y verificar el preflight E0 para la estructura de código descomprimida por Kaggle, sin GPU ni entrenamiento
- [x] Añadir un ejecutor reproducible de preflight offline que no importe CUDA ni inicie entrenamiento
- [x] Actualizar manualmente `aethel-nextgen-source-e0-v1` con el bundle que incorpora el preflight offline
- [x] Diagnosticar la ruta real de montaje del Dataset de código en Kaggle tras el bloqueo del preflight
- [x] Ajustar la celda de preflight a la ruta anidada de montaje expuesta por Kaggle
- [x] Inspeccionar en modo de solo lectura la estructura real de `corpus/` montada por Kaggle y resolver el bloqueo de validación sin modificar el Dataset
- [x] Añadir verificación de hashes de contenido plano para el montaje descomprimido por Kaggle, sin aceptar shards no verificables ni alterar el Dataset congelado
- [x] Subir manualmente el bundle privado de código V3 a una nueva versión de `aethel-nextgen-source-e0-v1` y repetir sólo el preflight offline
- [x] Seleccionar de forma verificable el bundle V3 dentro de las carpetas de código multi-versión montadas por Kaggle antes de ejecutar el preflight
- [x] Ejecutar la aceptación CUDA/Triton en Kaggle T4 ×2 sin autorizar ni iniciar el entrenamiento E0
- [x] Corregir el fallo real de compilación Triton en `causal_prefill_kernel` (`head_dim` constexpr tratado como entero) y repetir la aceptación aislada
- [x] Preservar el informe real de aceptación experimental V4 y mantener bloqueadas las rutas Triton estrictas hasta cubrir gradientes, dispatch/combine y la matriz CUDA completa
- [x] Añadir una identidad de release V4 verificable para impedir que Kaggle seleccione un bundle V3 anterior durante la repetición de aceptación
- [x] Incluir la identidad de release V4 en el informe de aceptación, ya que el código descomprimido de Kaggle no tiene metadatos Git disponibles
- [x] Repetir el preflight offline con el release V4 verificable y confirmar integridad de los 22 shards antes de la aceptación CUDA/Triton
- [x] Preparar una celda de lanzamiento Seed E0 que seleccione el release V4, requiera doble autorización y conserve checkpoints privados
- [x] Adaptar el lanzador E0 para resolver holdouts `.jsonl` descomprimidos por Kaggle sin alterar el Dataset ni permitir mezcla de formatos
- [x] Emitir un release V5 identificable que contenga la compatibilidad de shards descomprimidos y el lanzamiento E0 controlado
- [x] Sustituir la instalación `pip` del lanzador por una comprobación explícita y sin red de dependencias ya disponibles en Kaggle
- [x] Crear y ejecutar un preflight offline que seleccione explícitamente el release V5 antes del lanzamiento E0
- [x] Crear y ejecutar un preflight offline que seleccione explícitamente el release V5 antes del lanzamiento E0
- [x] Diagnosticar el fallo de la versión Kaggle tras el inicio real de E0 V5 y verificar si persiste algún checkpoint o recibo recuperable
- [x] Corregir la actualización de memoria líquida para alinear la traza hebbiana al dispositivo CUDA antes de cualquier reintento E0
- [x] Ejecutar en Kaggle la regresión CUDA V6 de memoria líquida y bloquear el reintento E0 al fallar la aserción de `memory_state` en CUDA
- [x] Empaquetar y verificar localmente el release V6 `e0-v6-liquid-cuda-alignment` sin incluir Dataset, pesos, checkpoints ni bytecode
- [x] Subir el bundle V6 verificado como una nueva versión privada de `aethel-nextgen-source-e0-v1`, sin modificar `aethel-nextgen-data-v1`
- [x] Actualizar la guía Kaggle y el handoff con el intento V5 abortado, la ausencia de checkpoint recuperable y las puertas V6
- [x] Corregir el notebook si el preflight sigue mostrando `SOURCE_RELEASE: e0-v5-plaintext-kaggle-e0-launch` y repetir sólo el preflight V6 hasta verificar el selector exacto
- [x] Corregir la reasignación de `memory_state` durante `forward`, detectada por el smoke V6 antes de E0
- [x] Empaquetar y verificar localmente el release V7 `e0-v7-registered-memory-state`, sin corpus, pesos, checkpoints ni bytecode
- [x] Emitir y ejecutar el smoke CUDA del release V7; quedó invalidado por la comparación no canónica entre `cuda` y `cuda:0`, sin iniciar E0
- [x] Diagnosticar el falso bloqueo V7: la igualdad entre `torch.device("cuda")` y `torch.device("cuda:0")` no es canónica, por lo que no acreditaba estado CPU
- [x] Emitir y verificar localmente el release V8 `e0-v8-canonical-cuda-device-check`, sin corpus, pesos, checkpoints ni bytecode
- [x] Ejecutar y validar en Kaggle el smoke CUDA V8 con comparación de tipo, índice e identidad del buffer antes de cualquier reintento E0
- [x] Documentar el preflight y smoke CUDA V8 aprobados, manteniendo explícitamente ausentes el entrenamiento, checkpoint y evaluación E0 V8
- [x] Reemplazar la tercera celda de lanzamiento V5 por el lanzador experimental V8 antes de solicitar el commit E0
- [x] Corregir la adjunción o actualización del input de código en Kaggle cuando el preflight V8 no encuentre `e0-v8-canonical-cuda-device-check`
- [x] Ejecutar el preflight V8 con el release exacto montado y validar los 22 shards sin red antes del smoke CUDA
- [x] Ejecutar E0 V8 mediante Commit tras el preflight y smoke aprobados; el log llegó al paso 4992 y el lanzador terminó sin error
- [x] Inspeccionar `latest.pt`, `recovery_receipt.json` y los artefactos persistidos de E0 V8 sin promover el checkpoint
- [x] Inspeccionar `checkpoint_inspection.json`: 150 tensores, metadatos completos, paso 4992 y reanudación reproducible declarada
- [x] Inspeccionar por separado `evaluation_holdout_en.json` y `evaluation_holdout_es.json` antes de citar pérdidas o métricas de holdout
- [x] Auditar `metrics_rank_0.jsonl` sin cargar pesos: 4992 pasos, telemetría de router y salvaguardas de memoria/curiosidad registradas
- [x] Actualizar el dashboard con el estado E0 V8 auditado, sus métricas holdout separadas y límites de no promoción
- [x] Documentar una remediación E0 basada en la pérdida, brecha EN/ES y salud del router observadas, sin reanudar ni ejecutar otro entrenamiento
- [x] Implementar y validar localmente la auditoría D0 de solo lectura para marcador de release, manifiesto raíz, hash del tokenizer, conteos congelados y evidencia estática E0 V8; no abre shards, holdout, pesos ni telemetría cruda
- [x] Preparar la celda Kaggle D0 sin GPU ni acceso a pesos, holdout, entrenamiento o modificaciones de Dataset; ejecutada una vez de forma controlada
- [x] Construir y auditar el bundle privado de código D0, excluyendo Dataset, pesos, checkpoints, bytecode, cachés y dependencias
- [x] Actualizar exclusivamente el Dataset privado de código `aethel-nextgen-source-e0-v1` con el bundle D0 tras confirmación inmediata del usuario; versión privada montada `(9)` y marcador exacto verificado
- [x] Editar y ejecutar la celda D0 en Kaggle sólo tras confirmaciones inmediatas separadas; `D0_AUDIT_READY` confirmó cero GPU, pesos, corpus/holdout crudo, red, entrenamiento, reanudación o D1
- [x] Preparar y revisar un protocolo D1A de diagnóstico desde inicialización nueva, con sólo train, telemetría de router y autorizaciones separadas; no se ejecutó GPU, Kaggle ni entrenamiento
- [x] Implementar y probar localmente el validador D1A train-only y el resumen sin pesos de `metrics_rank_0.jsonl`, con rechazo explícito de holdout, checkpoints, GPU y red
- [x] Construir y auditar localmente el bundle privado D1A, con release exacto y exclusiones de Dataset, pesos, checkpoints, bytecode, cachés y dependencias; después se subió sólo al Dataset privado de código tras confirmación separada
- [x] Actualizar exclusivamente el Dataset privado de código `aethel-nextgen-source-e0-v1` con el bundle D1A tras confirmación inmediata; release exacto verificado, sin modificar el Dataset de datos ni ejecutar notebook
- [x] Preparar una celda D1A para añadir al notebook, con la ejecución, GPU, entrenamiento y evaluación holdout todavía bloqueados
- [x] Añadir la celda D1A al notebook sólo tras confirmación inmediata; se ejecutó una vez sólo en modo bloqueado, sin seleccionar GPU
- [x] Ejecutar D1A desde inicialización nueva sólo tras una nueva autorización inmediata para seleccionar GPU y entrenar; 768 pasos concluyeron con `D1A_DIAGNOSTIC_COMPLETE`, manteniendo holdout, pesos E0, reanudación y D1B bloqueados
- [x] Preparar y validar localmente la modificación habilitable de la celda D1A, sin ejecutarla todavía ni relajar las puertas de holdout, E0, reanudación, promoción o D1B
- [x] Reemplazar la quinta celda D1A por la variante habilitable sólo tras confirmación inmediata; verificar primero que permanecía bloqueada y abrir sus cuatro puertas sólo tras autorización final
- [x] Abrir las cuatro puertas D1A autorizadas y ejecutar únicamente 768 pasos desde inicialización nueva; mantuvo holdout, pesos E0, reanudación, promoción y D1B bloqueados
- [x] Revisar y, si se autoriza de forma separada, preservar la salida D1A mediante `Save Version`; Version #3 fue exitosa sin cargar, mover, reanudar ni promover ningún checkpoint
- [x] Ejecutar Save Version privado para la salida D1A tras confirmación inmediata y verificar su versión creada, sin acciones sobre checkpoints
- [x] Redactar un plan documental de revisión de evidencia D1A que delimite metadatos permitidos, exclusiones y autorizaciones futuras, sin abrir salidas, checkpoints, corpus, holdout ni Kaggle
- [x] Preparar y validar un protocolo documental D1B basado sólo en la evidencia resumida D1A, sin abrir artefactos, modificar código/Dataset ni usar Kaggle o GPU
- [x] Implementar y probar contratos locales D1B bloqueados que exijan release exacto, train-only, inicio nuevo y autorizaciones explícitas, sin ejecutar GPU, Kaggle ni entrenamiento
- [x] Construir, auditar y actualizar manualmente sólo el Dataset privado de código con el release D1B; la interfaz confirmó creación exitosa, sin número de versión/montaje visible y sin modificar `aethel-nextgen-data-v1`, notebook, GPU, outputs ni checkpoints
- [x] Preparar, añadir y ejecutar una celda D1B bloqueada; el release montado `(11)` devolvió `D1B_CELL_PREPARED_NOT_EXECUTED` sin copiar código, leer datos, seleccionar GPU ni ejecutar entrenamiento
- [x] Preparar y validar localmente una variante habilitable D1B con cinco puertas separadas para reemplazo de celda, GPU y ejecución, sin modificar Kaggle ni ejecutar diagnóstico
- [x] Reemplazar manualmente la celda D1B bloqueada por la variante habilitable y verificar que las cinco puertas permanezcan cerradas; el montaje `(11)` devolvió `D1B_EXECUTION_PENDING_FINAL_AUTHORIZATION` sin GPU ni ejecución
- [x] Entregar y validar la Celda 5 D1B completa con encabezado de numeración, y establecer esa convención para futuras celdas sin ejecutarla
- [x] Ejecutar una única corrida D1B autorizada desde inicialización nueva, con GPU T4 ×2 y sólo train; `D1B_DIAGNOSTIC_COMPLETE` registró 44/768 pasos saludables y router no mejorado, sin abrir holdout ni manejar checkpoints
- [x] Realizar una revisión documental comparativa D1A/D1B que descarte hipótesis de router no apoyadas, sin abrir artefactos ni proponer ejecución
- [x] Definir y validar una ruta local de aceleración hacia un primer modelo Aethel verificable, con criterios de decisión y sin habilitar acciones externas no autorizadas
- [x] Implementar y validar un contrato local determinista de salud del router MoE con entradas sintéticas, sin corpus, pesos, GPU ni Kaggle
- [x] Auditar y preparar localmente el control de la pérdida auxiliar MoE para una hipótesis futura, sin abrir datos protegidos ni iniciar una corrida
- [x] Verificar localmente la dirección de la señal auxiliar MoE con tensores sintéticos, sin formular ni ejecutar una nueva corrida
- [x] Preparar y validar localmente D1C como diagnóstico train-only bloqueado del peso auxiliar MoE, sin actualizar Kaggle ni ejecutar entrenamiento
- [x] Preparar y validar localmente el release de código y la Celda 6 bloqueada de D1C, sin actualizar Kaggle ni ejecutar el diagnóstico
- [x] Crear manualmente una nueva versión privada del Dataset de código D1C con el bundle validado, sin tocar el Dataset de datos ni ejecutar notebook
- [x] Verificar el montaje D1C mediante la CELDA 6 bloqueada, sin usar GPU, abrir datos ni ejecutar entrenamiento
- [x] Preparar y validar localmente una plantilla D1C habilitable con puertas cerradas, sin usar GPU ni ejecutar el diagnóstico
- [x] Añadir y verificar la CELDA 7 D1C con sus cinco puertas cerradas, sólo tras una confirmación específica y sin ejecutar diagnóstico
- [x] Ejecutar una única D1C autorizada: inicialización nueva, train-only, 768 pasos y `router_aux_loss_weight=0.05`; el intento V1 quedó bloqueado al cerrar el resumen, sin holdout, pesos E0, reanudación, promoción ni serving
- [x] Corregir y validar localmente el soporte de resumen D1C que bloqueó el cierre de la única corrida autorizada, sin reanudar ni inspeccionar outputs
- [x] Preparar localmente el release D1C V2 de corrección de resumen y su verificación bloqueada, sin actualizar Kaggle ni repetir la corrida
- [x] Crear manualmente una nueva versión privada del Dataset de código D1C V2 con el bundle validado, sin tocar el Dataset de datos ni ejecutar notebook
- [x] Verificar el montaje del release privado D1C V2 mediante CELDA 8 bloqueada, sin usar GPU, abrir datos ni ejecutar retry
- [x] Preparar y validar localmente una plantilla de retry D1C V2 con puertas cerradas, sin repetir entrenamiento ni acceder a artefactos
- [x] Preparar y entregar la variante D1C de CELDA 7 con puertas abiertas para edición manual, sin ejecutarla
- [x] Preparar y entregar la variante D1C de CELDA 7 con puertas abiertas para edición manual, sin ejecutarla
- [x] Construir y validar localmente el release D1C V3 con la CELDA 9 de retry bloqueada, sin cargarlo a Kaggle ni acceder a Dataset, GPU, outputs o checkpoints
- [x] Subir manualmente el bundle privado D1C V3 al Dataset de código tras autorización específica y verificar visualmente su directorio; no autoriza editar/ejecutar CELDA 9, GPU ni retry
- [x] Guiar la creación manual autorizada de la versión privada de código D1C V3, sin cambios de notebook, GPU ni retry
- [x] Añadir manualmente la CELDA 9 D1C V3 con sus cinco puertas cerradas, sin ejecutarla ni seleccionar GPU
- [x] Ajustar y validar la CELDA 9 bloqueada para que seleccione exactamente el release D1C V3, sin abrir su rama de retry
- [x] Ejecutar una vez la CELDA 9 D1C V3 exclusivamente en modo bloqueado y registrar su salida segura «candidatos: ninguno», sin GPU, retry, Dataset train/holdout, pesos, outputs ni checkpoints
- [x] Diagnosticar y corregir localmente el selector de la CELDA 9 V3 tras el bloqueo seguro «candidatos: ninguno», sin habilitar GPU ni retry
- [x] Determinar que no era necesario reemplazar la CELDA 9 V3: el bloqueo procedía del input de código no actualizado en el notebook
- [x] Registrar la salida exitosa de CELDA 9 V3 tras actualizar el input de código del notebook, sin habilitar GPU ni retry
- [x] Delimitar y documentar localmente un eventual retry D1C como experimento nuevo, sin reanudar V1 ni habilitar acciones externas
- [x] Preparar contratos locales de una plantilla D1C V3-R1 con puertas cerradas, sin editar el notebook ni ejecutar retry
- [x] Implementar y validar localmente la plantilla de CELDA 10 D1C V3-R1 con todas las puertas cerradas, sin editar el notebook ni ejecutar retry
- [x] Corregir y validar localmente el contrato del lanzador D1C para un release V3-R1, manteniendo intacto el comportamiento histórico V1
- [x] Preparar y validar localmente el release de código D1C V4 para la ruta V3-R1, sin cargarlo a Kaggle ni editar o ejecutar notebooks
- [x] Preparar un empaquetador local seguro para V4 que permita referencias V3-R1 y excluya datos y artefactos
- [x] Preparar un empaquetador local seguro para V4 que permita referencias V3-R1 y excluya datos y artefactos
- [x] Documentar el bundle D1C V4 como preparación exclusivamente local, sin afirmar actualización de Kaggle ni ejecución
- [x] Guiar la carga manual autorizada del ZIP privado D1C V4, sin editar/ejecutar CELDA 10 ni usar GPU o retry
- [x] Registrar la confirmación visual de Version 16 privada y exitosa del release D1C V4, sin habilitar notebook, GPU o retry
- [x] Pegar manualmente la CELDA 10 D1C V3-R1 con seis puertas cerradas, sin ejecutarla ni habilitar GPU o retry
- [x] Ejecutar una única corrida D1C V3-R1 autorizada en T4: 768 pasos, train-only, inicialización nueva, sin holdout, reanudación ni acceso a artefactos
- [x] Validar estáticamente la CELDA 10 autorizada y comprobar que conserva el release, la salida nueva y la prohibición de reanudación
- [x] Documentar la corrida D1C V3-R1 completada como `D1C_ROUTER_NOT_IMPROVED`, sin abrir outputs/checkpoints ni iniciar D1D
- [x] Investigar localmente si la señal de balanceo basada en masa probabilística suave evita el atractor de dos expertos, sin GPU ni Dataset

- [x] Corregir y validar localmente la expectativa de signo de la regularización de entropía del router MoE
- [x] Redactar el protocolo formal D1D de regularización de entropía densa del router
- [x] Integrar D1D en el entrenador sólo después de validar el contrato CPU y mantenerla bloqueada para GPU
- [x] Añadir una prueba CPU de integración que verifique composición separada, propagación CLI y compatibilidad del peso D1D=0
- [x] Preparar y validar localmente el bundle V5 de código D1D sin datos ni artefactos
- [x] Crear y validar localmente la CELDA 11 D1D bloqueada, sin habilitar GPU ni ejecución
- [x] Diagnosticar el input Kaggle real que no contiene el marcador D1D y preparar una celda de resolución segura
- [x] Corregir la CELDA 11 para resolver el marcador D1D desde cualquier raíz de montaje confirmada por CELDA 11A
- [x] Preparar la celda ejecutable D1D train-only de 768 pasos con salida nueva y bloqueo de holdout/reanudación
- [x] Extender el resumidor y sus pruebas para aceptar D1D sin alterar los contratos D1A-D1C
- [x] Corregir la CELDA 12 para aceptar el bundle D1D cuando el marcador y los archivos estén bajo una raíz anidada del montaje
- [x] Registrar la corrida D1D real: 768 pasos CUDA, 52 saludables, router global no saludable y último paso saludable
- [x] Actualizar protocolo y handoff con la decisión D1D sin abrir holdout ni promoción (protocolo actualizado; el handoff maestro no está presente en el checkout local)
- [x] Formular protocolo D1E independiente basado en aumentar la señal densa de entropía, sin reutilizar checkpoints ni seleccionar después de observar resultados
- [x] Crear prueba estática y celda bloqueada D1E para el único peso 0.03, sin autorizar Kaggle ni GPU
- [x] Crear prueba estática y celda bloqueada D1E para el único peso 0.03, sin autorizar Kaggle ni GPU
- [x] Crear y auditar el bundle privado D1E V1 con protocolo, celda bloqueada y lanzador, excluyendo corpus y artefactos
- [x] Resolver la duplicidad de dos montajes D1E idénticos en el notebook antes de repetir la verificación bloqueada
- [x] Crear y validar una celda comparadora de copias D1E que inspeccione sólo contratos y archivos de código
- [x] Seleccionar únicamente la copia D1E con la CELDA 12 ejecutable (variante montada con sufijo `(1)`) antes de repetir la corrida

- [x] Corregir el lanzador D1E para usar `--corpus-dir`, `--tokenizer` y `--output`, y regenerar el bundle antes de repetir la corrida

- [x] Comparar las copias D1E `(1)` y `(3)` para confirmar cuál contiene el bundle corregido antes de reintentar la corrida

- [ ] Dejar un único input D1E activo en el notebook; `(1)` y `(3)` son equivalentes y cualquiera puede conservarse, pero no ambos

- [ ] Inspeccionar que `/kaggle/working/aethel-d1e-router-entropy-strength-v1` esté vacío antes de retirar sólo ese directorio residual

- [x] Registrar D1E como corrida abortada por OOM CUDA sin métricas válidas ni checkpoint reutilizable
- [x] Inspeccionar y reducir el consumo de memoria del runner sin cambiar el peso D1E ni los criterios del router
- [x] Regenerar y auditar el bundle D1E después del ajuste memory-safe antes de subir una nueva versión

- [x] Cambiar la salida de la CELDA 12 a `/kaggle/working/aethel-d1e-router-entropy-strength-v2` para no sobrescribir ni reanudar la salida v1 abortada

- [x] Conservar documentada como parcial la salida D1E v1 con `metrics_rank_0.jsonl` y `tokenizer.json`, sin borrar ni reanudar
- [x] Cambiar la salida de D1E memory-safe a `/kaggle/working/aethel-d1e-router-entropy-strength-v3` antes de regenerar el bundle

- [x] Alinear el lanzador D1E con la interfaz real del resumidor (`--metrics`, `--output`, `--diagnostic-id`) y ampliar la prueba de contrato
- [x] Regenerar y auditar el bundle D1E memory-safe después de alinear el lanzador con el resumidor
- [x] Simplificar el flujo D1E y eliminar pasos redundantes tras los errores recurrentes de montaje, residuos y variantes
- [x] Preparar una única ruta D1E verificable antes de pedir otra ejecución al usuario
- [x] Reorientar D1E hacia una corrida directa de entrenamiento real con Dataset v1 y checkpoint verificable
- [x] Auditar la configuración final de memoria y el contrato del Dataset v1 antes de ejecutar GPU
- [x] Preparar un launcher único de entrenamiento directo, sin diagnósticos D1 intermedios
- [x] Crear un cuaderno Kaggle nuevo y limpio para la corrida directa, conservando el cuaderno histórico sin modificar
- [x] Documentar la separación: código limpio en `aethel-direct-train-source-v1` y datos en `aethel-nextgen-data-v1`
- [x] Configurar el cuaderno nuevo `Aethel — Entrenamiento Directo Dataset V1` con exactamente tres celdas numeradas

- [x] Definir métricas verificables para razonamiento, bilingüismo español-inglés y matemáticas básicas
- [x] Diseñar un currículo escalonado de datos antes de ampliar el entrenamiento más allá del primer checkpoint
- [x] Proponer una ruta de escalado de arquitectura y cómputo condicionada a resultados reales, no a una métrica de IQ humano

- [x] Definir una batería de evaluaciones reproducibles de razonamiento, español-inglés y matemáticas como referencia de desempeño excepcional

- [x] Crear un dataset privado de código nuevo para entrenamiento directo y no reutilizar el contenedor histórico `aethel-nextgen-source-e0-v1`

- [x] Definir y ejecutar contratos de validación para La Roca, El Líquido, Ciclo de Sueño, MoE, memoria y neuromodulación
- [ ] Medir ultra-eficiencia con parámetros activos, VRAM, tokens por segundo, coste por token y comparación contra un baseline

- [x] Auditar el repositorio original Aethel Meta y clasificar todas sus capacidades como implementadas, parciales o previstas
- [x] Comparar las capacidades recuperadas de Aethel Meta con el núcleo actual antes de ampliar el entrenamiento
- [x] Preparar y auditar el bundle mínimo `aethel-direct-train-source-v1`, sin releases D1 históricos, datos, pesos ni métricas
- [x] Ejecutar la primera corrida directa y clasificar cada pilar como validado, telemetría presente, fallido o no ejecutado
- [ ] Corregir la inestabilidad global del router MoE antes de ampliar tokens, tamaño o promover el checkpoint directo V1
- [x] Ejecutar un baseline comparable antes de afirmar ultra-eficiencia relativa o ventaja de coste

- [x] Analizar la causa de 43/768 pasos saludables del router MoE en la corrida directa V1
- [x] Diseñar e implementar una única corrección mínima del router con criterios de éxito predefinidos
- [x] Preparar y auditar la revisión `router-selection-debias-v1` dentro del único bundle de código limpio
- [x] Actualizar `aethel-direct-train-source-v1` con la revisión del router y ejecutar la única corrida correctiva tras autorización explícita

- [x] Auditar por qué la entropía mínima y el desequilibrio máximo del router permanecen en los mismos límites tras la corrección de selección
- [x] Implementar y validar localmente jitter de selección del router, sin alterar mezcla ni entropía densa
- [x] Preparar el bundle y la guía de la única corrida correctiva `router-selection-jitter-v1`
- [x] Actualizar el único dataset de código con `router-selection-jitter-v1` y verificar las celdas 1–2 antes de nueva GPU
- [x] Ejecutar la corrida `router-selection-jitter-v1` y clasificar su mejora parcial del router
- [ ] Aislar y reducir la concentración del router en el tramo inicial que aún deja 322/768 pasos no saludables

- [x] Definir un protocolo de carga estricta y generación mínima para el checkpoint `router-selection-jitter-v1`, sin entrenamiento
- [x] Implementar y validar localmente el evaluador aislado y su recibo reproducible
- [x] Preparar el bundle y las tres celdas de Kaggle para evaluación sin ejecutar ni modificar el checkpoint

- [x] Registrar que la sesión actual de Kaggle no conserva el checkpoint jitter bajo `/kaggle/working`
- [x] Preparar una inspección de sólo lectura para localizar evidencia persistida del checkpoint jitter sin modificar artefactos
- [x] Confirmar que no hay checkpoint Aethel recuperable en `/kaggle/working` ni en los inputs montados de la sesión actual
- [x] Elegir repetir la corrida jitter en una salida inédita con preservación explícita
- [x] Preparar la repetición aislada `router-selection-jitter-v1` con salida nueva y un recibo de preservación de checkpoint
- [x] Ejecutar la repetición jitter preservable: 768 pasos, 446 saludables, checkpoint recuperable y validación de artefactos
- [x] Corregir y validar el empaquetado de preservación contra los nombres reales de artefacto, sin reentrenar ni modificar el checkpoint
- [x] Confirmar que la segunda salida jitter y su checkpoint ya no están disponibles en la sesión efímera antes de empaquetar
- [x] Rediseñar y validar la preservación dentro de la misma corrida, con paquete, compuerta `SAVE_KAGGLE_VERSION_NOW.txt` y `sync`
- [ ] Autorizar una nueva GPU sólo si el flujo de preservación corregido queda validado y se entiende el paso manual de Save Version
- [x] Definir el primer tramo largo de Aethel Edge de hasta 12 horas por sesión, con checkpoints periódicos y reanudación fiel
- [x] Investigar y seleccionar fuentes abiertas con licencia compatible para ampliar el corpus bilingüe, matemático y de razonamiento
- [x] Construir un corpus ampliado con manifiesto, deduplicación, filtros de calidad y holdout EN/ES separados
- [x] Diseñar el manifiesto Edge, los adaptadores trazables y el holdout separado sin descargar datos externos
- [x] Preparar y validar un bundle largo Edge sin iniciar Kaggle ni GPU
- [x] Auditar y reforzar el contrato de reanudación entre sesiones: pesos, optimizador, scheduler, scaler, RNG, configuración, tokenizador, hashes de datos y paso global
- [x] Probar localmente una reanudación completa CPU en dos sesiones: pasos 1–2, preservación y continuación 3–4
- [x] Comparar numéricamente una corrida continua contra la misma corrida interrumpida/reanudada: pesos, AdamW y RNG CPU equivalentes

- [x] Entregar el ZIP limpio de código y la guía final de tres celdas para el dataset y cuaderno nuevos
- [x] Corregir y validar el bundle Edge para que el ZIP incluya lanzadores, empaquetador, contratos, documentos y guía de tres celdas coherentes
- [x] Habilitar únicamente FineWeb2 EN/ES, HPLT 2.0 cleaned ES y OpenR1-Math-220k en un paquete de construcción Edge autorizado y versionado
- [x] Verificar los recursos vigentes de Oracle Cloud Free Tier y su viabilidad real para Aethel Edge
- [x] Corregir el identificador de configuración HPLT rechazado por Kaggle y añadir una prueba preventiva para la construcción Edge
- [x] Reintentar la construcción Edge sólo en una salida Kaggle inédita, preservando la salida fallida actual sin borrarla
- [x] Corregir el mínimo inglés incompatible con el límite FineWeb autorizado y rechazar esa incoherencia antes de descargar datos
- [x] Corregir la validación de listas de banderas OpenR1 y comprobar que sus ejemplos matemáticos verificados contribuyen al mínimo inglés
- [x] Corregir la incompatibilidad de `trust_remote_code` con `get_dataset_config_names` en la versión datasets de Kaggle y probar el preflight
- [x] Emitir un único ZIP Edge compatible con Kaggle y sustituir las guías anteriores por el reintento mínimo
- [x] Sustituir FineWeb2 EN inválido por FineWeb `sample-10BT` autorizado y fijar su revisión antes del reintento Edge
- [x] Incorporar el corpus Edge preservado como el único input de datos del cuaderno de entrenamiento, sustituyendo el dataset V1 sin añadir un tercer input
- [x] Crear el dataset privado reutilizable `aethel-edge-corpus-v1` desde la salida preservada del cuaderno de construcción
- [x] Preparar y validar un release de código específico para el primer entrenamiento largo Edge, con guía de exactamente tres celdas y sin iniciarlo
- [x] Verificar en Kaggle que el cuaderno de entrenamiento tiene exactamente los inputs de código Edge y `aethel-edge-corpus-v1`
- [x] Solicitar autorización explícita separada antes de ejecutar la primera sesión GPU Edge
- [x] Corregir y validar la resolución del root de shards en la CELDA 1 Edge tras el fallo seguro de conteo observado en Kaggle
- [x] Inventariar de forma no destructiva los nombres y rutas de archivos expuestos por la versión montada de `aethel-edge-corpus-v1`
- [x] Adaptar y validar el lector Edge y la verificación de integridad para los shards `.jsonl` descomprimidos por Kaggle
- [x] Confirmar la ejecución y preservación mediante Save Version de la primera sesión GPU Edge autorizada
- [x] Preparar y validar la evaluación aislada autorizada del checkpoint Edge `latest.pt` sin reanudar entrenamiento
- [ ] Crear el dataset privado `aethel-edge-phase1-artifacts-v1` desde la salida preservada y ejecutar la evaluación Edge aislada autorizada
- [x] Crear el dataset privado `aethel-edge-phase1-artifacts-v1` desde la salida preservada; queda pendiente ejecutar la evaluación aislada autorizada
- [x] Corregir el preflight de evaluación para seleccionar el checkpoint canónico y excluir la copia extraída del TAR en Kaggle
- [x] Entregar las celdas completas de evaluación Edge directamente en el chat para copia manual
- [x] Actualizar el repositorio Aethel con código, documentación y registro verificable de la sesión Edge preservada
- [ ] Definir y aplicar un mecanismo compatible para almacenar los pesos Edge grandes sin exceder los límites de GitHub

- [x] Crear `AETHEL_PRO_SPEC.md` con arquitectura Pro, parámetros, fórmulas de conteo y cálculos reproducibles de VRAM, distinguiendo diseño hipotético de evidencia validada.

- [x] Diseñar una estrategia fuera de la caja para maximizar eficiencia e inteligencia verificable por unidad de cómputo, con hipótesis, ablations y criterios de rechazo.

- [x] Evaluar la viabilidad de una arquitectura Aethel de hasta 100T parámetros con MoE y entrenamiento distribuido, incluyendo memoria, cómputo, comunicación y una progresión de escalas.

- [x] Investigar si Aethel puede ampliar capacidad durante el aprendizaje mediante nuevos expertos, módulos o parámetros, y distinguirlo de memoria externa y recuperación.

- [x] Añadir diagnóstico CPU determinista de asignación dura del router para cobertura, concentración, entropía y overflow, sin modificar el routing principal.

- [x] Revisar Aethel para exigir competencia base en conversación EN/ES y razonamiento antes de añadir módulos o parámetros dinámicos.

- [x] Añadir y ejecutar un contrato estático de `AETHEL_BASE_CAPABILITY_SPEC.md` para verificar objetivos EN/ES, razonamiento, matemáticas, crecimiento y límites honestos.

- [x] Integrar en `last_routing_stats` cobertura y densidades de asignación dura top-k, y validar sintaxis, diagnóstico, presupuesto y especificación base en CPU.

- [x] Probar en CPU si la temperatura positiva cambia la asignación determinista top-k; resultado: cambia entropía suave, no cambia índices duros.

- [x] Añadir una puerta opcional de salud top-k compatible hacia atrás y probar rechazo de concentración/overflow y aceptación de balanceo en CPU.

- [x] Evaluar en CPU si ruido reproducible antes de top-k reduce concentración inicial sin perder determinismo; resultado documentado, sin integrar al entrenamiento.
- [x] Diseñar el plan operativo de entrenamiento de Aethel Base 100M–300M con recursos actuales, priorizando 100M, sesiones reanudables, evaluación y puertas de escalado.

- [x] Consolidar la arquitectura integral de Aethel en una documentación maestra coherente con el código real.
- [x] Documentar el funcionamiento de Sólido, Líquido, Sueño, memoria, curiosidad, espacio de trabajo global y neuromodulación.
- [x] Documentar el flujo completo de datos, entrenamiento, inferencia, checkpoints, evaluación y publicación de artefactos.
- [x] Documentar la estrategia revolucionaria de eficiencia y expansión dinámica como hipótesis gobernadas por experimentos, no como capacidades demostradas.
- [x] Validar referencias cruzadas, contratos técnicos, límites de evidencia y guardar checkpoint de la documentación consolidada.
- [x] Crear un índice navegable de la documentación técnica y verificar sus enlaces internos.
- [x] Documentar los bloqueos externos restantes de Aethel y el procedimiento exacto para reanudar Kaggle, GPU, FSDP, Triton y servicio persistente.
- [x] Auditar si Google AI Studio eliminó o alteró archivos del repositorio y preservar/restaurar sólo cambios verificados.
- [ ] Sincronizar con GitHub el estado local verificado `59f1b1d`, preservando exclusiones y sin publicar datos privados.
