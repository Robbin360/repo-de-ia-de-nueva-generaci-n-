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
- [x] Bloquear explícitamente el prefill CUDA cuando `require_triton` esté activo hasta disponer de un kernel causal validado en GPU
- [x] Bloquear explícitamente el dispatch/combina MoE CUDA cuando `require_triton` esté activo hasta disponer de un kernel Triton validado
- [ ] Documentar la capacidad de la infraestructura actual y los escenarios de escalamiento de cómputo, memoria, red y datos requeridos para Aethel
- [ ] Documentar la topología operativa de Aethel: recursos para Sólido, Líquido, curiosidad, memoria, Sueño y servicios persistentes sin confundirlos con entrenamiento
- [ ] Definir una topología de ejecución concurrente que separe el bucle autoregresivo del modelo de los servicios líquidos, memoria, curiosidad y Sueño
- [ ] Definir un producto comercial inicial para Aethel, su propuesta de valor, operación, límites y criterios de lanzamiento
- [ ] Definir una ficha técnica escalonada de Aethel con parámetros, expertos, capas, contexto, memoria y requisitos de inferencia por variante
- [ ] Documentar el routing top-2 de Aethel Pro, incluyendo balanceo, capacidad, dispatch, combinación y contrato Triton de producción
- [ ] Definir la ruta de Aethel desde entrenamiento autorizado hasta operación comercial persistente, con infraestructura, seguridad y criterios de promoción
- [ ] Entrenar y evaluar un modelo propio Aethel Edge con el Dataset trazable sólo cuando exista GPU autorizada, sin usar extracción de conocimiento interno ni resultados simulados
- [ ] Evaluar y documentar una ruta local reproducible con GPU para entrenar Aethel Seed y Edge sin depender de sesiones Kaggle
- [x] Comparar plataformas gratuitas vigentes de GPU para el piloto Aethel, incluyendo límites de sesión, almacenamiento y persistencia de checkpoints
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
