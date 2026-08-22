# Aethel — Documento integral de continuidad

**Fecha:** 22 de agosto de 2026  
**Autor:** Manus AI  
**Proyecto:** `/home/ubuntu/aethel-platform`  
**Último checkpoint consolidado:** `manus-webdev://18f74cef`  
**Propósito:** permitir que un nuevo chat retome el proyecto Aethel sin reconstruir contexto, sin activar entrenamiento por accidente y sin presentar resultados hipotéticos como evidencia.

---

## 1. Resumen ejecutivo

**Aethel** es un proyecto para crear un modelo de lenguaje bilingüe español–inglés propio, acompañado de una arquitectura cognitiva gobernada. No se trata de una interfaz que simplemente llame a un modelo externo: el objetivo es entrenar pesos propios, evaluar sus capacidades con datos retenidos y evolucionar hacia un producto B2B de inteligencia técnica privada.

El proyecto tiene tres realidades que deben mantenerse separadas:

| Área | Estado actual | Qué no debe afirmarse todavía |
|---|---|---|
| **Software y arquitectura** | Implementados y probados en CPU: Transformer con RoPE, GQA, MoE, memoria, curiosidad, candidatos LoRA de Sueño y contratos de seguridad. | Que los kernels GPU estén validados o que el modelo haya aprendido. |
| **Datos** | Corpus bilingüe real, congelado y validado localmente: 40.000 documentos, 22 shards, manifiestos y tokenizador. | Que ya esté cargado a Kaggle o usado en entrenamiento. |
| **Entrenamiento y producto** | Receta Seed, evaluador, checkpoints y runbook listos. Dashboard web operativo. | Que exista un checkpoint GPU, pérdida, perplejidad, benchmark o producto comercial de modelo propio. |

> **Regla principal del proyecto:** no se usan simulaciones como métricas, resultados de entrenamiento ni pruebas de capacidad. Cada cifra de calidad, rendimiento, consumo o benchmark debe provenir de una corrida real y conservar su artefacto verificable.

La siguiente meta técnica no es Aethel Edge de 2,2 B parámetros. Es **Aethel Seed E0**, una corrida pequeña pero completamente real que pruebe la cadena: Dataset congelado → tokenizador → entrenamiento → checkpoint atómico → reanudación → evaluación de holdout en inglés y español → generación token a token. Sólo después de eso tiene sentido escalar.

---

## 2. Objetivo de producto y límites responsables

La propuesta inicial es **Aethel Workspace**, un espacio bilingüe de inteligencia técnica para equipos que trabajan con documentación privada y aprobada. El modelo debe servir para recuperar, explicar, comparar y redactar sobre esas fuentes, con memoria de sesión gobernada, procedencia y reversibilidad.

La intención del usuario es construir una IA que pueda organizar conocimiento, razonar, aprender de manera controlada y mejorar con el tiempo. La implementación traduce esa intención a mecanismos auditables, no a afirmaciones de conciencia o de equivalencia con un cerebro humano.

| Capacidad buscada | Mecanismo Aethel | Límite obligatorio |
|---|---|---|
| Continuidad y recuerdos | El Líquido, memoria de trabajo y recuperación con procedencia. | No cambia La Roca ni comparte memoria entre espacios sin política. |
| Aprendizaje controlado | Candidatos LoRA de Sueño, replay curado y promociones por gates. | No hay autoentrenamiento ni promoción automática. |
| Curiosidad | Señales de incertidumbre, novedad, contradicción y progreso esperado. | Sólo propone observar, recuperar localmente, pedir aclaración o proponer replay. |
| Eficiencia | GQA, RoPE, MoE top-2, LoRA y kernels especializados. | La eficiencia debe medirse en hardware real, no suponerse. |
| Capacidad comercial | Modelo propio más conocimiento privado, auditoría y rollback. | No se anuncia como servicio comercial hasta demostrar calidad, seguridad, operación y coste. |

No se deben inventar reseñas, testimonios, usuarios, ratings, benchmarks ni métricas de calidad. Tampoco se puede “destilar” conocimiento interno, pesos, cadenas de razonamiento privadas o datos propietarios de este asistente. El conocimiento de Aethel se debe construir desde fuentes con procedencia y derechos de uso documentados.

---

## 3. Restricciones vigentes que el nuevo chat debe respetar

| Restricción | Interpretación operativa |
|---|---|
| **Datos reales** | Cada Dataset conserva fuente, licencia, idioma, dominio, hash y split. No se presentan datos sintéticos como corpus real. |
| **Bilingüismo nativo** | Español e inglés se mantienen balanceados y las evaluaciones se reportan por separado. |
| **Holdout protegido** | Holdout no entra en tokenizador, entrenamiento, replay, ajuste o selección de candidato. |
| **Dataset congelado** | No se altera localmente sin una nueva versión, manifiesto, hashes y validación. |
| **Sin GPU ni entrenamiento automático** | Sólo ejecutar GPU/Kaggle tras autorización explícita y con el alcance exacto de la corrida. |
| **No publicar sin confirmación** | Crear un Dataset privado en Kaggle es una acción externa y requiere confirmación justo antes de pulsar `Create`. No hacerlo público. |
| **No impulsar a producción por fallback** | E0 con PyTorch, si se autoriza, es laboratorio. No satisface el contrato comercial Triton ni permite promocionar La Roca. |
| **My Browser** | Si el usuario exige usar sólo My Browser, nunca recurrir al navegador aislado. Si la herramienta indica `Browser: Sandbox`, detenerse e informar el bloqueo. |

---

## 4. Estado del repositorio y de los checkpoints

El proyecto está en `/home/ubuntu/aethel-platform` y utiliza React 19 + Tailwind 4 + Express 4 + tRPC 11 para la plataforma web. El motor cognitivo, Dataset y entrenamiento viven como código Python/PyTorch dentro del mismo repositorio.

| Checkpoint | Contenido principal |
|---|---|
| `manus-webdev://18f74cef` | Especificaciones de infraestructura, topología concurrente, producto Workspace, variantes, routing top-2 y ruta comercial. Vitest 5/5. |
| `manus-webdev://9fa4e583` | Lanzador Seed offline, runbook, validación de paquete/tokenizador antes de GPU y pruebas de rechazo. Sin GPU ni Kaggle. |
| `manus-webdev://67e2dcfe` | Evaluación de plataformas gratuitas: Kaggle para piloto Seed, Colab como respaldo, ZeroGPU descartado para entrenamiento prolongado. |
| `manus-webdev://78723e42` | Contratos estrictos que bloquean prefill CUDA y dispatch/combina MoE en ausencia de kernels Triton completos validados. |

Al redactarse este documento existen cambios locales aún no encapsulados en un checkpoint nuevo: `todo.md` y `training/KAGGLE_BROWSER_SESSION_CHECK_2026-08-22.md`, junto con este documento de traspaso. Antes de entregar cambios posteriores, revisar todo el `todo.md`, ejecutar las pruebas pertinentes y guardar un checkpoint.

---

## 5. Aplicación web actual

La web es un dashboard oscuro de Aethel y funciona de manera independiente del entrenamiento de pesos propios. Tiene Chat, visualizador de arquitectura, Trainer, Benchmarks y Engine Status. El chat usa un modelo de plataforma con una persona Aethel V3; por lo tanto, no debe confundirse con inferencia de un checkpoint Aethel entrenado.

| Componente | Ruta principal | Estado |
|---|---|---|
| Dashboard principal | `client/src/pages/Home.tsx` | Operativo; visualiza los cinco pilares y telemetría. |
| API tRPC | `server/routers.ts` | Guarda historial y provee procedimientos del dashboard. |
| Layout y chat | `client/src/components/` | Chat y navegación disponibles. |
| Pruebas web | `server/*.test.ts` | Última comprobación registrada: Vitest 5/5. |

Los paneles de benchmarks muestran la ausencia de resultados cuando no hay artefactos reales. Esa conducta debe preservarse: no llenar el dashboard con cifras de entrenamiento ficticias.

---

## 6. Motor Aethel y mapa de código

### 6.1 Núcleo Transformer

| Archivo | Responsabilidad |
|---|---|
| `engine/aethel_model.py` | Transformer con RoPE, GQA, Sparse MoE, KV-cache, LoRA y contratos de Triton. |
| `engine/triton_bridge.py` | Kernels/fallbacks CPU para SwiGLU, atención causal de decode y router top-2. |
| `engine/train_aethel_gpu.py` | Entrenador GPU con checkpoints, estados de optimizador y reanudación. |
| `engine/evaluate_nextgen.py` | Evaluación de checkpoint sobre corpus retenido. |
| `engine/export_artifacts.py` | Exportación de artefactos de corrida. |

El modelo utiliza **RoPE** para posiciones, **GQA** para reducir coste del KV-cache y **MoE sparse top-2** para activar sólo dos expertos por token. Su eficiencia futura depende de validar la totalidad de la ruta CUDA, no sólo de implementar fallbacks CPU.

### 6.2 Arquitectura cognitiva

| Pilar | Implementación | Garantía actual |
|---|---|---|
| **La Roca / Sólido** | Proyección estable, ancla congelada y manifest hashable. | No se muta en observación ni durante una conversación. |
| **El Líquido** | Traza hebbiana, snapshots versionados y ledger de eventos. | Guarda procedencia, TTL y `eligible_for_sleep=false` por defecto. |
| **Ciclo de Sueño** | Replay, candidato LoRA aislado, preflight y máquina de estados. | No entrena ni promociona sin aprobación, evidencia y hashes. |
| **Neuromodulación** | Señal de sorpresa/prioridad en rango [0,1]. | Se puede registrar; no equivale a aprendizaje sin permisos. |
| **Espacio de Trabajo Global** | Fusión gated de tres fuentes. | Opera como estructura de control; no expone razonamientos internos. |
| **Memoria de trabajo** | `GRUCell` por sesión. | Estado acotado al contexto y la sesión. |
| **Refinamiento adaptativo** | Pasos extra controlados por presupuesto. | Telemetría de presupuesto y coste. |
| **Curiosidad** | Incertidumbre, novedad, contradicción y progreso esperado. | Sólo emite propuestas locales sin acciones externas. |

El archivo central de esta capa es `engine/aethel_nextgen.py`. Las pruebas CPU más relevantes son `engine/test_memory_reasoning.py`, `engine/test_sleep_candidate.py`, `engine/test_sleep_replay_admission.py`, `engine/test_sleep_preflight.py`, `engine/test_sleep_state_machine.py` y `engine/test_sleep_orchestrator.py`.

### 6.3 Sueño como protocolo de cambios reversibles

El Sueño no es un entrenamiento continuo oculto. Su secuencia es:

```text
evento líquido con TTL y procedencia
→ curación independiente
→ aprobación independiente ligada por hash
→ admisión de replay sin holdout
→ candidato LoRA aislado en cuarentena
→ preflight (La Roca + Dataset + tokenizer + replay)
→ máquina de estados y evaluación
→ promoción humana o rollback por descarte
```

Los archivos de contrato son `training/AETHEL_SLEEP_CANDIDATE_CONTRACT_V1.md`, `training/AETHEL_SLEEP_REPLAY_ADMISSION_CONTRACT_V1.md`, `training/AETHEL_SLEEP_PREFLIGHT_CONTRACT_V1.md` y `training/AETHEL_SLEEP_STATE_MACHINE_CONTRACT_V1.md`.

---

## 7. Dataset bilingüe congelado

El corpus local real está en:

```text
/home/ubuntu/aethel-knowledge-corpus-v1-package/
```

Proviene de documentos de Wikipedia procesados desde fuentes oficiales de Wikimedia, con licencia CC BY-SA 4.0 registrada en los manifests. Contiene 40.000 documentos: 20.000 en inglés y 20.000 en español. Tras deduplicación y split, la composición validada es la siguiente.

| Split | Inglés | Español | Total |
|---|---:|---:|---:|
| `train` | 19.011 | 19.012 | 38.023 |
| `holdout` | 989 | 988 | 1.977 |
| **Total** | **20.000** | **20.000** | **40.000** |

El tokenizador BPE de 32.000 tokens se entrenó exclusivamente sobre `train`. El paquete tiene 22 shards comprimidos JSONL, junto a `tokenizer.json`, `metadata.json`, `package_manifest.json` y `validation_report.json`. El validador registró `valid=true` y cero solicitudes de red para la comprobación final.

| Archivo de Dataset | Responsabilidad |
|---|---|
| `training/materialize_aethel_knowledge_corpus.py` | Materialización offline inicial del corpus; no ejecutar dentro de Kaggle. |
| `training/validate_aethel_knowledge_corpus.py` | Valida origen, esquema, hashes, idioma y separación de holdout. |
| `training/package_aethel_knowledge_corpus.py` | Fragmenta, tokeniza y produce manifiestos. |
| `training/validate_aethel_knowledge_package.py` | Gate final de integridad del paquete congelado. |
| `training/AETHEL_KNOWLEDGE_DATASET_README.md` | Descripción de uso, procedencia y límites. |
| `training/LOCAL_DATASET_FREEZE_2026-08-19.md` | Registro formal de la congelación local. |

### 7.1 Estado de Kaggle y lección de ejecuciones anteriores

El Dataset congelado todavía no ha sido creado como el Dataset privado final `aethel-nextgen-data-v1`. El usuario pidió orientación y podría decidir crearlo manualmente. No se debe subir nada sin una confirmación específica antes de pulsar `Create`.

Las ejecuciones antiguas de Kaggle fallaron porque intentaban descargar o construir corpus dentro del Notebook: errores HTTP 429, HTTP 502, descargas incompletas y umbrales insuficientes de inglés. Esos errores no se deben repetir. La receta vigente no realiza descargas de corpus: Kaggle debe recibir primero el paquete cerrado como entrada privada.

La estructura esperada para el Dataset privado es:

```text
aethel-nextgen-data-v1/
├── corpus/
│   ├── train-*.jsonl.gz
│   ├── holdout-en-00000.jsonl.gz
│   └── holdout-es-00000.jsonl.gz
├── tokenizer.json
├── metadata.json
├── package_manifest.json
└── validation_report.json
```

No incluir checkpoints, conversaciones, secretos, bundles de código antiguos ni materializadores de datos dentro de este Dataset.

---

## 8. Aethel Seed E0: receta de entrenamiento preparada

La receta de inicio es intencionalmente pequeña: **Aethel Seed E0**. No es Edge ni un lanzamiento comercial. Es el primer experimento que demuestra que el sistema completo puede entrenar y recuperarse sobre datos reales.

| Parámetro de Seed | Valor de arranque propuesto | Estado |
|---|---:|---|
| Capas | 4 | Objetivo de receta, no corrida ejecutada. |
| Dimensión | 512 | Objetivo de receta. |
| Attention heads / KV heads | 8 / 2 | Objetivo de receta. |
| Expertos / activos | 8 / 2 | Objetivo de receta. |
| Secuencia | 1.024 | Objetivo de receta. |
| Batch / acumulación | 2 / 16 | Objetivo de receta. |
| Máximo de pasos | 4.992 | Objetivo de receta; adaptar a la ventana real. |
| Guardado | Cada 192 pasos, 3 snapshots | Mecanismo implementado, no probado en GPU aún. |

Los principales archivos son:

| Archivo | Función |
|---|---|
| `training/run_kaggle_seed_offline.sh` | Lanzador offline que valida paquete, controla autorización, GPU, Triton, entrenamiento, inspección y evaluación. |
| `training/test_kaggle_seed_offline.py` | Prueba estática de que la ruta bloquea falta de autorización, paquete o tokenizador incompatibles. |
| `training/AETHEL_SEED_OFFLINE_RUNBOOK_V1.md` | Instrucciones y límites de la corrida Seed. |
| `training/validate_training_readiness.py` | Verificación adicional de preparación de entrenamiento. |
| `training/inspect_checkpoint.py` | Confirma el contenido reproducible de checkpoints. |

El launcher debe establecer como mínimo:

```bash
export AETHEL_SOURCE_DIR=/kaggle/working/aethel-nextgen-source
export AETHEL_DATA_DIR=/kaggle/input/aethel-nextgen-data-v1
export AETHEL_OUTPUT_DIR=/kaggle/working/aethel-runs/aethel-seed-e0
export AETHEL_RUN_AUTHORIZED=YES
```

Sin `AETHEL_RUN_AUTHORIZED=YES` termina con bloqueo deliberado. Antes de cualquier CUDA valida el paquete completo. Sólo entrena `train` y evalúa los holds `holdout-en-00000.jsonl.gz` y `holdout-es-00000.jsonl.gz` por separado.

### 8.1 Artefactos que deben salir de una E0 real

| Artefacto | Significado |
|---|---|
| `package_preflight.json` | Resultado de integridad de Dataset, tokenizer, shards y split. |
| `latest.pt` | Checkpoint reanudable: pesos, optimizador, paso y configuración. |
| `step_*.pt` | Snapshots de contingencia. |
| `recovery_receipt.json` | Recibo de recuperación y paso seguro. |
| `metrics_rank_0.jsonl` | Métricas reales: pérdida, tokens/s, VRAM y salud del router. |
| `checkpoint_inspection.json` | Confirmación de metadatos reproducibles. |
| `evaluation_holdout_en.json` | Evaluación inglesa retenida. |
| `evaluation_holdout_es.json` | Evaluación española retenida. |

Mientras esos archivos no existan, no se debe afirmar pérdida, perplejidad, capacidad de generación, benchmark, VRAM o rendimiento de Aethel.

### 8.2 Contrato de Triton y fallback E0

El usuario pidió Triton como requisito industrial. Aethel por ello bloquea CUDA bajo `require_triton=True` cuando faltan dos componentes todavía no validados en GPU: prefill causal por bloques y dispatch/combina MoE completo. Los fallbacks CPU existen y las políticas puras están probadas, pero no reemplazan validación numérica y de rendimiento sobre CUDA.

Un E0 experimental puede usar fallback PyTorch sólo con una segunda aprobación explícita:

```bash
export AETHEL_LAB_FALLBACK_AUTHORIZED=YES
```

Ese valor no relaja el contrato de producto. Sólo permite una línea base experimental para conocer pérdida, reanudación, memoria y trazas reales. Ningún checkpoint de fallback puede ser promovido a La Roca comercial ni anunciado como Pro/Edge listo para producción.

---

## 9. Hardware y plataformas

La máquina de desarrollo actual no tiene CUDA disponible; la comprobación encontró seis CPU lógicas y aproximadamente 3,8 GiB de RAM. Puede ejecutar validaciones CPU, empaquetar datos y desarrollar, pero no entrenar Edge ni sostener inferencia comercial.

| Plataforma | Uso correcto | Límite |
|---|---|---|
| Kaggle Notebooks | Piloto Seed, validación de CUDA y pruebas cortas reanudables. | Sesiones y cuotas temporales; no es infraestructura comercial. |
| Google Colab gratuito | Respaldo para smoke tests puntuales. | Disponibilidad y duración no garantizadas. |
| Hugging Face ZeroGPU | Demos/inferencia breve. | No sirve para entrenamiento prolongado. |
| PC del usuario con NVIDIA | Posible ruta sin cuota si dispone de VRAM, energía y almacenamiento suficientes. | Debe verificarse GPU, drivers, RAM y persistencia. |
| Servidor GPU persistente | Entrenamiento Edge y servicio comercial posterior. | Requiere autorización, presupuesto, seguridad y monitoreo. |

Kaggle puede mostrar varias opciones de acelerador. Si aparecen **T4 ×2**, **P100** y **TPU**, la prioridad para Seed es T4 ×2, luego P100; TPU no es la ruta actual porque el motor se basa en PyTorch/CUDA/Triton. Las dos T4 no suman VRAM automáticamente; distribución/FSDP es una etapa posterior y debe validarse por separado.

El análisis documentado se encuentra en `training/FREE_GPU_PLATFORM_ASSESSMENT_2026-08-21.md` y `training/AETHEL_INFRASTRUCTURE_CAPACITY_AND_SCALING_V1.md`.

---

## 10. Variantes de modelo y top-2 MoE

| Familia | Especificación de diseño | Uso | Estado |
|---|---|---|---|
| **Seed E0** | 4 capas, dim 512, 8 expertos top-2. | Validar la canalización real. | Preparado; sin entrenamiento GPU. |
| **Edge** | 28 capas, dim 2.560, 20 cabezas, 5 KV heads, denso. | Asistente técnico privado cuantizable. | Objetivo ≈2,2 B; no entrenado. |
| **Pro** | 32 capas, dim 4.096, 32 heads, 8 KV heads, 8 expertos, top-2, SwiGLU 11.008. | Investigación y producto empresarial. | Objetivo ≈36,4 B totales / ≈10,4 B activos; no implementado a escala. |
| **Research** | Familia futura multi-GPU. | Investigación. | Referencia ≈139 B; sin microarquitectura congelada. |

En Pro, el router produce ocho logits por token, selecciona los dos expertos de mayor puntuación, normaliza sus gates entre ambos, aplica capacidad, agrupa tokens, ejecuta expertos SwiGLU y combina sus salidas ponderadas. El sistema debe registrar entropía, carga por experto, overflow, capacidad, buffers, gates y errores numéricos. La especificación completa está en `training/AETHEL_PRO_TOP2_ROUTING_SPEC_V1.md`.

---

## 11. Topología concurrente de producción prevista

Aethel no convierte cada subsistema cognitivo en una cadena token a token. La inferencia de La Roca se mantiene corta y determinista; memoria, curiosidad y Sueño viven en ritmos separados.

```text
Solicitud autorizada
    → recuperación permitida / memoria de sesión
    → La Roca (GPU) y respuesta autoregresiva
    → telemetría versionada
    → El Líquido / curiosidad local (CPU o Rust)
    → candidatos de replay bloqueados por defecto
    → Sueño aislado sólo tras curación, aprobación y preflight
```

| Dominio | Ejecutor previsto | Puede escribir | No puede hacer |
|---|---|---|---|
| La Roca | GPU de inferencia. | Nada durante la conversación. | Modificar pesos base. |
| Memoria de trabajo | CPU de sesión. | Estado acotado. | Cruzar tenants sin control. |
| El Líquido | Servicio CPU/Rust. | Eventos con TTL/procedencia. | Escribir en La Roca. |
| Curiosidad | Controlador CPU. | Propuestas locales. | Llamar servicios externos o entrenar. |
| Sueño | Worker GPU aislado. | Candidato LoRA en cuarentena. | Acceder holdout o promocionar solo. |

La arquitectura comercial deberá desplegar un servicio persistente para memoria/gobierno y una ruta GPU para inferencia. Ninguna de esas dos piezas está desplegada 24/7 todavía.

---

## 12. Documentos de referencia obligatoria

| Documento | Cuándo leerlo |
|---|---|
| `ARCHITECTURE_COGNITIVE_OPERATING_MODEL.md` | Antes de alterar la arquitectura o sus ritmos cognitivos. |
| `training/AETHEL_COGNITIVE_EXPERIMENT_CONTRACT_V1.md` | Antes de ejecutar E0, ablations, promoción o Sueño. |
| `training/AETHEL_CURIOSITY_CONTROLLER_SPEC_V1.md` | Antes de cambiar curiosidad o permitir nuevas acciones. |
| `training/AETHEL_SEED_OFFLINE_RUNBOOK_V1.md` | Antes de configurar un Notebook GPU. |
| `training/FREE_GPU_PLATFORM_ASSESSMENT_2026-08-21.md` | Antes de elegir plataforma gratuita. |
| `training/AETHEL_CONCURRENT_EXECUTION_TOPOLOGY_V1.md` | Antes de implementar servicios persistentes o concurrencia. |
| `training/AETHEL_TECHNICAL_VARIANTS_SPEC_V1.md` | Antes de fijar una configuración Seed, Edge o Pro. |
| `training/AETHEL_COMMERCIAL_PRODUCT_SPEC_V1.md` | Antes de presentar el producto o definir un piloto. |
| `training/AETHEL_TRAINING_TO_COMMERCIAL_ROADMAP_V1.md` | Antes de escalar presupuesto, hardware o alcance comercial. |
| `training/KAGGLE_BROWSER_SESSION_CHECK_2026-08-22.md` | Antes de volver a usar My Browser/Kaggle en este chat. |

---

## 13. Estado de My Browser y Kaggle al cerrar este chat

El usuario intentó habilitar **Mi navegador** en Manus Desktop. El conector aparecía activado y posteriormente el usuario mostró una barra que decía: `Manus AI Browser Operator comenzó a depurar este navegador`.

Sin embargo, cada comprobación realizada desde el agente devolvió explícitamente `Browser: Sandbox`. Kaggle se mostró como sesión no autenticada y no apareció ninguna tarjeta de conexión en la herramienta. Por el requisito explícito del usuario de no usar el navegador aislado, se debe mantener bloqueada cualquier nueva operación Kaggle hasta que My Browser sea realmente expuesto como sesión personal a la conversación.

> **No se creó ningún Dataset, no se subió ningún archivo, no se creó Notebook, no se consumió GPU ni se ejecutó entrenamiento durante esos intentos.**

Si el usuario decide hacer manualmente el paso de Dataset privado, el nuevo chat debe limitarse a verificar la estructura a partir de una captura o una URL, sin afirmar que pudo usar su navegador.

---

## 14. Próximos pasos ordenados

### Ruta A — el usuario crea el Dataset privado de forma manual

1. Crear en Kaggle un Dataset privado llamado `aethel-nextgen-data-v1`.
2. Subir los 22 shards bajo `corpus/` más los cuatro archivos raíz de manifiesto/tokenizador.
3. Verificar que la pestaña `Data` muestra la estructura correcta y que la visibilidad dice `Private`.
4. Compartir una captura o URL con el nuevo chat.
5. Preparar un Notebook privado que adjunte el Dataset de datos y la versión actual del Dataset de código.
6. Verificar inputs y ejecutar primero el preflight CPU/offline, sin GPU si no hay autorización de entrenamiento.
7. Pedir autorización explícita para la GPU y, por separado, para fallback E0 si Triton completo sigue pendiente.

### Ruta B — My Browser se conecta correctamente

1. Comprobar que la herramienta de navegador ya no informa `Browser: Sandbox`.
2. Abrir Kaggle usando exclusivamente la sesión personal conectada.
3. Verificar si la cuenta está autenticada; si aparece login o CAPTCHA, detenerse y pedir al usuario que intervenga.
4. Preparar el formulario `New Dataset` con los valores privados, sin pulsar `Create`.
5. Pedir confirmación explícita justo antes de crear/publicar el Dataset privado.
6. Después de creado, comprobar los archivos en `Data`; todavía no activar GPU.

### Ruta C — sólo después de autorización de entrenamiento

1. Adjuntar Dataset de datos y código al Notebook privado.
2. Elegir T4 ×2 si está disponible; registrar GPU, CUDA, VRAM y versiones.
3. Ejecutar `run_kaggle_seed_offline.sh` con `AETHEL_RUN_AUTHORIZED=YES`.
4. Si el contrato Triton bloquea CUDA, detenerse o solicitar autorización separada para `AETHEL_LAB_FALLBACK_AUTHORIZED=YES` como E0 de laboratorio.
5. Verificar artefactos de checkpoint, reanudación y evaluaciones en/es.
6. Guardar el resultado como artefactos privados de forma que el siguiente Notebook pueda recuperar el checkpoint.
7. Reportar sólo los valores observados, más su configuración, hash y limitaciones.

---

## 15. Instrucción sugerida para abrir el siguiente chat

Copiar este texto, adjuntar este documento y el agente podrá continuar con el alcance correcto:

> Lee `AETHEL_PROJECT_HANDOFF_2026-08-22.md` completo antes de actuar. Quiero continuar Aethel con datos reales, sin simulaciones y sin presentar métricas inexistentes. El Dataset local está congelado en `/home/ubuntu/aethel-knowledge-corpus-v1-package/`; no actives Kaggle, GPU ni entrenamiento sin mi autorización explícita. Respeta la separación de holdout, La Roca/Líquido/Sueño, los contratos Triton y los gates de promoción. Si solicito usar My Browser, no uses navegador aislado: detente si la herramienta indica `Browser: Sandbox`. Primero verifica el estado actual de `todo.md`, los checkpoints y el Dataset privado en Kaggle si existe.

---

## 16. Referencias internas y externas

Las afirmaciones de estado de este documento se fundamentan en los archivos y checkpoints del repositorio citados anteriormente. Para límites y selección de infraestructura gratuita, consultar las fuentes primarias incluidas en `training/FREE_GPU_PLATFORM_ASSESSMENT_2026-08-21.md`.

[1] [Kaggle — Efficient GPU Usage Tips](https://www.kaggle.com/docs/efficient-gpu-usage)  
[2] [Google Colab — Frequently Asked Questions](https://research.google.com/colaboratory/faq.html)  
[3] [Hugging Face — ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu)  
[4] [NVIDIA — H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)
