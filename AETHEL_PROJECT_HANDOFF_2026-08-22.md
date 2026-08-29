# Aethel — Documento integral de continuidad

**Fecha:** 22 de agosto de 2026  
**Autor:** Manus AI  
**Proyecto:** `/home/ubuntu/aethel-platform`  
**Último checkpoint consolidado:** `manus-webdev://26256b0f`
**Estado de sincronización GitHub:** requiere una nueva sincronización tras registrar la evidencia V8 aprobada.
**Propósito:** permitir que un nuevo chat retome el proyecto Aethel sin reconstruir contexto, conservando la historia de decisiones, sin activar entrenamiento por accidente y sin presentar resultados hipotéticos como evidencia.

---

## 1. Resumen ejecutivo

**Aethel** es un proyecto para crear un modelo de lenguaje bilingüe español–inglés propio, acompañado de una arquitectura cognitiva gobernada. No se trata de una interfaz que simplemente llame a un modelo externo: el objetivo es entrenar pesos propios, evaluar sus capacidades con datos retenidos y evolucionar hacia un producto B2B de inteligencia técnica privada.

El proyecto tiene tres realidades que deben mantenerse separadas:

| Área | Estado actual | Qué no debe afirmarse todavía |
|---|---|---|
| **Software y arquitectura** | Implementados y probados en CPU: Transformer con RoPE, GQA, MoE, memoria, curiosidad, candidatos LoRA de Sueño y contratos de seguridad. La aceptación Triton V4 fue experimental; V8 confirmó en Kaggle la alineación CUDA de `memory_state` y El Líquido mediante un smoke aislado. | Que las rutas Triton estrictas estén validadas. |
| **Datos** | Corpus bilingüe real, congelado y validado en Kaggle: 40.000 documentos, 22 shards, manifiestos y tokenizador; el holdout se evaluó de forma separada en EN y ES al cierre de E0 V8. | Que los datos o una sola corrida impliquen calidad de modelo. |
| **Entrenamiento y producto** | V5 inició E0 real y se abortó antes de checkpoint; V8 completó el E0 planificado y persistió checkpoint, recibo, métricas y evaluaciones holdout. Dashboard web operativo. | Que exista benchmark comparativo, utilidad demostrada, soporte Triton estricto o producto comercial de modelo propio. |

> **Regla principal del proyecto:** no se usan simulaciones como métricas, resultados de entrenamiento ni pruebas de capacidad. Cada cifra de calidad, rendimiento, consumo o benchmark debe provenir de una corrida real y conservar su artefacto verificable.

La siguiente meta técnica no es Aethel Edge de 2,2 B parámetros. Es **Aethel Seed E0**, una corrida pequeña pero completamente real que pruebe la cadena: Dataset congelado → tokenizador → entrenamiento → checkpoint atómico → reanudación → evaluación de holdout en inglés y español → generación token a token. Sólo después de eso tiene sentido escalar.

### Actualización operativa posterior: V5 abortado, puertas V6–V8 y E0 V8 completado

El Dataset privado `aethel-nextgen-data-v1` ya fue validado en Kaggle como un montaje de 22 shards, con train inglés=19.011, train español=19.012, holdout inglés=989 y holdout español=988. Kaggle los expuso como `.jsonl` descomprimidos; el contrato de hashes y tamaños de contenido plano lo aceptó sin mezclar formatos. El release privado de código V5 pasó su preflight sin red y después ejecutó un intento E0 real en una Tesla T4, con la segunda T4 sin usar y sin DDP.

La ejecución V5 alcanzó los pasos 1–99, pero se abortó antes del checkpoint inicial del paso 192. La causa verificada fue que `AethelNextGen.observe` convertía `memory_state` a CPU antes de llamar `ElLiquido.observe`, mientras `hebbian_trace` estaba en CUDA. Se persistieron `launch_manifest.json`, `package_preflight.json`, `tokenizer.json` y métricas parciales, pero no `latest.pt`, `recovery_receipt.json` ni `checkpoint-*`. No existe un checkpoint recuperable, benchmark, evaluación holdout ni modelo Seed funcional.

El release V6 `e0-v6-liquid-cuda-alignment` corrigió esa separación y fue subido a Kaggle. Su preflight real pasó, pero el smoke CUDA falló en la igualdad entre `torch.device("cuda")` y `torch.device("cuda:0")`. V7 fue subido y ejecutado con una corrección prudente de identidad del buffer (`copy_` en `forward` y `zero_` en `reset_session`), pero su smoke volvió a fallar en la misma igualdad no canónica antes de `forward`. Ninguno de esos fallos demuestra que el buffer permaneciera en CPU; ambos bloquearon correctamente E0, y no crearon checkpoint, métricas de entrenamiento ni modelo.

El release V8 `e0-v8-canonical-cuda-device-check` conserva la actualización in-place de V7 y cambia el smoke para exigir `device.type == "cuda"`, el índice activo de CUDA y el mismo `data_ptr` antes y después de `forward` y `observe`. El bundle V8 fue verificado con SHA-256 `c179817f70256141a7fcc16ca939439495ee2e3fe2be7e5ed95868c0f8c221d6`, sin corpus, pesos, checkpoints, bytecode ni cachés, y se subió como una versión privada del Dataset de código. En Kaggle, el preflight V8 validó los 22 shards sin red y el smoke CUDA devolvió `VERIFIED_LIQUID_CUDA_ALIGNMENT`.

Tras esas puertas, el commit E0 V8 llegó al paso **4.992** en una única CUDA, terminó sin error del lanzador y preservó `latest.pt`, snapshots de pasos 4.608/4.800/4.992, `recovery_receipt.json`, `checkpoint_inspection.json`, `metrics_rank_0.jsonl` y evaluaciones holdout separadas. La inspección textual confirmó 150 tensores y metadatos completos, con contrato de reanudación explícito; los pesos no se descargaron, cargaron ni promovieron. Holdout EN (256 segmentos) registró pérdida 7.771877005696297 y perplejidad 2372.921097307039; holdout ES (256 segmentos) registró pérdida 10.774345595389605 y perplejidad 47779.19608484159. La telemetría final del router siguió `healthy: false`, por lo que la corrida acredita artefactos y una brecha experimental EN/ES, no salud MoE de producción ni capacidad comercial.

La auditoría detallada y reproducible, incluida la diferencia entre parámetros entrenables y los serializados, está en [`training/AETHEL_E0_V8_ARTIFACT_AUDIT_2026-08-23.md`](training/AETHEL_E0_V8_ARTIFACT_AUDIT_2026-08-23.md). Toda mención previa en este documento que diga que no existió Dataset Kaggle, intento GPU, checkpoint o evaluación holdout debe interpretarse como historial anterior, reemplazado por esta actualización operativa.

### Estado D0 posterior a E0 V8

La preparación local D0 se validó contra el manifiesto raíz congelado sin abrir shards, textos holdout, pesos ni checkpoints; no importó PyTorch, no usó CUDA y no hizo red. Después, tras confirmaciones separadas, el bundle de código D0 se añadió como versión privada `(9)` de `aethel-nextgen-source-e0-v1`, se añadió una cuarta celda D0 y se ejecutó sólo esa celda. La salida real fue `D0_AUDIT_READY` con release `d0-v1-e0-v8-static-audit`, 4.992 pasos históricos enlazados, hash de manifiesto `91cfd0e2b14ba6a863143f17ff85629e5f28c88cf13b09627ab8ef34bc78435a`, hash de tokenizer `4a3608e4e45c9117415d1f4fa236aebe20771dc3a3ce85760d9fb9d218fa0815`, `checkpoint_loaded=false`, `raw_corpus_read=false`, `holdout_content_read=false`, `gpu_used=false` y `network_requests=0`. El router final permanece `healthy=false`; D0 no autoriza D1, promoción, serving, carga/reanudación de pesos ni una nueva acción Kaggle. El contrato operativo completo está en [`training/AETHEL_E0_V8_D0_AUDIT_CONTRACT.md`](training/AETHEL_E0_V8_D0_AUDIT_CONTRACT.md).

Se preparó el protocolo **D1A** en [`training/AETHEL_D1_ROUTER_DIAGNOSTIC_PROTOCOL_2026-08-23.md`](training/AETHEL_D1_ROUTER_DIAGNOSTIC_PROTOCOL_2026-08-23.md). Sus componentes locales `validate_aethel_train_only_mount.py`, `summarize_d1a_router_metrics.py` y `run_kaggle_d1a_router_diagnostic.sh` se probaron con fixtures sintéticas. El bundle privado D1A SHA-256 `6391310a5b4aa0e78644c7454a110054fc0f23c7313fb2758afaf863463ee39f` se añadió como versión de código y su release exacto se verificó bajo el montaje `(10)`. La quinta celda preparatoria primero devolvió `D1A_CELL_PREPARED_NOT_EXECUTED`; después, tras confirmaciones finales separadas, D1A se ejecutó desde inicialización nueva y terminó con `D1A_DIAGNOSTIC_COMPLETE`.

El log observó 768 pasos y 1.572.864 tokens, con pérdida mínima/media/máxima 7,648315/9,259973/10,438221. El router tuvo 78 pasos saludables y 690 no saludables; la mínima entropía fue 0,333333 y el máximo desequilibrio 0,187500. La agregación terminó en `D1A_METRICS_SUMMARIZED` y reiteró `checkpoint_loaded=false`, `raw_corpus_read=false`, `holdout_content_read=false`, `network_requests=0` y `promotion_authorized=false`. La salida no es benchmark, candidato ni promoción. La persistencia de checkpoint/salida D1A no se ha verificado independientemente; ningún artefacto será cargado, movido, reanudado o promovido sin autorización nueva. D1B, holdout, serving y Edge/Pro siguen bloqueados.

Después de una confirmación adicional y separada, el usuario ejecutó **Save Version** privado en Kaggle. La interfaz indicó **Version #3 — Successful**, preservando la versión del notebook y su salida D1A. Esta señal no acredita revisión binaria de una salida ni autoriza descargar, cargar, mover, inspeccionar, reanudar o promover un checkpoint; esa frontera permanece intacta y requiere autorización específica nueva.

Se preparó además [`training/AETHEL_D1A_EVIDENCE_REVIEW_PLAN_2026-08-23.md`](training/AETHEL_D1A_EVIDENCE_REVIEW_PLAN_2026-08-23.md), únicamente como plan documental. No ejecutó revisión de salidas, Kaggle, GPU ni artefactos. Si en el futuro se desea consultar metadatos visibles de Version #3, debe pedirse una autorización nueva que limite expresamente el alcance a metadatos y excluya `.pt`, contenido de outputs, corpus, holdout, descargas, movimientos, reanudación, ejecución y promoción.

También se preparó [`training/AETHEL_D1B_ROUTER_BIAS_PROTOCOL_2026-08-23.md`](training/AETHEL_D1B_ROUTER_BIAS_PROTOCOL_2026-08-23.md) como diseño de una futura D1B. La hipótesis es falsable y sólo propone reducir `router_bias_step` de 0,05 a 0,01 en una ventana nueva de 768 pasos, manteniendo datos train, topología, umbrales y restricciones D1A. Tras autorización B1 específica, se añadieron y validaron localmente el lanzador bloqueado D1B, la identidad de resumen D1B y sus contratos. Tras autorización B3, el bundle privado D1B SHA-256 `488990206ad61eaa9098cc68e32b0c9c0bbce197724415401c616ae0102ce0c0` se subió al Dataset de código y la interfaz del usuario confirmó que la versión se creó correctamente; no se observó número de versión ni montaje. No cambió el Dataset de datos, notebook, GPU ni artefactos.

Tras autorización B4 para preparación, se creó y validó localmente `training/AETHEL_D1B_ROUTER_BIAS_BLOCKED_CELL.py` y una copia manual de transferencia. Esta celda sólo busca el release exacto y declara `D1B_CELL_PREPARED_NOT_EXECUTED`; la rama de ejecución está intencionalmente ausente y no puede seleccionar GPU, copiar código, leer Dataset ni entrenar. Tras una confirmación adicional, el usuario la añadió y ejecutó sólo en modo bloqueado. El log seleccionó el release exacto bajo el montaje `(11)`, emitió `D1B_CELL_PREPARED_NOT_EXECUTED` y `D1B_CELL_EXECUTION_BRANCH_INTENTIONALLY_ABSENT`, y confirmó ausencia de GPU, copia de código, lectura de Dataset, pesos y entrenamiento. B5–B6 continúan cerradas; cualquier paso posterior requiere confirmación nueva y específica.

Tras autorización B5a, se creó y validó localmente `training/AETHEL_D1B_ROUTER_BIAS_EXECUTION_CELL.py` y su copia manual. Esta variante inicia con cinco puertas cerradas para ejecución D1B, corrida, GPU, token final y fallback PyTorch experimental. Mientras cualquiera difiera del valor aprobado, devuelve `D1B_EXECUTION_PENDING_FINAL_AUTHORIZATION` antes de resolver el Dataset de datos o tocar el directorio de trabajo. Sus contratos locales, los contratos D1B/D1A, sintaxis Bash, `diff --check`, Vitest y TypeScript pasaron. Tras autorización B5b, el usuario reemplazó y ejecutó sólo la comprobación cerrada bajo el montaje `(11)`; el log devolvió el release D1B exacto y `D1B_EXECUTION_PENDING_FINAL_AUTHORIZATION`, sin GPU, copia de código, lectura de Dataset, pesos ni entrenamiento. Tras las autorizaciones B5c separadas, la interfaz mostró GPU T4 ×2 y la Celda 5 se ejecutó una única vez hasta `D1B_DIAGNOSTIC_COMPLETE`.

Después de B5b, se entregó la versión completa como **CELDA 5**, con las cinco puertas configuradas tras autorización B5c-1 y una convención que exige números y propósito dentro de cada celda futura. La salida final D1B fue `D1B_METRICS_SUMMARIZED` y `D1B_DIAGNOSTIC_COMPLETE`: 768 pasos, 1.572.864 tokens, pérdida mínima/media/máxima 7,667897/9,273529/10,441325 y router con 44 pasos saludables frente a 724 no saludables. La reducción a `router_bias_step=0,01` no superó los 78 pasos saludables de D1A; por tanto queda clasificada `D1B_ROUTER_NOT_IMPROVED`, no como candidato o éxito. El resumen confirmó `checkpoint_loaded=false`, `raw_corpus_read=false`, `holdout_content_read=false`, `network_requests=0` y `promotion_authorized=false`; no se inspeccionó ni manejó output/checkpoint. La síntesis documental final está en [`training/AETHEL_D1A_D1B_ROUTER_EVIDENCE_REVIEW_2026-08-23.md`](training/AETHEL_D1A_D1B_ROUTER_EVIDENCE_REVIEW_2026-08-23.md): no atribuye causalidad ni autoriza una nueva ejecución. D2, D3, holdout, promoción y serving continúan bloqueados hasta un nuevo plan y autorizaciones separadas.

La preparación documental posterior para acelerar sin perder trazabilidad está en [`training/AETHEL_MODEL_READINESS_ACCELERATION_PLAN_2026-08-23.md`](training/AETHEL_MODEL_READINESS_ACCELERATION_PLAN_2026-08-23.md). A1–A3 cerraron contratos locales de métrica, peso y dirección de señal del router. D1C V1 definió una única variación `router_aux_loss_weight: 0.01 → 0.05` contra D1A y fue ejecutada una vez, desde inicialización nueva y con sólo train, tras autorizaciones separadas. El intento se bloqueó al invocar el resumen seguro porque la **Version 13** privada sólo aceptaba `D1A` y `D1B` como identificadores CLI; por ello no emitió `D1C_METRICS_SUMMARIZED` ni `D1C_DIAGNOSTIC_COMPLETE` y no puede clasificarse. No se abrirán ni manipularán sus outputs o checkpoints, ni habrá reanudación o repetición automática. La corrección local añade `D1C` al contrato CLI, pruebas de regresión, el marcador `d1c-v2-summary-cli-fix-train-only` y una **CELDA 8** bloqueada. Su bundle local validado (`a659a615…a97aad07`) excluye corpus, pesos, checkpoints, métricas crudas y bytecode. El usuario confirmó la creación manual de una nueva versión privada con ese ZIP, sin número visual compartido. La CELDA 8 resolvió el release V2 exacto y declaró `D1C_V2_CELL_PREPARED_NOT_EXECUTED`; no hubo GPU, lectura de Dataset, pesos ni retry. V2 no autoriza retry, GPU, Dataset, holdout, checkpoints, D2/D3, promoción ni serving.

Después se preparó el release `d1c-v3-retry-cell-train-only`. Su bundle ZIP de transferencia `aethel-nextgen-source-d1c-v3-retry-cell-train-only.zip` tiene SHA-256 `7028a42ac0246ae1b455e0c7036f5e865b5fe6b9c16331867a3ce40dc0377f06` e incorpora la plantilla de retry con cinco puertas cerradas, la corrección V2 y pruebas estáticas. El usuario aportó una captura que muestra el directorio V3 dentro del Dataset de código privado y confirmó que pegó manualmente una **CELDA 9 V3 bloqueada** en el notebook. La primera comprobación se detuvo de forma segura con `candidatos: ninguno` porque el input V3 aún no estaba actualizado; tras actualizarlo, CELDA 9 resolvió el release V3 exacto y emitió `D1C_V3_CELL_PREPARED_NOT_EXECUTED` y `D1C_V3_RETRY_PENDING_FINAL_AUTHORIZATION`. No fue necesario reemplazar la celda y no hubo GPU, retry, Dataset train/holdout, pesos, outputs, checkpoints, promoción o serving. El protocolo local [`AETHEL_D1C_V3_RETRY_DECISION_PROTOCOL_2026-08-23.md`](training/AETHEL_D1C_V3_RETRY_DECISION_PROTOCOL_2026-08-23.md) y su contrato de autorizaciones con todas las puertas en `false` delimitan un eventual D1C V3-R1 como experimento nuevo.

Durante la preparación local V3-R1 se detectó que el lanzador histórico todavía exigía exclusivamente el release V1, por lo que habría fallado cerrado frente a un retry nuevo. Se añadió un perfil local permitido para el release futuro `d1c-v4-v3-r1-launcher-profile-train-only`, protegido por una autorización independiente de perfil y con V1 como valor predeterminado. También se creó la plantilla local **CELDA 10** `AETHEL_D1C_V3_R1_RETRY_EXECUTION_CELL.py`, con seis puertas cerradas —las cinco de ejecución y una de perfil de release—, inicio nuevo, salida inédita y bloqueo explícito de reanudación. El bundle V4 se construyó y validó sólo en local: TAR SHA-256 `7905caff0c40552b0ae6780f5991827f0106cb34b6dafa1bd51f9508db061c51` y ZIP SHA-256 `08d51374a9684340d7ffe47d48a2f9edf6eb36b0bb123b72ae56bd0f397c043a`, ambos excluyen corpus, JSONL, pesos, checkpoints y bytecode. El usuario aportó una captura que confirma **Version 16 — complete** y el estado **Success** para esta nueva versión privada de código. No se ha adjuntado V4 al notebook, no se ha añadido CELDA 10 y no hubo GPU, Dataset, retry, outputs/checkpoints, holdout, promoción ni serving.

---

## 1.1 Arquitectura canónica de lenguajes

La matriz canónica actual es [`AETHEL_LANGUAGE_AND_RUNTIME_ARCHITECTURE_V1.md`](AETHEL_LANGUAGE_AND_RUNTIME_ARCHITECTURE_V1.md). Resume las fronteras entre **TypeScript/React/Node.js** (producto y transparencia), **Python/PyTorch** (datos, laboratorio y modelo de referencia), **Triton/CUDA** (kernels GPU sujetos a aceptación), **Rust** (memoria y gobierno local), y **Mojo** (inferência local futura por contrato).

| Lenguaje/ruta | Estado al cierre de esta actualización |
|---|---|
| TypeScript/React/Node.js | Implementado en la plataforma web, sin inferencia Aethel propia. |
| Python/PyTorch | Implementado para datos, modelo, evaluación y contratos CPU; V5 alcanzó pasos de entrenamiento reales y V6 ejecutó un smoke CUDA bloqueante. |
| Triton/CUDA | Aceptación experimental V4 en T4 ×2; validación CUDA de producción y rutas Triton estrictas pendientes. |
| Rust | Crate de memoria implementado, compilado en release y con 4 pruebas correctas; sin despliegue 24/7. |
| Mojo | Contrato de inferencia futuro; sin implementación o validación local. |
| C++/CUDA C++ y C# | Sin código en `main`; rutas condicionales, no requisitos activos del núcleo. |

No se debe presentar C++, C#, Mojo o CUDA como un componente operativo sólo porque aparezcan en una idea de arquitectura. La fuente de verdad sobre estas fronteras, evidencia y criterios de promoción es el documento de matriz enlazado arriba.

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

## Historia consolidada del proyecto

Esta sección registra los hitos, intentos y cambios de rumbo que explican el estado actual. Las conversaciones y los logs históricos sirven de contexto operativo; no se consideran evidencia de que exista un modelo propio entrenado.

| Etapa | Qué ocurrió | Decisión vigente que debe conservarse |
|---|---|---|
| **Origen del proyecto** | El usuario pidió localizar el repositorio de IA distinto de `katalog-ai`, entender su documentación y convertirlo en una plataforma y modelo propios. | Aethel debe respetar la arquitectura documentada: Sólido, Líquido, Sueño, curiosidad, MoE y eficiencia; no sustituirla por una demo genérica. |
| **Arquitectura políglota** | Se auditaron los lenguajes mencionados por el usuario y repositorio: TypeScript/Node.js y React/Tailwind para interfaz, Python/PyTorch para laboratorio, Triton/CUDA para rutas GPU, Rust para memoria/gobierno persistente y Mojo como contrato de inferencia local. | No declarar que Rust 24/7, Mojo o Triton completo estén desplegados sólo porque sus contratos/documentos existen. |
| **Primera plataforma web** | Se construyó un dashboard React/tRPC con chat persistente, arquitectura, Trainer, Benchmarks y Engine Status. | La plataforma es real, pero el chat usa el LLM integrado de plataforma y no un checkpoint Aethel. |
| **Corrección de simulaciones** | Tras la instrucción del usuario de no usar simulaciones, se eliminaron curvas/estados engañosos, se dejaron benchmarks sin cifras, se anularon métricas sin proceso y se bloquearon mutaciones tRPC de entrenamiento desde la UI. | Sin artefacto verificable, las métricas deben quedar ausentes; no restaurar etiquetas que sugieran entrenamiento o runtime Aethel propio. |
| **Intentos antiguos de Kaggle** | Se prepararon celdas V3/V5/V9/V10/V11 y un Dataset histórico de bundle de código `aethel-nextgen-source`. Los logs mostraron entradas sin `.gz`, copias múltiples, HTTP 429/502, mínimos ingleses no alcanzados e `IncompleteRead`. | No reconstruir ni descargar corpus durante la corrida. Estas celdas son historial de diagnóstico, no la ruta actual de Seed. |
| **Corpus final local** | Se sustituyó la construcción remota inestable por un paquete local cerrado de 40.000 documentos Wikipedia, con BPE, hashes, holdout y validación offline. | El paquete sigue congelado y local. No se creó ni verificó el Dataset privado final `aethel-nextgen-data-v1`. |
| **Arquitectura cognitiva** | Se implementaron y probaron en CPU La Roca hashable, El Líquido versionado, curiosidad con TTL/procedencia, candidatos LoRA, admisión de replay, preflight, máquina de estados de Sueño y rollback. | Curiosidad no ejecuta acciones externas; Sueño no entrena/promueve sin autorización, evidencia y separación de holdout. |
| **Ruta Triton** | Se añadieron referencias CPU de prefill/SDPA, capacidad y dispatch/combine MoE, un prefill Triton experimental, auditoría de brechas, matriz CUDA y ejecutor de aceptación. | Prefill y dispatch/combina estrictos siguen bloqueados hasta una validación CUDA completa; no presentar referencias CPU como kernels GPU aceptados. |
| **My Browser/Kaggle** | El usuario pidió usar exclusivamente su navegador. Aunque el conector parecía habilitado, la sesión del agente devolvía `Browser: Sandbox`; no se expuso una sesión personal de Kaggle utilizable. | No usar Sandbox como sustituto. Detenerse ante Sandbox, login, CAPTCHA o conexión pendiente. |
| **Estado de cierre y sincronización** | Se amplió este documento, se guardó el checkpoint `dec9ee6a`, se publicó el historial local en el repositorio GitHub privado de IA y se verificó el runtime Rust local. | No se activó GPU, no se creó Dataset final, no se subieron los 22 shards, no se inició entrenamiento ni gasto. |

### Cambios de rumbo que no deben revertirse

La prioridad dejó de ser construir corpus por red dentro de Kaggle y pasó a ser ejecutar **Seed offline** con un Dataset ya validado. La prioridad dejó de ser presentar una interfaz como si fuera un modelo Aethel vivo y pasó a ser declarar de forma visible cuándo sólo hay LLM de plataforma. La prioridad dejó de ser perseguir Edge/Pro inmediatamente y pasó a ser demostrar el primer checkpoint Seed reproducible. Estas decisiones responden a los fallos observados y al requisito central del usuario: datos y resultados reales, no simulaciones.

> Las aprobaciones antiguas de crear celdas, subir bundles de código o explorar Kaggle no sustituyen la autorización actual para crear el Dataset final de datos, seleccionar una GPU, ejecutar Save & Run All o entrenar. Cada una de esas acciones requiere confirmación específica e inmediata.

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
| `manus-webdev://dec9ee6a` | Documento de continuidad ampliado con la historia, Dataset, arquitectura cognitiva, Kaggle/My Browser, restricciones, brechas y rutas de continuación. |
| `manus-webdev://3aa05046` | Corrección global de transparencia: encabezado con `LLM de plataforma conectado · Aethel Seed sin entrenar`, sidebar `modelo propio no iniciado`, métricas nulas y auditoría visual. No se activó GPU, Kaggle, Dataset ni entrenamiento. |
| `manus-webdev://ed495c63` | Inspector local reproducible de código, Dataset, salida, CUDA y Triton que informa `BLOCKED` o `READY_FOR_AUTHORIZATION` sin entrenar. |
| `manus-webdev://028b9859` | Versión anterior del documento de continuidad, ampliada y sustituida por la presente. |
| `manus-webdev://441878ee` | Dashboard transparente: configuración teórica marcada como tal, ausencia explícita de checkpoint/métricas Aethel y bloqueo de entrenamiento desde la interfaz. TypeScript y Vitest pasan. |
| `manus-webdev://b4ccad4e` | Ejecutor CUDA de aceptación Triton que registra bloqueo `NOT_RUN` sin hardware y no habilita contratos. |
| `manus-webdev://9eacabca` | Matriz de aceptación CUDA para paridad, gradientes, memoria, rendimiento, límites y rollback. |
| `manus-webdev://e972b492` | Kernel Triton experimental de prefill causal, separado de la ruta estricta hasta validación CUDA. |
| `manus-webdev://53ba522a` | Referencia CPU de prefill causal verificada contra SDPA. |
| `manus-webdev://d765a95f` | Referencia determinista de capacidad MoE y overflow explícito. |
| `manus-webdev://e2161abf` | Referencia CPU de dispatch/combina MoE y pruebas de equivalencia. |
| `manus-webdev://0cd1a7dc` | Runbook local de GPU para Seed/Edge sin activar hardware. |
| `manus-webdev://18f74cef` | Especificaciones de infraestructura, topología concurrente, producto Workspace, variantes, routing top-2 y ruta comercial. |
| `manus-webdev://9fa4e583` | Lanzador Seed offline, runbook, validación de paquete/tokenizador antes de GPU y pruebas de rechazo. Sin GPU ni Kaggle. |
| `manus-webdev://67e2dcfe` | Evaluación de plataformas gratuitas: Kaggle para piloto Seed, Colab como respaldo, ZeroGPU descartado para entrenamiento prolongado. |
| `manus-webdev://78723e42` | Contratos estrictos que bloquean prefill CUDA y dispatch/combina MoE en ausencia de kernels Triton completos validados. |

Antes de la presente actualización, los cambios estaban consolidados hasta `manus-webdev://dec9ee6a` y sincronizados en GitHub hasta `2a41392`. La compilación local más reciente del crate Rust verificó `cargo check --locked`, `cargo build --locked --release`, 4 pruebas Rust correctas y la plantilla systemd. La evidencia de TypeScript/Vitest y pruebas CPU no sustituye una ejecución CUDA ni un despliegue persistente.

---

## 5. Aplicación web actual

La web es un dashboard oscuro de Aethel y funciona de manera independiente del entrenamiento de pesos propios. Tiene Chat, visualizador de arquitectura, Trainer, Benchmarks y Engine Status. El chat usa un modelo de plataforma con una persona Aethel V3; por lo tanto, no debe confundirse con inferencia de un checkpoint Aethel entrenado.

| Componente | Ruta principal | Estado |
|---|---|---|
| Dashboard principal | `client/src/pages/Home.tsx` | Operativo; visualiza pilares y sólo evidencia disponible. Encabezado: `LLM de plataforma conectado · Aethel Seed sin entrenar`. |
| API tRPC | `server/routers.ts` | Guarda historial, informa `sin checkpoint Aethel` y rechaza iniciar entrenamiento no autorizado desde el dashboard. |
| Layout y chat | `client/src/components/` | Chat y navegación disponibles. |
| Pruebas web | `server/*.test.ts` | Las pruebas de transparencia y guardia impiden etiquetas ambiguas y launches locales desde la UI. |

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
| `training/run_triton_cuda_acceptance.py` | Ejecutor futuro de aceptación CUDA; sin CUDA/Triton emite `NOT_RUN`, sale con código 2 y no habilita contratos. |
| `training/inspect_local_aethel_host.py` | Inspector local sin entrenamiento de fuentes, Dataset, salida, disco, CUDA y Triton. |

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

## 15. Checklist de reinicio y brechas que siguen abiertas

El siguiente chat debe comenzar comprobando el estado actual del repositorio, no suponiendo que los checks históricos representan una ejecución viva. Primero debe leer este archivo y los documentos obligatorios, revisar `git status`, `todo.md` y el último checkpoint. Si se modifica código, debe preservar las pruebas y actualizar los textos de transparencia. Si se solicita una GPU o Kaggle, debe separar con claridad: inspección reversible, preparación de formulario, creación de Dataset, selección de acelerador y comienzo de corrida.

| Brecha pendiente | Estado y condición de avance |
|---|---|
| Dataset Kaggle final | El paquete local existe; `aethel-nextgen-data-v1` no está confirmado como Dataset privado final. Requiere creación manual o My Browser realmente conectado y confirmación inmediata antes de crear. |
| Navegador personal | My Browser no fue expuesto al agente. No navegar en Sandbox bajo la condición del usuario. |
| Host GPU | El sandbox carece de CUDA. Inspeccionar un host autorizado antes de pedir inicio de Seed. |
| Triton de producción | Prefill y dispatch/combina MoE están incompletos para contrato estricto; ejecutar matriz CUDA y guardar informes antes de habilitar. |
| Seed real | No hay checkpoint ni telemetría. Exige Dataset, GPU, autorizaciones, preflight, salida persistente y evaluación bilingüe. |
| Edge/Pro/FSDP | Bloqueados por ausencia de evidencia Seed, corpus de escala superior, validación CUDA y hardware/presupuesto apropiados. |
| Servicio Rust 24/7 | Existe preparación local y contratos; falta host persistente autorizado, supervisión y restauración real de snapshot. |
| Runtime Mojo | Sólo contrato/documentación; no existe runtime instalado ni validado. |
| Corpus comercial | El Dataset actual sirve para Seed; ampliar a matemática, ciencia, ingeniería y programación requiere una nueva versión trazable, no mutar la congelada. |

### Acciones que no deben ejecutarse sin nueva instrucción del usuario

1. No modificar, borrar, publicar ni subir el paquete congelado local.
2. No crear, publicar o alterar un Dataset Kaggle ni pulsar `Save & Run All`.
3. No habilitar acelerador, lanzar entrenamiento, reservar recursos, comprar infraestructura ni ejecutar una corrida continua.
4. No usar Browser Sandbox cuando el usuario pidió My Browser; tampoco pedir al usuario credenciales por chat.
5. No relajar guards `require_triton`, no aceptar un kernel sólo por pruebas CPU y no promover una E0 fallback.
6. No presentar parámetros teóricos, chats de plataforma, referencias CPU o documentación como checkpoint, aprendizaje, benchmark o producto Aethel operativo.
7. No mezclar holdout con tokenizador, entrenamiento, replay, selección de candidato o decisiones de promoción.

---

## 16. Instrucción sugerida para abrir el siguiente chat

Copiar este texto, adjuntar este documento y el agente podrá continuar con el alcance correcto:

> Lee `AETHEL_PROJECT_HANDOFF_2026-08-22.md` completo antes de actuar. Quiero continuar Aethel con datos reales, sin simulaciones y sin presentar métricas inexistentes. El Dataset local está congelado en `/home/ubuntu/aethel-knowledge-corpus-v1-package/`; no actives Kaggle, GPU ni entrenamiento sin mi autorización explícita. Respeta la separación de holdout, La Roca/Líquido/Sueño, los contratos Triton y los gates de promoción. Si solicito usar My Browser, no uses navegador aislado: detente si la herramienta indica `Browser: Sandbox`. Primero verifica el estado actual de `todo.md`, los checkpoints y el Dataset privado en Kaggle si existe.

---

## 17. Referencias internas y externas

Las afirmaciones de estado de este documento se fundamentan en los archivos y checkpoints del repositorio citados anteriormente. Para límites y selección de infraestructura gratuita, consultar las fuentes primarias incluidas en `training/FREE_GPU_PLATFORM_ASSESSMENT_2026-08-21.md`.

[1] [Kaggle — Efficient GPU Usage Tips](https://www.kaggle.com/docs/efficient-gpu-usage)  
[2] [Google Colab — Frequently Asked Questions](https://research.google.com/colaboratory/faq.html)  
[3] [Hugging Face — ZeroGPU](https://huggingface.co/docs/hub/en/spaces-zerogpu)  
[4] [NVIDIA — H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/)

---

## 18. Auditoría técnica externa y precisión de entorno

El 22 de agosto se recibió una auditoría que confirma el veredicto prudente del proyecto: arquitectura y controles preparados, pero sin Seed GPU, benchmark, producto comercial ni aceptación Triton/CUDA demostrados. El contraste posterior documentado en `training/AETHEL_AUDIT_RESPONSE_2026-08-22.md` distingue esa inspección de un checkout que no contenía activos externos del entorno local actual, donde el paquete congelado sí está presente y validado. El paquete continúa fuera de GitHub y no se debe asumir su disponibilidad en Kaggle u otros hosts.

El archivo versionado `engine/artifacts/aethel_real.pt` queda explícitamente clasificado como **histórico no certificado y no promocionable**. No debe cargarse como modelo Aethel ni usarse para justificar resultados. La clasificación estática y sus requisitos de auditoría se encuentran en `engine/artifacts/aethel_real.audit.json`.

La evidencia de pruebas vigente debe comunicarse de manera desagregada: TypeScript correcto, Vitest 4 archivos/9 pruebas, contratos CPU específicos correctos, Rust compilado y probado previamente; `pytest` no se ejecutó en este host y la aceptación CUDA/Triton sigue pendiente. La nueva lista `training/requirements-test.txt` declara el entorno requerido para una futura ejecución reproducible de pytest sin afirmar que ya se haya ejecutado.

---

## 19. Activos que permanecen sólo en la máquina

El inventario completo está en `training/AETHEL_LOCAL_ASSET_INVENTORY_2026-08-22.md`. La distinción crítica es que GitHub contiene código y documentación, mientras el Dataset congelado de 22 shards sigue solamente bajo `/home/ubuntu/aethel-knowledge-corpus-v1-package/`. También existen dos artefactos experimentales CPU externos en `/home/ubuntu/aethel-artifacts/`, ambos de paso 100; son evidencia local acotada, no Seed certificado, no holdout bilingüe ni modelo promocionable.

No se deben borrar ni publicar esos activos por defecto. Antes de moverlos se requiere destino privado, manifiesto, hashes, ACL y verificación de copia. `node_modules`, `runtime/aethel-memory-rust/target` y `.manus-logs` son dependencias, productos de compilación o logs regenerables de esta máquina, no activos de modelo que deban preservarse como fuente de verdad.

## 20. Cierre auditado D1C V3-R1

D1C V3-R1 terminó una única corrida autorizada desde inicialización nueva, con 768 pasos, 1.572.864 tokens, `router_aux_loss_weight=0.05`, train-only y runtime PyTorch/CUDA experimental. El resumidor corregido funcionó y emitió `D1C_DIAGNOSTIC_COMPLETE`. La clasificación es **`D1C_ROUTER_NOT_IMPROVED`**: 67/768 pasos saludables (8,72 %, requisito mínimo 117), entropía mínima 0,3333333433 (requisito estrictamente mayor que 0,333333), desequilibrio máximo 0,1875 (cumple el límite) y pérdida media 9,43690848 (supera el máximo 9,35257273).

El resumen compartido indicó `checkpoint_loaded=false`, `holdout_content_read=false`, `network_requests=0` y ausencia de lectura de corpus crudo. No se abrieron, descargaron, movieron ni deserializaron outputs o checkpoints. D1D no se inicia automáticamente; holdout, promoción, serving y cualquier nuevo experimento requieren una decisión y autorización específicas. Este resultado demuestra una corrida diagnóstica auditable, no un modelo funcional comercial ni un runtime Aethel propio activo.
