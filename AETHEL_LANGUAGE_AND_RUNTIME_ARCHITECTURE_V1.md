# Aethel — Matriz de lenguajes y runtimes

**Estado documental:** 22 de agosto de 2026  
**Ámbito:** repositorio `Robbin360/repo-de-ia-de-nueva-generaci-n-`, rama `main`  
**Propósito:** fijar qué lenguaje resuelve cada responsabilidad, qué ya existe en el repositorio y qué sigue siendo una ruta de implementación. Este documento evita presentar contratos, prototipos CPU o planes de producto como servicios desplegados.

> **Principio rector:** cada lenguaje debe tener una frontera de responsabilidad clara. Aethel no adopta lenguajes por prestigio; los introduce únicamente cuando mejoran verificabilidad, seguridad, rendimiento o mantenibilidad de una ruta concreta.

---

## 1. Vista de conjunto

```text
Usuario / equipo
  └─ TypeScript + React + Tailwind ── interfaz, controles y transparencia
       └─ TypeScript + Node.js + tRPC ── API, sesiones y reglas de UI

Laboratorio y preparación verificable
  └─ Python + PyTorch ── Dataset, tokenizer, Transformer, evaluación, Sueño
       └─ Triton + CUDA ── kernels GPU de producción tras aceptación CUDA

Memoria y gobierno persistente
  └─ Rust ── memoria citable, JSONL, snapshots, Unix socket, supervisión

Inferencia local futura
  └─ Mojo ── carga de artefactos, prefill, decode y KV-cache validados

Interoperabilidad futura y sólo si existe una necesidad demostrada
  ├─ C++ / CUDA C++ ── extensiones de bajo nivel o FFI no cubiertas por Triton
  └─ C# ── clientes o conectores empresariales .NET, nunca el núcleo de entrenamiento
```

La arquitectura actual tiene **tres implementaciones con código versionado**: TypeScript/Node.js para la plataforma, Python/PyTorch/Triton para laboratorio y motor, y Rust para el núcleo local de memoria. Mojo tiene un contrato escrito, pero no una implementación validada. No existe código C++, CUDA C++ ni C# en la rama `main` a la fecha de este documento.

---

## 2. Matriz canónica por lenguaje

| Lenguaje o tecnología | Responsabilidad canónica | Evidencia en el repositorio | Estado verificable | No debe afirmarse |
|---|---|---|---|---|
| **TypeScript** | Interfaz web, tRPC, autenticación, historial, estados transparentes de entrenamiento y visualización. | `client/src/`, `server/routers.ts`, pruebas Vitest en `server/*.test.ts`. | **Implementado.** La interfaz declara que usa LLM de plataforma y que Seed no está entrenado. | Que el frontend ejecuta inferencia de un checkpoint Aethel propio. |
| **React 19 + Tailwind 4** | Componentes del dashboard, navegación, chat, arquitectura, Trainer, Benchmarks y Engine Status. | `client/src/pages/Home.tsx`, componentes y `client/src/index.css`. | **Implementado.** | Que las pantallas de métricas contienen resultados de GPU sin artefactos reales. |
| **Node.js + Express + tRPC** | Backend de la plataforma, contratos de procedimientos, sesión, persistencia web y guardas de acciones. | `server/routers.ts`, `server/_core/`, `server/training.guard.test.ts`. | **Implementado.** Las mutaciones de entrenamiento por dashboard se rechazan deliberadamente. | Que Node.js es el entrenador o runtime del modelo Aethel. |
| **SQL / Drizzle / MySQL** | Metadatos de la aplicación, usuarios e historial persistente de UI. | `drizzle/`, `server/db.ts`, esquema y migraciones. | **Integrado con la plataforma web.** | Que almacena pesos, shards de Dataset o bytes de checkpoints. |
| **Python 3 + PyTorch** | Preparación y validación de corpus, BPE, Transformer, RoPE, GQA, MoE, entrenamiento, evaluación, exportación y pruebas de los contratos cognitivos. | `engine/aethel_model.py`, `engine/aethel_nextgen.py`, `training/*.py`. | **Implementado y probado principalmente en CPU.** La receta GPU está preparada, no ejecutada con autorización. | Que haya entrenamiento GPU, checkpoints propios o benchmark Aethel real. |
| **Triton** | Kernels GPU especializados de SwiGLU, decode causal, router top-2 y prefill experimental; objetivo industrial para rutas CUDA de producción. | `engine/triton_bridge.py`, `training/run_triton_cuda_acceptance.py`, matriz y auditoría Triton. | **Implementación parcial y referencias CPU probadas.** La aceptación CUDA está pendiente y el ejecutor emite `NOT_RUN` sin CUDA/Triton. | Que prefill, router y dispatch/combina MoE ya estén aceptados para producción. |
| **CUDA** | Plataforma de ejecución objetivo para el entrenamiento Seed/Edge y para aceptar kernels Triton. | Guardas de preflight y matriz de aceptación; no hay GPU disponible en el entorno actual. | **Dependencia de hardware pendiente.** | Que CUDA haya sido usada para entrenar, perfilar o validar rendimiento de Aethel. |
| **Rust** | Memoria/gobierno local: JSONL, snapshots, consolidación, recuperación trazable, Unix socket, plantilla systemd. | `runtime/aethel-memory-rust/`. | **Implementado y compilado localmente.** `cargo check`, `cargo build --release` y 4 pruebas pasaron; no está desplegado 24/7. | Que haya un servicio persistente, multi-tenant o productivo activo. |
| **Mojo** | Runtime local futuro de inferencia token a token y KV-cache, con paridad contra artefactos PyTorch. | `runtime/mojo/INFERENCE_CONTRACT.md`. | **Contrato documentado, no implementado ni validado en este entorno.** | Que Mojo ya genere tokens, cargue pesos o sustituya a PyTorch. |
| **Bash** | Lanzadores y gates reproducibles para Seed offline, preflight, autorización y reanudación. | `training/run_kaggle_seed_offline.sh`, scripts de empaquetado y validación. | **Implementado como orquestación.** | Que un script autorizado equivalga a una corrida iniciada. |
| **JSON, JSONL y Markdown** | Contratos, manifests, ledger de procedencia, snapshots, reportes y documentación auditable. | Dataset package, runtime Rust, `training/*.md`. | **Formatos de intercambio, no runtimes.** | Que un manifiesto sustituya la validación de una corrida real. |
| **C++ / CUDA C++** | Posible FFI o extensión de muy bajo nivel sólo si una medición demuestra que Triton, Rust o Mojo no cubren una ruta crítica. | No hay archivos `.cpp`, `.cc`, `.cxx`, `.cu` ni extensión compilada en `main`. | **No seleccionado para el núcleo actual; sin implementación.** | Que exista kernel C++ propio, FlashAttention C++ o un runtime C++ de Aethel. |
| **C# / .NET** | Posibles SDKs, conectores de identidad, herramientas Windows o integración empresarial futura. | No hay archivos `.cs` ni proyecto .NET en `main`. | **No implementado; fuera del camino crítico actual.** | Que C# participe actualmente en entrenamiento, inferencia o memoria. |

---

## 3. Decisiones de implementación

### 3.1 Plataforma y producto: TypeScript de extremo a extremo

TypeScript es el lenguaje de la plataforma comercial porque permite que el dashboard React, la API tRPC y los contratos de interfaz compartan tipos. Node.js/Express atiende las rutas de producto, pero **no** ejecuta entrenamiento, no conserva checkpoints de modelo y no debe simular telemetría que no provenga de una corrida.

El chat web usa el LLM de plataforma disponible en la infraestructura de la aplicación. Esa integración sirve para explicar arquitectura y conservar conversación; no es una vía oculta de inferencia de pesos Aethel.

### 3.2 Laboratorio, datos y modelo: Python + PyTorch

Python/PyTorch es el entorno temporal y reproducible de investigación: construye y valida el Dataset, entrena el tokenizador BPE sólo con `train`, expresa el Transformer de referencia y ejecuta evaluación sobre holdouts separados en inglés y español. También contiene los contratos de La Roca, El Líquido, curiosidad y Sueño.

PyTorch es la referencia semántica, no la afirmación de rendimiento final. Sus fallbacks permiten comprobación CPU y una línea base E0 de laboratorio bajo autorización separada; una ruta PyTorch fallback no promociona un checkpoint a La Roca comercial.

### 3.3 Ruta GPU: Triton primero, CUDA como requisito de aceptación

Triton es la capa de kernel elegida para las rutas GPU críticas de Aethel: SwiGLU, atención causal, router top-2, capacidad y posteriormente dispatch/combina MoE. Se mantiene junto a Python para facilitar paridad contra las referencias PyTorch y pruebas reproducibles.

La condición para cambiar una ruta a modo estricto no es que exista un archivo Triton. Requiere evidencia CUDA de entorno, tolerancia numérica, causalidad, gradientes o etiqueta `inference-only`, límites, memoria, rendimiento, overflow y rollback. Mientras no exista esa evidencia, los guardas deben continuar bloqueando promoción o declaración de producción.

### 3.4 Servicio persistente: Rust para memoria y gobierno

Rust contiene el núcleo que procesa JSONL, snapshots y recuperación trazable mediante Unix socket. La elección apunta a una operación robusta de larga duración con una frontera explícita entre memoria gobernada y el modelo. El crate actual compiló y sus cuatro pruebas cubren persistencia/restauración/consolidación, recuperación citable y healthcheck local.

El siguiente paso de Rust no es añadir funciones hipotéticas: es desplegarlo en un host autorizado con directorios persistentes, permisos mínimos, healthcheck, observabilidad y restauración probada. Hasta entonces, “Rust 24/7” es un objetivo, no un hecho.

### 3.5 Inferencia local: Mojo por contrato antes de adopción

Mojo es una ruta objetivo para que artefactos Aethel puedan cargarse y ejecutarse localmente con prefill, decode y KV-cache. El contrato exige hashes de artefacto, paridad de logits/tokens con PyTorch, control explícito de memoria y métricas medidas en hardware objetivo.

No se instalará ni declarará un runtime Mojo por cumplir solamente un documento. Debe implementarse, probarse contra un checkpoint Aethel real y ejecutarse sobre hardware concreto antes de integrarlo al producto.

### 3.6 C++ y C#: extensiones condicionadas, no dependencias ficticias

C++/CUDA C++ podría entrar sólo en dos casos: una extensión de bajo nivel que Triton no pueda expresar o una integración FFI con una biblioteca necesaria y evaluada. C# podría entrar si un cliente empresarial .NET, un agente Windows o una integración de identidad lo exigen. Ninguno debe anticiparse con código vacío o aparecer como requisito del núcleo mientras el equipo no haya medido la necesidad.

---

## 4. Flujo de artefactos y límites entre lenguajes

| Productor | Artefacto o interfaz | Consumidor | Regla de seguridad |
|---|---|---|---|
| Python | Paquete Dataset, BPE, manifests y checkpoints atómicos | Entrenador PyTorch, evaluador, Mojo futuro | Hashes y separación de holdout obligatorios. |
| Python/PyTorch | Referencia numérica de atención y MoE | Triton | Paridad antes de activar rutas estrictas. |
| Triton/CUDA | Reporte de aceptación GPU | Launcher y gates de entrenamiento | `NOT_RUN` o fallo bloquea promoción. |
| Rust | JSONL, snapshots, respuestas de Unix socket | Orquestación local y futura plataforma | No abre puertos de red ni recibe secretos/corpus sin política. |
| TypeScript/Node | Solicitudes de producto y estado verificable | UI React | No inicia entrenamiento por la interfaz. |
| Mojo futuro | Logits, tokens y KV-cache | Cliente local/TypeScript mediante integración posterior | Paridad con checkpoint y políticas locales antes de exponerlo. |
| C++/C# futuro | Sólo interfaces aprobadas por RFC y benchmark | Componente concreto que las justifique | Sin introducir rutas alternativas no auditadas. |

---

## 5. Estado exacto de implementación al actualizar este documento

| Componente | Estado |
|---|---|
| Dashboard React/tRPC transparente | Implementado y versionado. |
| Dataset bilingüe de 40.000 documentos y 22 shards | Congelado y validado localmente; no publicado como Dataset final de Kaggle. |
| Seed offline, gates, checkpointing y evaluación bilingüe | Preparados; no se inició corrida GPU. |
| Modelo Transformer, arquitectura cognitiva y referencias MoE/prefill | Implementados y probados en CPU según las pruebas de cada contrato. |
| Triton en rutas de producción | Bloqueado hasta la matriz de aceptación CUDA real. |
| Runtime Rust | Compila en release y 4 pruebas pasan; no desplegado persistente. |
| Runtime Mojo | Contrato escrito; no implementación validada. |
| C++/CUDA C++ y C# | Sin código ni runtime implementado. |
| Checkpoint Aethel entrenado, benchmark o generación propia | No existe todavía. |

---

## 6. Política de evolución

Antes de añadir o promover cualquier lenguaje/ruta nueva, se debe documentar: el problema medido, la interfaz de entrada/salida, la prueba de equivalencia con la referencia, el coste de operación, la estrategia de rollback y la propiedad de datos. El cambio debe conservar los principios de Dataset trazable, holdout protegido, autorización explícita y ausencia de métricas simuladas.

La secuencia de prioridad permanece: **Seed real y reproducible → aceptación CUDA de Triton → inferencia propia verificada → servicio Rust persistente → runtime Mojo → extensiones C++/C# sólo si se justifican**.

