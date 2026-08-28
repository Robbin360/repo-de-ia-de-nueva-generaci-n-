# Plan de revisión futura de evidencia D1A — sin ejecución

**Fecha:** 23 de agosto de 2026
**Estado:** plan documental aprobado; **ninguna revisión de salida ha comenzado**.
**Alcance:** definir una futura revisión limitada de evidencia D1A sin abrir salidas, checkpoints, corpus, holdout ni Kaggle desde este trabajo.

## 1. Punto de partida verificable

D1A fue una ventana diagnóstica *train-only* desde inicialización nueva, no un candidato de modelo. El resumen seguro registra `D1A_DIAGNOSTIC_COMPLETE`, 768 pasos, 1.572.864 tokens, 78 pasos de router saludables y 690 no saludables. También declara que no se cargó un checkpoint de entrada, no se leyó corpus crudo ni holdout, no hubo red y no se autorizó promoción. [1]

La interfaz de Kaggle aportada por el usuario mostró **Version #3 — Successful** después de la confirmación explícita para guardar la versión. Esto acredita la creación exitosa de una versión privada del notebook; por sí solo **no acredita** qué archivos de salida contiene, la integridad binaria de un checkpoint ni la posibilidad de reanudarlo. [2]

> **Regla de interpretación:** D1A sigue siendo evidencia diagnóstica de inestabilidad del router. No es un benchmark, un modelo saludable, una base para serving, Edge o Pro, ni permiso para iniciar D1B.

## 2. Qué hace este plan y qué no hace

Este documento sólo establece fronteras operativas. No navega Kaggle, no solicita GPU, no modifica Dataset/código, no lista directorios de salida, no descarga archivos y no lee metadatos nuevos. La única evidencia tratada aquí es el resumen ya versionado localmente y la señal visual compartida por el usuario sobre el éxito de Version #3. [1] [2]

| Actividad | Estado bajo este plan | Motivo |
|---|---|---|
| Revisar el resumen local `d1a_v1_router_diagnostic_evidence.json` ya versionado | Permitido como contexto documental | No contiene pesos, corpus, holdout ni JSONL crudo. [1] |
| Citar la señal visual **Version #3 — Successful** | Permitido como confirmación de creación de versión | No prueba contenido o integridad de salidas. [2] |
| Entrar en Kaggle o abrir la página de outputs | No ejecutado y requiere nueva confirmación inmediata | Es una acción externa y puede exponer metadatos de artefactos. |
| Enumerar nombres, tamaños, fechas o hashes de outputs D1A | Prohibido hasta autorización específica de **revisión de metadatos** | Aunque no abra pesos, sigue accediendo a artefactos de salida. |
| Abrir `router_diagnostic.json` generado, logs nuevos o cualquier JSON de salida D1A | Prohibido hasta autorización específica | Sería una revisión de salida no realizada aún. |
| Abrir, deserializar, descargar, mover, copiar, cargar o reanudar `.pt` | Prohibido | Ninguna necesidad diagnóstica actual justifica tocar pesos. |
| Abrir corpus, shards train, contenido holdout o tokenizador montado | Prohibido | El objetivo no es reevaluar datos ni romper la separación del holdout. |
| Ejecutar D1A, D1B, D2, D3, `Save & Run All`, GPU, promoción o serving | Prohibido | Requiere un plan y confirmaciones independientes. [2] |

## 3. Secuencia permitida para una futura revisión de metadatos

Una futura revisión sólo podría empezar tras una confirmación nueva, inmediata y limitada. Debe centrarse en **metadatos de persistencia**, no en contenidos: por ejemplo, que exista una versión identificada como #3, la fecha/estado visible y, si el usuario lo permite de forma inequívoca, una lista de nombres y tamaños sin abrir archivos. La revisión se detendrá antes de cualquier nombre con extensión `.pt`, antes de abrir JSON de salida y antes de navegar a una acción de descarga, copia, ejecución o modificación.

| Puerta | Evidencia mínima aceptable | Decisión permitida | No permitido |
|---|---|---|---|
| M1 — confirmación de alcance | Autorización textual nueva que nombre **Version #3** y “sólo metadatos” | Iniciar una inspección visual limitada si se dispone del acceso autorizado | Inferir consentimiento de autorizaciones D1A anteriores. |
| M2 — revisión de metadatos | Estado/version y, sólo si fue autorizado, nombre/tamaño/fecha visibles | Registrar qué quedó verificable y qué no | Abrir archivos o navegar a descargas. |
| M3 — informe de evidencia | Resumen que separe hechos observados, no observados y bloqueos | Conservar la trazabilidad documental | Declarar checkpoint válido, recuperable o promocionable. |
| M4 — decisión posterior | Revisión separada de la evidencia de telemetría ya permitida | Proponer, no ejecutar, una discusión de diseño D1B | Seleccionar por holdout, cambiar código o iniciar GPU. |

## 4. Texto de autorización requerido antes de M1

Antes de cualquier acceso posterior a Kaggle o a la salida, la autorización debe ser explícita y vigente en ese momento. El formato mínimo recomendado es el siguiente:

> **“Autorizo revisar únicamente metadatos visibles de la salida D1A de Kaggle Version #3. No autorizo abrir, descargar, cargar, mover, copiar, inspeccionar contenido ni reanudar archivos; en particular no `.pt`, corpus, holdout, JSONL crudo o tokenizador. No autorizo ejecutar celdas, usar GPU, modificar Dataset, guardar otra versión, iniciar D1B/D2/D3, promoción ni serving.”**

Si la autorización omite el límite de metadatos, menciona descarga/carga, o la interfaz muestra una pantalla ambigua, la revisión se detiene y se pide aclaración. El usuario puede también aportar una captura estática; esa alternativa permite evaluar lo visible sin operar sobre su cuenta, pero tampoco acredita contenidos no mostrados.

## 5. Ruta antes del siguiente entrenamiento

No existe una fecha automática para un nuevo entrenamiento. El orden seguro es condicional y no equivale a autorización para avanzar:

| Etapa futura | Condición previa | Resultado permitido |
|---|---|---|
| Revisión M1–M3 de evidencia D1A | Nueva autorización de metadatos, si el usuario desea realizarla | Informe de preservación limitado, sin tocar contenido. |
| Discusión de diseño D1B | Evidencia documental suficiente y aprobación para planificar | Propuesta única, *train-only*, sin ejecutar. |
| Preparación D1B | Plan aprobado, cambio de código revisado, release privado nuevo y fuente verificada | Celda bloqueada y contratos locales; sin GPU. |
| Ejecución D1B | Confirmaciones nuevas y separadas para celda, GPU y entrenamiento | Diagnóstico desde cero con selección basada sólo en train. |
| D2 y D3 | Resultados D1B auditados y planes independientes aprobados | Análisis de metadatos de datos y, mucho después, posible corrida candidata. |
| Evaluación final holdout | Candidato fijado sin usar holdout para elegirlo y autorización específica | Una evaluación aislada; nunca selección iterativa. |

La primera corrida que pudiera llamarse **candidata** no es D1A ni D1B: correspondería a una futura D3, sólo tras completar las puertas anteriores. Incluso entonces, seguiría siendo experimental hasta demostrar de forma separada calidad, reproducibilidad, seguridad, coste y operación. [2] [3]

## 6. Criterios de cierre de esta etapa documental

Esta etapa queda cerrada cuando el plan está registrado, el tracker refleja que aún no se revisaron artefactos y no se ha efectuado ninguna acción externa. El siguiente paso sólo existe si el usuario emite la autorización M1 exacta o solicita de manera separada un plan D1B; ambos casos requerirán un nuevo alcance y confirmaciones inmediatas.

## Referencias

[1]: ./d1a_v1_router_diagnostic_evidence.json "Resumen seguro de evidencia D1A"
[2]: ./AETHEL_D1_ROUTER_DIAGNOSTIC_PROTOCOL_2026-08-23.md "Protocolo y límites D1A"
[3]: ./AETHEL_E0_V8_REMEDIATION_PLAN_2026-08-23.md "Puertas D1–D3"
