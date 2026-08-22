# Auditoría de transparencia del dashboard — 22 de agosto de 2026

## Propósito

La interfaz web no puede hacer que una configuración calculada, una conversación respaldada por un LLM de plataforma o una telemetría inexistente parezcan resultados de un checkpoint Aethel entrenado. Esta auditoría registra los reemplazos aplicados en `client/src/pages/Home.tsx` y el bloqueo correspondiente de servidor.

| Elemento anterior ambiguo | Etiqueta o comportamiento corregido | Evidencia |
|---|---|---|
| “familia piloto calculada tiene X M parámetros” | “configuración piloto calculada” y “parámetros teóricos”; declara ausencia de checkpoint entrenado. | `ChatView` y `server/dashboard.transparency.test.ts`. |
| “Aethel V3 / conversación neuronal” | “Aethel / interfaz de laboratorio”; declara que el chat usa LLM de plataforma y no checkpoint propio. | `ChatView`. |
| “Capacidad calculada” | “Configuración de diseño”; declara ausencia de pesos, benchmarks y telemetría Aethel. | `ChatView`. |
| Entrenador configurable desde el dashboard | Vista informativa “BLOQUEADO POR DISEÑO”; explica Dataset congelado, GPU autorizada y gates Triton. | `TrainerView`. |
| Curva “TRAINING LOSS / REAL PROCESS” | “TRAINING METRICS / EVIDENCIA REAL” y `NOT RUN` hasta que existan artefactos. | `TrainerView`. |
| Mutaciones tRPC que iniciaban procesos locales | `training.start` y `training.nextgenStart` devuelven `PRECONDITION_FAILED`; no hacen `spawn`. | `server/routers.ts` y `server/training.guard.test.ts`. |

La prueba `server/dashboard.transparency.test.ts` verifica literalmente los avisos de configuración teórica, ausencia de checkpoint, ausencia de telemetría propia y la eliminación del botón que afirmaba iniciar Aethel. Esta prueba complementa la guarda de servidor; no sustituye la futura integración de artefactos reales de Seed.
