# Diseño propuesto del corpus Aethel Edge V1

> **Estado:** propuesta local. No autoriza descargas, cambios de dataset, uso de red ni entrenamiento.

## Objetivo

Construir un corpus trazable para un primer modelo Edge bilingüe. La mezcla propuesta prioriza texto general en inglés y español, una fracción adicional de español filtrado y una fracción menor de razonamiento matemático en inglés. No se declara que esta composición otorgue razonamiento, conocimiento amplio o matemáticas fiables: esas capacidades exigen evaluación independiente.

| Componente | Rol | Límite inicial de documentos | Idioma | Estado |
|---|---:|---:|---|---|
| FineWeb2 EN | Texto general | 100.000 | EN | Deshabilitado, pendiente de aprobación |
| FineWeb2 ES | Texto general | 100.000 | ES | Deshabilitado, pendiente de aprobación |
| HPLT2 ES | Refuerzo de español | 30.000 | ES | Deshabilitado, pendiente de aprobación |
| OpenR1-Math | Matemática con problema, solución y respuesta verificables | 20.000 | EN | Deshabilitado, pendiente de aprobación |

Los límites son un punto de partida y **no equivalen a una meta de tokens**. Antes de entrenar se medirá el conteo real de tokens con el tokenizador Edge. Si el corpus no cubre con diversidad suficiente el presupuesto de la fase, se ampliará mediante nuevas fuentes documentadas o un muestreo controlado, nunca mezclando el holdout.

## Controles

El preparador aplica normalización Unicode, eliminación básica de PII, deduplicación exacta entre fuentes, límites de longitud, hashes por registro, procedencia por ejemplo y una partición estable de validación del 0,5 %. Los ejemplos OpenR1 sólo forman texto con `problem`, `solution` y `answer`; exigen las señales `is_reasoning_complete=true` y `correctness_math_verify=true` y excluyen las generaciones auxiliares.

MGSM, GSM8K y Belebele continúan reservados para evaluación. No se usarán sus filas en el corpus ni en el entrenamiento del tokenizador.

## Aprobaciones necesarias

La futura construcción exige dos decisiones distintas: aprobar las fuentes concretas y autorizar una ejecución de preparación con red. La preparación debe producir un manifiesto resuelto, hashes de shards, conteo EN/ES y tokenizador antes de proponer un cambio del dataset `aethel-nextgen-data-v1`.
