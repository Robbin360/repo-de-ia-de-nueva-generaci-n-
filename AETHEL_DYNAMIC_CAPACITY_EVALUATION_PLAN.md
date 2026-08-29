# Aethel — Plan de evaluación de capacidad dinámica

**Estado:** diseño de evaluación; no contiene resultados medidos.

## Propósito

Este documento define cómo decidir si una expansión de memoria, adaptador, experto o bloque mejora Aethel sin confundir más parámetros con mayor inteligencia. Cada comparación debe usar el mismo checkpoint base, los mismos datos de evaluación, la misma semilla cuando sea aplicable y un presupuesto explícito de latencia, memoria y FLOPs.

## Métricas principales

| Área | Métrica | Protocolo | Criterio de aceptación inicial |
|---|---|---|---|
| Razonamiento | Exactitud en problemas nuevos y tasa de pasos válidos | Conjunto congelado, respuesta final y verificador externo | Mejora frente al checkpoint base sin regresión EN/ES |
| Matemáticas básicas | Exactitud por operación y dificultad | Sumas, restas, multiplicación, división y problemas textuales; sin contaminar entrenamiento | Mejora estadísticamente significativa o intervalo de confianza no solapado |
| Bilingüismo | Exactitud y pérdida separadas EN/ES, además de transferencia cruzada | Pares paralelos y tareas monolingües nunca vistas | Ningún idioma cae por encima del umbral de regresión fijado antes del experimento |
| Conocimiento | Exactitud con y sin recuperación, frescura y atribución | Preguntas con evidencia fechada y documentos distractores | La respuesta distingue evidencia, incertidumbre y deducción |
| Memoria | Retención, interferencia y olvido | Replay de episodios antiguos después de cada expansión | Regresión dentro del presupuesto; eliminación y corrección respetadas |
| Eficiencia | tokens/s, latencia p50/p95, FLOPs activos, VRAM y energía si existe sensor | Misma longitud, batch, hardware y configuración | Ganancia de calidad por FLOP/byte sin exceder presupuestos |
| Estabilidad | Salud del router, saturación, overflow y varianza entre semillas | Ventanas por idioma y dominio | Sin colapso de expertos ni concentración no explicada |

Los umbrales numéricos no deben inventarse antes de una medición base. Primero se ejecutará el checkpoint sin expansión; después se congelarán límites por área y se compararán las variantes.

## Currículo escalonado

| Etapa | Datos | Capacidad permitida | Puerta |
|---|---|---|---|
| A | Lenguaje EN/ES limpio y deduplicado | Tronco y tokenizer congelados | Pérdida y cobertura bilingüe reproducibles |
| B | Matemáticas básicas verificables | Adaptador pequeño o experto candidato | Exactitud por operación y ausencia de regresión |
| C | Razonamiento compuesto y planificación | Ruta Deliberate/Verify | Mejora contra baseline con FLOPs igualados |
| D | Conocimiento cambiante con fuentes | Memoria externa y recuperación | Evidencia correcta, fecha y rechazo de distractores |
| E | Aprendizaje continuo | Replay, expansión versionada y rollback | Retención después de varias actualizaciones |

El holdout permanece fuera de cada etapa. Los datos nuevos entran primero en `Inbox`, pasan deduplicación y auditoría de procedencia, y sólo luego pueden alimentar `Memory` o `Adaptation`.

## Batería reproducible

La batería mínima contiene subconjuntos separados de EN y ES, traducción y comprensión paralela, matemáticas básicas generadas a partir de plantillas verificables, razonamiento de varios pasos con respuesta comprobable, recuperación con documentos con hash, pruebas de contradicción y medición de latencia/VRAM. Cada ejemplo debe conservar identificador, idioma, fuente, versión, hash y etiqueta de evaluación.

La comparación principal será `V_next` frente a `V_prev` y frente a una variante que recibe el mismo número de tokens pero no obtiene expansión. Para evitar atribuir a los parámetros lo que pertenece al cómputo adicional, se reportarán dos vistas: **presupuesto igualado** y **calidad máxima permitida**.

## Regla de promoción

```text
promote =
    quality_next >= quality_prev - regression_budget
    AND math_next >= math_prev - math_regression_budget
    AND bilingual_next >= bilingual_prev - bilingual_regression_budget
    AND memory_retention >= retention_floor
    AND p95_latency <= latency_budget
    AND active_flops <= flops_budget
    AND provenance_complete
```

Una expansión que sólo reduce la pérdida de entrenamiento, aumenta la memoria o mejora una tarea a costa de olvidar otra se rechaza. El controlador puede conservarla como experimento archivado, pero no puede convertirla en la versión activa automáticamente.

## Qué demostraría cada resultado

Una mejora de memoria demostraría almacenamiento consultable, no aprendizaje paramétrico. Una mejora de un adaptador demostraría especialización limitada. Una mejora de un experto nuevo demostraría que la expansión aporta capacidad bajo ese dominio y presupuesto. Una mejora de razonamiento sólo sería creíble si persiste con problemas nuevos, verificación independiente y comparación de FLOPs. Ningún resultado de esta batería equivale por sí solo a inteligencia general, conciencia o razonamiento humano.

## Estado

Este plan es una especificación de pruebas. Aethel todavía necesita ejecutar la batería con un checkpoint y datos autorizados antes de declarar cualquier ganancia.

## Referencias

[1]: https://arxiv.org/abs/1708.01547 "Lifelong Learning with Dynamically Expandable Networks"
[2]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
[3]: https://aclanthology.org/2024.acl-long.681/ "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding"
[4]: https://arxiv.org/abs/1910.02054 "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models"

---

**Nota:** Las referencias se conservaron como enlaces de trabajo; no se han ejecutado benchmarks en este documento.
