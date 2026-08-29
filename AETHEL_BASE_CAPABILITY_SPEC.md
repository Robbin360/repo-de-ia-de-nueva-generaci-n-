# Aethel Base Capability Specification

**Estado:** propuesta experimental para revisión y entrenamiento controlado  
**Autor:** Manus AI  
**Fecha:** 2026-08-28  
**Relación:** complementa `AETHEL_DYNAMIC_CAPACITY_SPEC.md`; no declara capacidades ya demostradas.

> **Principio de diseño.** Aethel no debe depender de módulos dinámicos para compensar una base débil. El núcleo inicial debe mostrar competencia medible en conversación bilingüe EN/ES, seguimiento de instrucciones, razonamiento elemental y matemáticas básicas antes de recibir nuevas habilidades o parámetros.

## 1. Alcance y aclaración de tamaño

La propuesta inicial discutida es un núcleo de aproximadamente **100 millones de parámetros (100M)**. En español, 100B significa 100.000 millones; no debe confundirse con este objetivo de 100M.

Un modelo de 100M puede ser una semilla eficiente y competente en un dominio bien definido, pero no debe describirse como razonador general fuerte sin pruebas. La evidencia de que modelos pequeños pueden generar lenguaje coherente en un dominio controlado no implica competencia general, bilingüismo amplio ni razonamiento robusto [1]. Por ello, Aethel debe fijar el dominio, la mezcla de datos y los umbrales antes del entrenamiento.

## 2. Qué significa una base competente

“Hablar fluido” no significa únicamente producir frases gramaticales. La aceptación mínima requiere que el modelo mantenga contexto, siga instrucciones, cambie de idioma cuando se le pide, no mezcle EN/ES sin razón y responda de forma coherente en ejemplos no vistos.

“Razonar” no significa imprimir una cadena de pensamiento privada ni usar palabras que parezcan lógicas. La evaluación debe comprobar que el modelo puede identificar premisas, aplicar reglas, detectar contradicciones, resolver problemas cortos y corregir errores cuando recibe evidencia. Las respuestas se puntúan por resultado verificable y explicación breve, sin exigir la exposición de razonamiento interno.

| Capacidad base | Evidencia mínima requerida | Prueba recomendada |
|---|---|---|
| Conversación EN | Diálogo coherente y seguimiento de instrucciones no vistas | Conjunto de conversaciones aislado por familias |
| Conversación ES | Igual criterio, con registro y concordancia adecuados | Conjunto equivalente en español |
| Transferencia EN↔ES | Responder en el idioma solicitado sin degradación abrupta | Pares paralelos y cambios de idioma |
| Razonamiento | Resolver tareas nuevas, no repetir plantillas | Problemas con plantillas retenidas fuera del entrenamiento |
| Matemáticas básicas | Resultado correcto y procedimiento comprobable | Aritmética, fracciones, porcentajes, potencias y logaritmos simples |
| Seguridad epistémica | Reconocer incertidumbre y no inventar datos | Preguntas sin respuesta y contradicciones controladas |
| Retención | Mantener habilidades después de aprender una tarea nueva | Replay y regresión contra el checkpoint anterior |

Estos umbrales son un contrato de aceptación que debe calibrarse con un conjunto piloto. No son resultados actuales de Aethel.

## 3. Arquitectura base y capacidad ampliable

El núcleo de 100M debe permanecer pequeño, estable y entrenable en hardware limitado. Se propone un Transformer causal con RoPE, GQA y un MoE moderado únicamente si una comparación densa demuestra que el coste de routing no cancela la ganancia. El router debe tener telemetría de cobertura, concentración, overflow y variación temporal.

La base se divide conceptualmente en tres superficies:

| Superficie | Contenido | Política de cambio |
|---|---|---|
| Núcleo lingüístico | Embeddings, bloques compartidos, normalización y salida | Cambios lentos, sólo mediante releases evaluados |
| Habilidades ampliables | Adaptadores, expertos o pequeños bloques | Se entrenan de forma aislada y versionada |
| Memoria y herramientas | Documentos, episodios, calculadora y verificadores | Puede crecer sin reentrenar el núcleo |

La capacidad dinámica no debe escribir directamente en los pesos durante una conversación. Una interacción puede producir un episodio provisional o una propuesta de ejemplo. La expansión se realiza después, en un lote controlado, cuando existe un patrón persistente de fallos y el módulo nuevo tiene datos, holdout, replay y presupuesto.

## 4. Currículo de entrenamiento inicial

El entrenamiento inicial debe priorizar calidad y transferencia, no sólo cantidad de tokens. Primero se aprende la forma general del idioma con datos EN/ES limpios y deduplicados. Después se introducen instrucciones, diálogo, traducción y tareas de razonamiento. Las matemáticas deben incluir ejemplos generados o recopilados con trazas verificadas, pero el holdout debe contener variaciones no presentes literalmente en el entrenamiento.

| Etapa | Objetivo | Composición orientativa | Puerta de salida |
|---|---|---|---|
| A. Lenguaje | Fluidez y modelado causal | Texto EN/ES balanceado y deduplicado | Pérdida estable por idioma y muestras legibles |
| B. Interacción | Instrucciones y conversación | Diálogo, preguntas, resumen y traducción | Seguimiento de instrucciones sin mezcla lingüística indebida |
| C. Procedimientos | Razonamiento corto | Clasificación, composición, contradicciones y planificación | Generalización a plantillas no vistas |
| D. Matemáticas | Cálculo y verificación | Aritmética, álgebra elemental y logaritmos simples | Exactitud comprobada por un evaluador externo |
| E. Consolidación | Evitar olvido | Replay de A–D y ejemplos difíciles | Sin regresión significativa en EN, ES o matemáticas |

El corpus de validación debe permanecer fuera del entrenamiento y dividirse por familia de problema, no sólo por líneas aleatorias. De lo contrario, el modelo puede memorizar la plantilla y parecer más capaz de lo que es.

## 5. Aprender una habilidad nueva: ejemplo de logaritmos

Si Aethel encuentra repetidos errores con logaritmos, el sistema no debe añadir parámetros inmediatamente. El flujo recomendado es:

```text
error detectado → agrupar casos → verificar que el fallo persiste
→ recuperar ejemplos y soluciones comprobadas → entrenar adaptador candidato
→ probar generalización y regresión → promover o descartar
```

Un adaptador `math-log-v1` podría aprender definiciones, propiedades y transformaciones frecuentes. Un verificador aritmético podría comprobar `log₂(8)=3` y detectar errores de signos o dominio. La memoria externa podría conservar definiciones y ejemplos con procedencia. El núcleo sólo debería absorber el procedimiento si la habilidad se usa con suficiente frecuencia y el beneficio supera el coste de integrar nuevos pesos.

La promoción mínima exige que el candidato mejore el conjunto nuevo, mantenga las pruebas EN/ES y no empeore tareas matemáticas anteriores. Una reducción de pérdida de entrenamiento por sí sola no es suficiente.

## 6. Criterios de aceptación de la base

Antes de activar el crecimiento dinámico, debe existir un recibo de evaluación que incluya hash del checkpoint, tokenizador, manifiesto de datos, configuración, semilla y versión del evaluador. La batería debe medir pérdida y perplejidad por idioma, pero también calidad funcional.

| Grupo | Métricas | Condición de decisión |
|---|---|---|
| Lenguaje | Exactitud de instrucción, coherencia, continuidad de contexto | Comparar con baseline fijo y revisar errores por idioma |
| Bilingüismo | Exactitud EN, ES y transferencia | No usar una media que oculte una degradación de un idioma |
| Razonamiento | Exactitud en problemas nuevos, contradicciones y planificación | Separar memorización de generalización |
| Matemáticas | Exactitud con verificador independiente | Registrar errores por operación, no sólo promedio |
| Eficiencia | Tokens/s, latencia p50/p95, memoria y energía si está disponible | Comparar mismo hardware, batch y longitud |
| Estabilidad | Overflow, concentración del router, NaN, pérdida y regresión | Rechazar candidatos inestables |
| Aprendizaje | Mejora nueva menos regresión en replay | Promover sólo con recibo completo |

Los umbrales numéricos deben congelarse antes de comparar variantes. No se debe mover la meta después de observar los resultados.

## 7. Recomendación de escala

La base de 100M es razonable como **Edge competente inicial**, pero no como garantía de razonamiento general fuerte. Si no supera la batería de conversación EN/ES y razonamiento elemental, aumentar parámetros o añadir módulos sólo ocultará el problema. Si sí la supera, se puede añadir la primera habilidad versionada y medir transferencia.

La ruta recomendada es:

```text
100M base competente
→ adaptadores pequeños para habilidades persistentes
→ memoria externa con procedencia
→ expertos adicionales sólo si el adaptador no basta
→ expansión del tronco sólo después de ablaciones y regresión
```

El objetivo correcto es maximizar **calidad por parámetro, por FLOP y por byte**, no alcanzar un número grande de parámetros. El sistema debe poder crecer, pero su crecimiento debe ser una consecuencia de evidencia y no un sustituto de una base bien entrenada.

## 8. Límites explícitos

Esta especificación no demuestra que un modelo de 100M pueda alcanzar conversación general excelente, razonamiento fuerte, bilingüismo nativo o aprendizaje continuo sin olvido. Esos son objetivos experimentales. La primera versión sólo podrá promoverse cuando los resultados aislados y reproducibles confirmen la competencia base.

## Referencias

[1]: https://arxiv.org/abs/2305.07759 "TinyStories: How Small Can Language Models Be and Still Speak Coherent English?"
[2]: https://arxiv.org/abs/1708.01547 "Lifelong Learning with Dynamically Expandable Networks"
[3]: https://arxiv.org/abs/1606.04671 "Progressive Neural Networks"
[4]: https://arxiv.org/abs/2005.11401 "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
