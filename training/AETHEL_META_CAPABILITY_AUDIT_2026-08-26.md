# Auditoría de capacidades de Aethel Meta

**Autor:** Manus AI  
**Fecha:** 26 de agosto de 2026  
**Alcance:** comparación estática entre la rama `main` del repositorio original de Aethel Meta y el núcleo actual de `aethel-platform`. Esta auditoría no ejecuta GPU, no carga checkpoints, no abre el corpus ni fabrica métricas.

## Conclusión ejecutiva

El repositorio Meta sí contenía una visión mucho más amplia que un Transformer MoE. La visión combina un **modelo bilingüe eficiente**, tres tipos de memoria, plasticidad limitada, consolidación offline, razonamiento con presupuesto, trazabilidad, gobierno de versiones y varios runtimes especializados. Sin embargo, la auditoría separa con rigor tres niveles: **código presente**, **mecanismos parciales** y **objetivos documentados sin evidencia operativa**. La primera corrida directa debe validar el núcleo y producir un checkpoint; no equivale por sí sola a completar Aethel Meta.

> La filosofía Meta correcta no es añadir nombres neurobiológicos. Es imponer fronteras de mutación, procedencia, evaluación y rollback a cada módulo. [1]

## Matriz canónica de capacidades

| Dominio | Capacidad Meta recuperada | Evidencia en el repositorio | Estado frente al núcleo actual | Puerta de validación necesaria |
|---|---|---|---|---|
| Modelo base | Transformer causal con RoPE, GQA, KV-cache, embeddings atados y Sparse MoE top-2 | Arquitectura y hoja de eficiencia [2] [3] | **Implementado.** El núcleo actual conserva estos elementos y añade el control de entropía del router. | Checkpoint que carga, pérdida finita, generación y telemetría de MoE. |
| Eficiencia MoE | Separar capacidad total de cómputo activo; medir carga, balance, capacidad y tokens descartados | Hoja de eficiencia [2] | **Parcialmente implementado.** Hay telemetría de router; todavía falta comparación contra baseline y medición de tokens descartados/capacidad efectiva. | Baseline idéntico sin la mejora, tokens/s, VRAM, pérdida y salud de router en la misma GPU. |
| La Roca | Base estable, tokenizador, manifiesto de datos y parámetros de seguridad con hash inmutable, promoción y rollback | Modelo operativo [1] | **Parcial.** Existe ruta sólida/ancla; no hay `rock_manifest.json`, firma, promoción ni rollback de una versión entrenada. | Hash antes/después de sesión, manifiesto de versión, carga y rollback atómico. |
| El Líquido | Traza rápida de sesión y adaptadores LoRA candidatos, con TTL, cuarentena, revocación y procedencia | Modelo operativo [1] | **Parcial.** Hay traza hebbiana y eventos; no hay aislamiento por usuario, persistencia en checkpoint, revocación ni adaptación aprobada. | Prueba de que `observe()` no altera pesos, esquema completo de evento, expiración/revocación y candidato LoRA aislado. |
| Memoria de trabajo | Estado recurrente GRU acotado por sesión | Especificación expuesta [4] | **Implementado como módulo.** | Reset explícito, aislamiento de sesión y efecto medido en tareas de contexto. |
| Memoria episódica y semántica | Registro JSONL, recuperación ponderada y prototipos consolidados | Modelo operativo y especificación [1] [4] | **Implementado a nivel de núcleo.** También existe runtime Rust para recuperación citable. | Procedencia, TTL, borrado, privacidad, precisión de recuperación y ausencia de fuga de holdout. |
| Runtime Rust | Servicio local JSONL con `remember`, `retrieve`, `retrieve_context`, `sleep`, `snapshot`, recuperación citable y socket Unix | Implementación Rust [5] | **Código presente y sin diferencias locales relevantes.** No está desplegado 24/7 ni conectado a inferencia. | Compilación, restauración, fallo/reinicio, permisos mínimos, backup y medición bajo carga. |
| Ciclo de Sueño | Curación, desduplicación, replay estratificado, LoRA temporal, regresión y promoción reversible | Modelo operativo [1] | **Parcial.** El replay existe; no existe el pipeline operativo completo de candidato, evaluación, revisión y promoción. | Split auditado, replay por idioma/dominio, retención, adaptación candidata, regresión y rechazo reversible. |
| Espacio de Trabajo Global | Integración de fuentes y evolución hacia bus competitivo de K ranuras con origen, confianza y coste | Modelo operativo [1] | **Parcial.** Existe fusión; no hay bus K-ranuras, competición ni ablación. | Igual presupuesto frente a fusión actual, calidad/coste y traza de fuentes seleccionadas. |
| Neuromodulación y curiosidad | Señales de sorpresa, incertidumbre, novedad, conflicto, riesgo y coste; sólo arbitran recursos | Modelo operativo [1] | **Parcial.** Hay señales y acciones locales acotadas; contradicción/progreso longitudinal siguen incompletos. | Calibración contra error, falsos positivos, relación coste-calidad y prueba de que no inicia acciones externas. |
| ARC/refinamiento adaptativo | Más pasos sólo para estados difíciles dentro de un presupuesto explícito | Especificación y hoja de eficiencia [2] [4] | **Experimental y apagado por defecto.** | Ablación con misma calidad objetivo: pérdida, tokens/s, p50/p95, VRAM y fracción de tokens refinados. |
| Bilingüismo | Entrenamiento y evaluación aislada para español e inglés | Matriz de runtime y producto [3] [6] | **Dataset y contrato presentes; capacidad no demostrada.** | Pérdida, generación y evaluación holdout por idioma sin degradación cruzada. |
| Razonamiento verificable | Recuperación → integración → refinamiento presupuestado → predicción, exponiendo evidencia y no cadena de pensamiento privada | Especificación expuesta [4] | **Protocolo y telemetría parcial presentes; no benchmark real.** | Batería de problemas con respuesta verificable, trazas de fuentes y comparación contra baseline. |
| Triton/CUDA | Kernels para SwiGLU, decode, router top-2 y futuras rutas de prefill/dispatch/El Líquido | Matriz de runtime [3] [7] | **Parcial.** La aceptación CUDA anterior valida alineación de memoria; no acredita rendimiento de todas las rutas. | Paridad numérica, gradientes cuando apliquen, VRAM, rendimiento y rollback por kernel. |
| Inferencia de producto | Runtime local Mojo con paridad PyTorch para prefill/decode/KV-cache | Arquitectura políglota [7] | **Sólo contrato.** No hay runtime Mojo validado con pesos reales. | Exportación con hashes, paridad logits/tokens, KV-cache y benchmark de latencia/memoria. |
| Escalado | Familias piloto, investigación 300M, 1B y Edge/Pro, condicionadas por evidencia | Presets y variantes [4] [8] | **Diseño.** La corrida actual es de calibración; no autoriza escalar automáticamente. | Datos/cómputo proporcionados, FSDP multi-GPU, estabilidad MoE, evaluación y coste por variante. |
| Producto Workspace | Conversación bilingüe, conocimiento privado con fuentes, memoria gobernada, aprendizaje controlado y operación con rollback | Especificación comercial [6] | **Visión de producto, no producto listo.** | Acceso, separación de tenants, recuperación citable, soporte, observabilidad y datos reales de coste/calidad. |

## Qué cambia en la corrida directa

La corrida directa conserva el valor de iniciar pronto: debe producir un checkpoint real, `metrics_rank_0.jsonl`, `router_diagnostic.json`, recibo de recuperación y el validador de pilares. No obstante, el validador actual sólo demuestra **presencia de telemetría y artefactos**, no eficacia cognitiva, seguridad de promoción ni ultra-eficiencia demostrada.

Por tanto, el estado correcto al finalizar la CELDA 3 será:

| Resultado posible | Interpretación correcta |
|---|---|
| Checkpoint y validación de artefactos pasan | El núcleo entrenó de forma recuperable y emitió evidencia mínima. |
| MoE saludable y pérdida estable | Señal favorable para seguir; todavía no prueba razonamiento ni bilingüismo nativo. |
| Memoria/replay/neuromodulación aparecen en telemetría | Los módulos participaron; falta probar que mejoren una tarea o la eficiencia. |
| Throughput medido | Línea base de hardware; no prueba ultra-eficiencia sin una ablación comparable. |
| Router inestable, OOM o pérdida no finita | La corrida no autoriza escalar; se conserva evidencia y se corrige la causa concreta. |

## Puertas de integración recuperadas de Meta

La ruta debe priorizar evidencia y no acumular variantes. Después del checkpoint inicial, las puertas se ejecutarán en este orden:

1. **Núcleo verificable:** checkpoint recuperable, evaluación bilingüe separada, generación mínima y baseline de eficiencia.
2. **Gobierno de La Roca:** manifiesto hashable, rollback y candidato LoRA separado.
3. **Sueño gobernado:** replay estratificado, auditoría de holdout, regresión y promoción reversible.
4. **Workspace y ARC:** ablaciones con presupuesto igual para justificar calidad o ahorro real.
5. **Runtime y producto:** servicio Rust supervisado, recuperación citable, inferencia propia validada y políticas de acceso.
6. **Escalado:** pasar de piloto a 300M/1B sólo con datos, hardware, tests distribuidos y métricas que demuestren que el modelo más grande aporta valor neto.

## Decisión recomendada

Se debe mantener el cuaderno nuevo y el dataset de código aislado para evitar residuos históricos. Antes de ejecutar la CELDA 3, el bundle debe incorporar esta auditoría y el validador de pilares. Después de la corrida se entregará una clasificación real de cada pilar: **validado**, **telemetría presente pero sin beneficio demostrado**, **fallido** o **no ejecutado**. Ninguna categoría se inferirá de la documentación o del nombre del módulo.

## Referencias

[1]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/ARCHITECTURE_COGNITIVE_OPERATING_MODEL.md "Modelo operativo cognitivo verificable"
[2]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/EFFICIENCY_AND_REASONING_ROADMAP.md "Eficiencia y razonamiento verificable"
[3]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/AETHEL_LANGUAGE_AND_RUNTIME_ARCHITECTURE_V1.md "Matriz de lenguajes y runtimes"
[4]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/server/aethelSpecs.ts "Especificación expuesta de Aethel"
[5]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/runtime/aethel-memory-rust/src/lib.rs "Runtime Rust de memoria"
[6]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/training/AETHEL_COMMERCIAL_PRODUCT_SPEC_V1.md "Especificación comercial de Aethel Workspace"
[7]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/POLYGLOT_RUNTIME_ARCHITECTURE.md "Arquitectura políglota objetivo"
[8]: https://github.com/Robbin360/repo-de-ia-de-nueva-generaci-n-/blob/main/training/AETHEL_TECHNICAL_VARIANTS_SPEC_V1.md "Variantes técnicas escalonadas"
