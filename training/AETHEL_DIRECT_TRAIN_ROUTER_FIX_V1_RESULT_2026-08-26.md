# Resultado de la corrida correctiva del router — Direct Train Router Fix V1

**Fecha:** 26 de agosto de 2026
**Estado:** `MEASURED_NOT_PROMOTED`
**Origen de evidencia:** salida CUDA de Kaggle compartida por el usuario.
**Alcance:** entrenamiento desde inicialización nueva, 768 pasos, Dataset v1 de entrenamiento; sin checkpoint previo, sin holdout y sin promoción.

## Decisión

La revisión `router-selection-debias-v1` mejoró la salud medida del router, pero no alcanzó el criterio predefinido de estabilidad global. El checkpoint es recuperable y la pérdida es finita; sin embargo, la tasa de pasos saludables permanece demasiado baja para promoverlo, ampliar tokens o afirmar estabilidad MoE.

> El sesgo adaptativo quedó separado de las probabilidades usadas para mezclar expertos, calcular entropía y aplicar pérdidas auxiliares. La corrección mejoró el indicador, pero no resolvió por sí sola el colapso temprano de carga.

## Comparación contra la corrida directa V1

| Métrica | Corrida directa V1 | Router Fix V1 | Cambio |
|---|---:|---:|---:|
| Pasos ejecutados | 768 | 768 | 0 |
| Pasos saludables del router | 43 | 57 | +14 |
| Tasa de salud | 5,598958 % | 7,421875 % | +1,822917 puntos porcentuales |
| Mejora relativa de salud | — | — | +32,558139 % |
| Pasos no saludables | 725 | 711 | −14 |
| Entropía mínima | 0,3333333433 | 0,3333333433 | Sin mejora |
| Desequilibrio máximo | 0,1875 | 0,1875 | Sin mejora |
| Pérdida inicial | 10,4948348999 | 10,4948348999 | Igual configuración inicial |
| Pérdida final | 8,1372 | 8,1299285889 | Mejora marginal |
| Reducción de pérdida | 22,464716 % | 22,534002 % | +0,069286 puntos porcentuales |
| Tokens/s medios | 6.168,58 | 6.836,667605 | +10,830492 % |

## Artefactos y telemetría confirmados

La corrida produjo y validó los siguientes artefactos bajo `/kaggle/working/aethel-direct-train-router-fix-v1/`:

| Artefacto | Estado | Interpretación |
|---|---|---|
| `latest.pt` | Recuperable | Checkpoint disponible, no promovido |
| `metrics_rank_0.jsonl` | Emitido | Telemetría por paso disponible |
| `router_diagnostic.json` | Emitido | Resumen de router disponible |
| `aethel_direct_validation.json` | Validado | Validación de artefactos y telemetría completada |

La validación también confirmó telemetría para La Roca, El Líquido, Ciclo de Sueño, memoria episódica y semántica, neuromodulación y Espacio de Trabajo Global. Esto prueba que los módulos emitieron sus trazas dentro del alcance de la corrida; no prueba todavía sus propiedades completas de rollback, consolidación, recuperación, competencia global o promoción.

## Eficiencia observada y límites

La configuración mantuvo activación dispersa top-2 sobre 8 expertos y GQA con 2 cabezas KV para 8 cabezas de atención, ambos con ratio estructural de 0,25. El rendimiento medio medido fue de 6.836,67 tokens/s. Aun así, no se afirma ultra-eficiencia relativa: falta un baseline comparable y una aceptación CUDA/Triton que permita comparar coste, memoria y rendimiento de forma controlada.

## Capacidades que permanecen sin demostrar

La corrida no evaluó razonamiento, bilingüismo nativo, matemáticas, eficiencia relativa, inmutabilidad y rollback de La Roca, promoción LoRA, consolidación completa del Sueño, espacio de trabajo competitivo ni runtime Rust desplegado. Ninguna de esas capacidades debe presentarse como validada.

## Próxima puerta

Antes de otra ejecución GPU se requiere diagnosticar por qué la entropía mínima y el desequilibrio máximo permanecen fijados en sus límites, y diseñar una sola intervención adicional con una hipótesis falsable. El checkpoint Router Fix V1 debe conservarse como evidencia, sin promoción ni reanudación automática.
