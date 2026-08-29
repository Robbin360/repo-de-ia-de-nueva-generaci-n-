# Resultado de sonda: ruido reproducible antes de top-k

**Fecha:** 2026-08-28  
**Estado:** diagnóstico CPU aislado; no modifica el router ni el entrenador.

## Resultado

Con 128 tokens, 8 expertos, top-k=2 y logits inicialmente concentrados en los expertos 0 y 1:

| Sigma | Cobertura limpia | Cobertura con ruido | Concentración limpia | Concentración con ruido |
|---:|---:|---:|---:|---:|
| 0.00 | 0.25 | 0.25 | 0.50 | 0.50 |
| 0.01 | 0.25 | 0.25 | 0.50 | 0.50 |
| 0.05 | 0.25 | 0.25 | 0.50 | 0.50 |
| 0.20 | 0.25 | 0.375 | 0.50 | 0.50 |

La sonda es determinista para la misma semilla y configuración. En este escenario, el ruido grande amplía algo la cobertura, pero no reduce la concentración máxima: los expertos dominantes siguen recibiendo la mitad de las asignaciones. Por ello no se integra automáticamente al entrenamiento ni se presenta como solución del colapso. La siguiente decisión requiere una ablación con pérdida de balanceo, capacidad/overflow y métricas sobre secuencias reales.
