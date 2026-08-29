# Sonda load-aware del router — 2026-08-28

## Estado

`PROBE_ONLY`. La sonda no modifica `AethelModel`, no entrena, no carga pesos, no usa GPU y no ejecuta datos externos.

## Hipótesis

Si una pareja de expertos está saturada, restar a sus logits una penalización proporcional a una carga histórica centrada podría cambiar la asignación top-k dura y reducir concentración.

## Caso controlado

Logits iniciales: `[1.0, 0.99, 0.2, 0.1]`.

Carga histórica: `[10.0, 9.0, 0.0, 0.0]`.

Con `k=2`, la selección inicial es `(0, 1)`. Con `strength=0.2`, la selección ajustada es `(2, 3)`.

## Validación

`engine/router_load_aware_probe.py` y `engine/test_router_load_aware_probe.py` completan sus contratos CPU deterministas: cambio de top-k en el caso saturado, identidad con fuerza cero y rechazo de longitudes incompatibles o valores no finitos.

## Interpretación conservadora

El resultado sólo demuestra que una penalización load-aware puede cambiar una selección top-k sintética cuando la carga histórica es extrema. No demuestra estabilidad durante entrenamiento, calidad, ausencia de oscilaciones, compatibilidad con gradientes, ganancia de throughput ni mejora en una GPU. Antes de integrarla habría que ejecutar una ablación con EMA, límites de penalización, capacidad por experto, overflow, pérdida principal, entropía, cobertura y rendimiento.

La sonda queda separada del router principal para permitir rollback y comparación limpia.
