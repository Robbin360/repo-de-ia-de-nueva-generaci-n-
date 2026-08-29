# Aethel D1D — Handoff de decisión

**Fecha:** 2026-08-28  
**Experimento:** regularización de entropía densa del router MoE  
**Estado:** `D1D_ROUTER_NOT_IMPROVED`  
**Alcance:** train-only; no es una evaluación de calidad del modelo.

## Decisión

D1D queda cerrada como intervención no mejorada bajo sus criterios globales predefinidos. No se promueve el checkpoint, no se abre el holdout, no se reanuda entrenamiento desde esta salida, no se publica ni se habilita serving.

## Evidencia registrada

La corrida se ejecutó con inicialización nueva, seed `17`, `768` pasos y `1.572.864` tokens. El peso de entropía fue `0.01`; el término auxiliar histórico permaneció en `0.05`. La corrida confirmó `checkpoint_loaded=false`, `holdout_content_read=false` y `network_requests=0`.

| Métrica | Criterio | Resultado | Estado |
|---|---:|---:|---|
| Pasos saludables | `>=117/768` | `52/768` | No cumple |
| Entropía mínima global | `>0.5` | `0.3333333433` | No cumple |
| Desequilibrio máximo global | `<0.3` | `0.1875` | Cumple |
| Pérdida media | `<=9.53127756` | `9.3994066852` | Cumple |

El último paso fue saludable de forma aislada (`entropía mínima 0.5975525975`, desequilibrio máximo `0.1442871094`), pero no reemplaza los criterios definidos sobre la corrida completa. La señal tardía es una observación, no una justificación para seleccionar el mejor tramo después de observarlo.

## Interpretación

El resultado descarta esta configuración como solución suficiente para la estabilidad temporal del router. El fracaso se concentra en salud global y estabilidad, no en un deterioro de la pérdida media. No se debe concluir que toda regularización de entropía sea inútil; cualquier variante futura debe tener un protocolo nuevo, una salida inédita, criterios fijados antes de observar resultados y una prueba de no regresión.

## Próximo paso permitido

Sólo se permite trabajo local de análisis o un protocolo posterior independiente. Una nueva GPU, edición de notebook, cambio de peso, nueva sesión Kaggle, carga de checkpoint o modificación de datos requiere autorización específica y separada.

## Referencias internas

- `training/AETHEL_D1D_ROUTER_ENTROPY_PROTOCOL_2026-08-25.md`
- `training/AETHEL_D1D_RESULT_2026-08-25.md`
- `todo.md`

Este documento no afirma que Aethel sea un modelo funcional, bilingüe, razonador, frontier o de inteligencia general.

---

*Documento de handoff; no contiene pesos ni corpus.*

Contenido basado únicamente en el protocolo y resultado D1D registrados en el repositorio.
