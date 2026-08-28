# Resultado de Aethel — Entrenamiento Directo V1

**Fecha de evidencia:** 26 de agosto de 2026
**Fuente:** salida textual compartida por el usuario desde el cuaderno Kaggle limpio `Aethel — Entrenamiento Directo Dataset V1`.
**Estado de promoción:** `MEASURED_NOT_PROMOTED`.

## Alcance ejecutado

La corrida terminó 768 pasos desde inicialización fresca sobre los shards de entrenamiento del Dataset v1. La evidencia declara que no cargó checkpoint, no leyó holdout y no realizó solicitudes de red. El checkpoint recuperable quedó en `/kaggle/working/aethel-direct-train-v1/latest.pt`.

| Métrica | Resultado medido |
|---|---:|
| Pasos | 768 / 768 |
| Tokens observados | 786.432 |
| Parámetros entrenables | 97.154.564 |
| Pérdida inicial | 10,4948348999 |
| Pérdida final | 8,1372184753 |
| Reducción absoluta de pérdida | 2,3576164246 |
| Reducción relativa de pérdida | 22,4645 % |
| Rendimiento medio informado | 6.168,5772 tokens/s |
| Rendimiento final informado | 6.864,4344 tokens/s |

## Router MoE

El router ejecutó 8 expertos con activación top-2. Aunque hubo pérdida descendente y el último estado instantáneo compartido se marcó saludable, el resumen de los 768 pasos no soporta promoción: sólo 43 pasos fueron saludables (5,5989 %) y 725 no saludables (94,4010 %). La entropía mínima alcanzó el suelo de 0,3333333433 y el desequilibrio máximo fue 0,1875.

> **Conclusión de router:** checkpoint conservado como evidencia; estabilidad global insuficiente para promoción, ampliación de entrenamiento o afirmaciones de capacidad. La corrida directa no demuestra que el peso de entropía 0,03 haya resuelto el colapso.

## Pilares Aethel

| Pilar | Evidencia emitida | Clasificación honesta |
|---|---|---|
| La Roca | Telemetría presente; 3 eventos de pérdida de replay | Telemetría presente; rollback/inmutabilidad no probados |
| El Líquido | Telemetría presente; versión 7 | Telemetría presente; no se promovió adaptador LoRA |
| Ciclo de Sueño | 7 registros de replay | Telemetría presente; consolidación completa no probada |
| Memoria | 7 registros episódicos y 7 semánticos | Registros emitidos; recuperación y calidad no probadas |
| Neuromodulación | Telemetría presente; valor final 0,7635257244; sorpresa final 8,1372184753 | Telemetría presente; política no validada |
| Espacio de Trabajo Global | Telemetría presente | Workspace competitivo no probado |
| MoE | 8 expertos, top-2, 43/768 pasos saludables | Fallido como estabilidad global de promoción |

## Eficiencia

La arquitectura conserva evidencia estructural de ahorro: el MoE activa 2 de 8 expertos (ratio activo 0,25) y GQA usa 2 cabezas KV para 8 cabezas de atención (ratio KV 0,25). Esas dos relaciones evitan el 75 % de expertos no seleccionados y el 75 % de cabezas KV respecto de sus alternativas densas internas, respectivamente. Sin embargo, no existe aún corrida baseline comparable y `require_triton` estuvo desactivado; por tanto no se puede afirmar ultra-eficiencia medida, aceleración Triton ni ventaja relativa de coste.

## Límites explícitos de la evidencia

No se demostró razonamiento, bilingüismo nativo, matemáticas, generación de calidad, eficiencia frente a baseline, inmutabilidad de La Roca, promoción LoRA, consolidación de Sueño completa, workspace competitivo ni runtime Rust desplegado. El siguiente paso válido es analizar y corregir la estabilidad del router antes de aumentar tokens, tamaño o alcance cognitivo.
