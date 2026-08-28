# Protocolo de intervención del router — Jitter de selección V1

## Evidencia de partida

La corrida directa inicial produjo **43/768** pasos saludables y la corrección de sesgo de selección produjo **57/768**. En ambos casos, la entropía mínima y el desequilibrio máximo alcanzaron los límites de la puerta de salud. La mejora fue insuficiente para promover un checkpoint o ampliar escala.

## Hipótesis falsable

> La selección top-2 se estabiliza demasiado pronto. Añadir ruido gaussiano reproducible y pequeño (`0.01`) exclusivamente a los logits usados para seleccionar expertos, durante `model.train()`, incrementará la exploración temprana y mejorará la salud global del router sin contaminar las probabilidades densas, la mezcla de expertos, la pérdida auxiliar, la regularización de entropía ni la inferencia.

## Alcance fijo

La intervención conserva: seed 17, 768 pasos, Dataset v1, batch 1, secuencia 1024, BF16, acumulación 16, 8 expertos, top-2, `router_aux_loss_weight=0.05` y `router_entropy_loss_weight=0.03`. No carga checkpoints, no lee holdout, no cambia la arquitectura, no modifica el corpus y no promueve resultados.

## Criterios de clasificación

| Resultado | Criterio |
|---|---|
| Mejora material | Pérdida finita, artefactos completos y más de 57/768 pasos saludables sin violar las guardas de salida. |
| Estabilidad suficiente para siguiente puerta | Pérdida finita, artefactos completos y al menos 384/768 pasos saludables; requiere después evaluación independiente. |
| No mejorado | 57 o menos pasos saludables, pérdida no finita, salida incompleta o fallo de validación. |

## Límites de interpretación

El jitter es una sola sonda de estabilidad del router, no una prueba de razonamiento, bilingüismo, matemáticas, memoria recuperable, rendimiento Triton ni ultra-eficiencia relativa. Cualquier resultado se conserva y se clasifica sin promoción automática.
