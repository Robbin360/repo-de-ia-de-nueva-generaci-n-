# Revisión de evidencia del router D1A–D1B

**Estado:** `DOCUMENTARY_REVIEW_ONLY`  
**Fecha:** 2026-08-23  
**Autor:** Manus AI  
**Alcance:** comparación de resúmenes seguros ya registrados; no se abrieron outputs, checkpoints, corpus, shards ni holdout.

## Propósito y límites

Esta revisión contrasta D1A y D1B para evaluar una única hipótesis operacional: si reducir `router_bias_step` de `0.05` a `0.01`, manteniendo el resto del diseño diagnóstico, mejoraba la salud del router MoE. No es una comparación de capacidades lingüísticas, un benchmark, una selección de modelo ni una prueba causal concluyente. D1A y D1B partieron de inicialización nueva y sólo usaron el alcance permitido de *train*; ambos resúmenes declaran que no cargaron checkpoints, no abrieron holdout, no leyeron corpus crudo, no usaron red y no autorizaron promoción. [1] [2]

> **Regla de interpretación:** la evidencia muestra qué ocurrió en estas dos corridas diagnósticas. No demuestra que el sesgo sea la causa única del fallo del router ni justifica alterar datos, abrir holdout, reanudar pesos o iniciar D2/D3.

## Comparación observada

| Métrica resumida | D1A: `router_bias_step=0.05` | D1B: `router_bias_step=0.01` | Lectura limitada |
|---|---:|---:|---|
| Pasos / tokens | 768 / 1.572.864 | 768 / 1.572.864 | El presupuesto diagnóstico fue igual. |
| Pasos saludables | 78 (10,156250 %) | 44 (5,729166 %) | D1B tuvo 34 pasos saludables menos; descenso relativo de 43,589743 %. |
| Pasos no saludables | 690 (89,843750 %) | 724 (94,270833 %) | El router siguió no saludable y el conteo no saludable aumentó. |
| Entropía mínima | 0,333333 | 0,333333 | No hubo mejora en este extremo observado. |
| Desequilibrio máximo | 0,187500 | 0,187500 | No hubo mejora en este extremo observado. |
| Pérdida mínima | 7,648315 | 7,667897 | D1B fue 0,019582 mayor en esta estadística descriptiva. |
| Pérdida media | 9,259973 | 9,273529 | D1B fue 0,013557 mayor, aproximadamente 0,146399 %. |
| Pérdida máxima | 10,438221 | 10,441325 | D1B fue 0,003104 mayor. |

Los valores de D1A provienen de su resumen seguro y los de D1B de la salida final resumida registrada. [1] [2] La igualdad de pasos y tokens permite una comparación descriptiva directa, pero no elimina fuentes de variación no medidas ni convierte dos corridas en una demostración causal.

## Hipótesis examinadas

| Hipótesis | Predicción que habría apoyado la hipótesis | Observación | Estado documental |
|---|---|---|---|
| Un paso de sesgo `0.05` era demasiado agresivo para esta ventana. | Con `0.01`, aumentarían los pasos saludables y disminuirían los no saludables. | Los saludables bajaron de 78 a 44 y los no saludables subieron de 690 a 724. | **No apoyada para la dirección probada.** |
| Reducir el paso de sesgo resolvería el extremo de entropía. | La entropía mínima superaría el valor de D1A. | Ambos mínimos fueron 0,333333. | **No apoyada.** |
| Reducir el paso de sesgo reduciría el extremo de desequilibrio. | El desequilibrio máximo sería menor que 0,187500. | Ambos máximos fueron 0,187500. | **No apoyada.** |
| La modificación produciría una mejora general visible en pérdida. | Las estadísticas de pérdida bajarían de forma consistente. | Mínimo, media y máximo fueron ligeramente mayores en D1B. | **No apoyada; no es una prueba de calidad.** |
| El sesgo del router es la causa única del problema. | Dos corridas bastarían para aislar causalidad. | Sólo se examinó una dirección de un parámetro en dos diagnósticos. | **No evaluable.** |

La conclusión operativa es estrecha: **`router_bias_step=0.01` no debe considerarse una mejora sobre `0.05` para esta ventana diagnóstica concreta**. Esta conclusión no autoriza revertir, retocar otros hiperparámetros, repetir D1B, iniciar otro diagnóstico ni modificar el Dataset. Cualquier propuesta futura tendría que plantear una hipótesis distinta, pre-registrar criterios *train-only*, conservar el holdout sellado y atravesar nuevamente todas las puertas de autorización.

## Decisiones y bloqueos

La revisión descarta usar D1B como base para un candidato o para una promoción. D2, D3, evaluación holdout, serving, movimiento o inspección de artifacts/checkpoints y cambios del Dataset permanecen bloqueados. La opción `router_bias_step=0.01` queda registrada como **no mejorada en D1B**, no como un ajuste recomendado.

No se propone ejecutar ninguna acción adicional en este documento. Antes de cualquier futura investigación de router haría falta, en orden, una nueva pregunta experimental independiente, un protocolo documental, contratos locales, un release privado de código, una celda bloqueada, autorizaciones inmediatas de notebook/GPU/ejecución y sólo entonces una corrida aislada. Ninguna de esas puertas se abre mediante esta revisión.

## Referencias

[1]: ./d1a_v1_router_diagnostic_evidence.json "Resumen seguro D1A"

[2]: ./d1b_v1_router_diagnostic_evidence.json "Resumen seguro D1B"

[3]: ./AETHEL_D1B_ROUTER_BIAS_PROTOCOL_2026-08-23.md "Protocolo y límites D1B"
