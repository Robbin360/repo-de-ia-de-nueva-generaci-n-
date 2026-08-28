# Resultado de la corrida correctiva `DIRECT_TRAIN_ROUTER_JITTER_V1`

## Alcance y límites

Este informe registra únicamente la salida de Kaggle compartida por el usuario el 26 de agosto de 2026. La corrida empezó desde inicialización nueva, ejecutó 768 pasos sobre entradas de entrenamiento, no cargó checkpoints previos, no leyó holdout y no autorizó promoción. Por ello, esta evidencia permite clasificar el entrenamiento, la telemetría de pilares y la estabilidad medida del router, pero no demuestra razonamiento, bilingüismo nativo, matemáticas, eficiencia relativa, consolidación completa de Sueño ni capacidades de producto.

## Configuración ejecutada

| Campo | Valor medido |
|---|---:|
| Diagnóstico | `DIRECT_TRAIN_ROUTER_JITTER_V1` |
| Capas / expertos / activos | 4 / 8 / 2 |
| Parámetros entrenables | 97.154.564 |
| Pasos | 768 / 768 |
| Tokens finales | 786.432 |
| Jitter de selección | 0,01, sólo durante entrenamiento |
| Pérdida auxiliar / entropía | 0,05 / 0,03 |
| Checkpoint | Recuperable, no promovido |

## Resultado frente a corridas anteriores

| Corrida | Pasos saludables | Tasa saludable | Clasificación |
|---|---:|---:|---|
| Direct Train V1 | 43 / 768 | 5,60 % | Router no estable globalmente |
| Router Fix V1 | 57 / 768 | 7,42 % | Mejora insuficiente |
| Router Jitter V1 | 446 / 768 | 58,07 % | Mejora material; estabilidad aún no global |

La intervención de jitter aumentó los pasos saludables en 403 respecto a Direct Train V1 y en 389 respecto a Router Fix V1. La salida final fue saludable, con entropía mínima por capa de 0,9851 y desequilibrio máximo de 0,0240 en el último paso. Sin embargo, el agregado completo conserva extremos tempranos de entropía 0,3333 y desequilibrio 0,1875; por ello el checkpoint sigue clasificado como **medido y no promovido**.

## Métricas de entrenamiento y eficiencia estructural

| Métrica | Valor medido |
|---|---:|
| Pérdida inicial / final | 10,4951 / 8,1311 |
| Pérdida media | 9,3555 |
| Tokens/s medio / final | 6.664,31 / 6.755,89 |
| Activación MoE dispersa | 2 / 8 expertos = 25 % |
| Ratio GQA KV | 2 / 8 cabezas = 25 % |
| Comparación baseline | Pendiente de corrida separada |

## Pilares observados

La validación de artefactos registró telemetría de La Roca, El Líquido, Ciclo de Sueño, memoria episódica y semántica, neuromodulación, curiosidad y espacio de trabajo global. Estos indicadores prueban emisión de telemetría y archivos de la corrida; no prueban todavía sus garantías de rollback, promoción LoRA, consolidación de Sueño, recuperación correcta, política adaptativa ni runtime Rust desplegado.

## Decisión

La hipótesis de que un jitter de selección acotado reduciría la concentración temprana del router quedó **parcialmente respaldada**: la tasa saludable pasó de 7,42 % a 58,07 % sin pérdida no finita y con checkpoint recuperable. No se autoriza promoción ni escalado porque 322 de 768 pasos permanecen no saludables y los extremos globales siguen presentes. El siguiente análisis debe localizar en qué intervalo inicial ocurre la concentración y decidir, antes de otra GPU, si corresponde introducir un calentamiento de balanceo o revisar el criterio de salud temporal.
