# Evidencia para mejorar y escalar Aethel

Este documento registra decisiones de ingeniería; no implica que Aethel haya alcanzado resultados de los artículos citados.

| Riesgo | Hallazgo relevante | Aplicación a Aethel |
|---|---|---|
| Colapso de expertos MoE | El desbalance de carga provoca colapso de routing o sobrecarga; un sesgo dinámico por experto puede equilibrar el routing sin gradientes auxiliares interferentes [1]. | Medir carga por capa, ajustar un sesgo de router de forma lenta, limitar su cambio y conservar la pérdida auxiliar actual como salvaguarda configurable. |
| Olvido catastrófico | El replay de representaciones y un modelo EMA de referencia mejoran la retención y robustez en aprendizaje continuo, aunque la evidencia procede de visión [2]. | Añadir replay estratificado por saliencia/edad y una referencia congelada o EMA para medir deriva; no actualizar El Líquido sin una señal de sorpresa y un control de regresión. |
| Escalado ineficiente | El estudio Chinchilla encontró que, para su familia de modelos y presupuesto, tamaño del modelo y tokens debían crecer conjuntamente; modelos grandes pueden quedar subentrenados [3]. | Elegir parámetros y tokens por presupuesto medido, no por tamaño aspiracional; hacer pilotos de escalado antes de la corrida larga y guardar curvas de pérdida/validación. |

## Decisiones de diseño resultantes

1. **Router estable.** Aethel usará telemetría por experto y una corrección de sesgo limitada; el entrenamiento se detendrá o degradará a una configuración segura si detecta inactividad persistente de expertos.
2. **El Líquido regulado.** La plasticidad seguirá versionada, pero su consolidación se condicionará a sorpresa, diversidad del replay y una prueba de no-regresión sobre un holdout congelado.
3. **Currículo antes de escala.** La primera etapa priorizará texto curado y deduplicado, seguida de datos técnicos/código y razonamiento, solo si las métricas de calidad y seguridad lo justifican.
4. **Hitos verificables.** Los resultados se expresarán mediante pérdida de validación, distribución de experts, retención tras tareas secuenciales y benchmarks reproducibles, nunca por afirmaciones de consciencia o inteligencia humana.

## Fuentes

[1] Wang et al., *Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts* (2024): https://arxiv.org/abs/2408.15664

[2] Jeeveswaran et al., *BiRT: Bio-inspired Replay in Vision Transformers for Continual Learning* (2023): https://arxiv.org/abs/2305.04769

[3] Hoffmann et al., *Training Compute-Optimal Large Language Models* (2022): https://arxiv.org/abs/2203.15556
