# Investigación de balanceo MoE para Aethel

## Fuentes primarias consultadas

La documentación de **Switch Transformers** identifica la inestabilidad de entrenamiento como un obstáculo práctico para MoE disperso y describe el entrenamiento en BF16 como parte de su solución operativa.[1]

El trabajo de DeepSeek sobre **Auxiliary-Loss-Free Load Balancing** explica que el desequilibrio puede causar colapso de routing o sobrecarga. Propone seleccionar expertos con puntuaciones que incluyen un sesgo por experto actualizado con la carga histórica, pero conservar las puntuaciones de gating sin ese sesgo para ponderar la salida; así evita gradientes de interferencia de una pérdida auxiliar fuerte.[2]

El informe técnico de DeepSeek-V3 adopta explícitamente una estrategia de balanceo sin pérdida auxiliar para limitar la degradación de rendimiento causada por regularización de balanceo, además de una arquitectura MoE eficiente.[3]

Shazeer et al. introducen **Noisy Top-K Gating** como mecanismo de exploración de selección dispersa; el ruido se aplica a la selección, no a los pesos deterministas de inferencia.[4]

ST-MoE propone la **router z-loss** para estabilizar la escala numérica de los logits. Es una intervención distinta del jitter de selección y no se mezclará en la próxima prueba, para conservar atribución causal.[5]

## Implicación para Aethel

La corrección candidata debe ser mínima y falsable: separar la **selección** top-2, que puede usar el sesgo adaptativo, de la **ponderación** de los expertos seleccionados, que debe derivar de las probabilidades no sesgadas. La corrección no autoriza una corrida GPU por sí sola: requiere pruebas locales, umbrales predefinidos y autorización explícita del usuario.

## Referencias

[1]: https://arxiv.org/abs/2101.03961
[2]: https://arxiv.org/html/2408.15664v1
[3]: https://arxiv.org/html/2412.19437v2
[4]: https://arxiv.org/abs/1701.06538
[5]: https://arxiv.org/abs/2202.08906
