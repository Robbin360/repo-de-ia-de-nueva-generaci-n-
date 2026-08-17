# Fuentes primarias — eficiencia de Aethel

## Escalado densidad-datos

- **Hoffmann et al., 2022 — Training Compute-Optimal Large Language Models.** El estudio entrena más de 400 modelos entre 70 M y 16 B de parámetros y concluye que, bajo su régimen, el tamaño del modelo y el número de tokens deben crecer conjuntamente para un entrenamiento eficiente en cómputo. Fuente: https://arxiv.org/abs/2203.15556

## Mezcla dispersa de expertos

- **Du et al., 2022 — GLaM: Efficient Scaling of Language Models with Mixture-of-Experts.** Estudia modelos decoder-only dispersos y el escalado eficiente mediante rutas de expertos activadas selectivamente. Fuente: http://proceedings.mlr.press/v162/du22c.html
- **Dai et al., 2024 — DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models.** Propone expertos de grano fino y un experto compartido para mejorar especialización y eficiencia de MoE. Fuente: https://arxiv.org/abs/2401.06066
- **Krajewski et al., 2025 — Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models.** Estudia la relación entre parámetros, FLOPs y nivel de dispersión para MoE. Fuente: https://arxiv.org/abs/2501.12370

## Secuencias e inferencia

- **Gu y Dao, 2023 — Mamba: Linear-Time Sequence Modeling with Selective State Spaces.** Propone un modelo de espacio de estados selectivo de tiempo lineal para secuencias largas. Es una alternativa para bloques selectivos, no una sustitución no validada de todo el Transformer de Aethel. Fuente: https://arxiv.org/abs/2312.00752
- **Gloeckle et al., 2024 — Better & Faster Large Language Models via Multi-token Prediction.** Evalúa entrenar para predecir varios tokens futuros; reporta mejoras de eficiencia de muestra y de razonamiento en sus configuraciones evaluadas. Fuente: https://arxiv.org/abs/2404.19737
- **Leviathan, Kalman y Matias, 2023 — Fast Inference from Transformers via Speculative Decoding.** Estudia generar propuestas con un modelo auxiliar y verificarlas con el modelo principal para acelerar decodificación sin cambiar la distribución objetivo bajo las condiciones del método. Fuente: https://proceedings.mlr.press/v202/leviathan23a.html

## Adaptación con menos parámetros entrenables

- **Hu et al., 2021 — LoRA: Low-Rank Adaptation of Large Language Models.** Congela pesos base e inyecta matrices entrenables de bajo rango; el trabajo compara la adaptación contra ajuste completo en varios modelos. Fuente: https://arxiv.org/abs/2106.09685
- **Dettmers et al., 2023 — QLoRA: Efficient Finetuning of Quantized LLMs.** Combina cuantización de la base y adaptadores de bajo rango para reducir memoria durante ajuste fino. La técnica debe evaluarse para el stack específico de Aethel antes de adoptar sus resultados como garantía. Fuente: https://arxiv.org/abs/2305.14314
