# Ficha técnica escalonada de Aethel v1

**Estado de las cifras:** objetivos de diseño, no modelos entrenados ni benchmarks. Los parámetros reales se recalcularán a partir del código y se publicarán sólo junto a checkpoints verificables.

| Variante | Estado | Núcleo propuesto | Parámetros | Uso objetivo |
|---|---|---|---:|---|
| **Seed E0** | Preparada; no entrenada con GPU. | 4 capas, dim 512, 8 cabezas, 2 KV heads, 8 expertos/top-2, contexto 1.024. | Se registrará desde el checkpoint. | Calibrar Dataset, router, checkpoints y evaluación bilingüe. |
| **Edge** | Objetivo futuro. | Denso, 28 capas, dim 2.560, 20 cabezas, 5 KV heads. | ≈2,2 B objetivo. | Asistente técnico privado, cuantizable y con memoria gobernada. |
| **Pro** | Objetivo de investigación/producto empresarial. | 32 capas, dim 4.096, 32 cabezas, 8 KV heads, 8 expertos top-2 y SwiGLU 11.008. | ≈36,4 B totales; ≈10,4 B activos/token. | Razonamiento y conocimiento técnico bilingüe de alta capacidad. |
| **Research** | Sin microarquitectura congelada. | Escalado posterior a evidencia Edge/Pro. | ≈139 B como referencia de familia. | Investigación multi-GPU; no plan de lanzamiento inicial. |

Todas las variantes comparten RoPE, GQA cuando aplique, KV-cache, telemetría de router, La Roca como referencia inmutable y servicios líquidos fuera de la ruta crítica. El uso de MoE no elimina el coste de comunicación, capacidad de expertos o memoria de pesos; por ello Pro no se inicia antes de validar kernels, balance y recuperación distribuida.

## Requisitos de inferencia por variante

| Variante | Ruta de modelo | Servicios complementarios | Criterio mínimo antes de servir usuarios |
|---|---|---|---|
| Seed | PyTorch de laboratorio. | Ninguno obligatorio. | Checkpoint, evaluación por idioma y restauración real. |
| Edge | Runtime GPU o local cuantizado pendiente de validación. | Memoria/RAG y gobierno de sesiones en CPU/Rust. | Latencia y calidad observadas, política de privacidad y rollback. |
| Pro | GPU con MoE y kernel especializado. | Memoria aislada, cola de Sueño y auditoría. | Triton prefill/dispatch validado, router sano y coste medido. |

La “inteligencia” no se deduce del número de parámetros. La aprobación depende de resultados retenidos, seguridad, utilidad para la tarea y coste observado.
