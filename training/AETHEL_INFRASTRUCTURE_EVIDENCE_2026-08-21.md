# Evidencia de infraestructura para Aethel — 2026-08-21

Este registro conserva información externa usada para dimensionar escenarios técnicos. No constituye una estimación de coste ni una promesa de rendimiento de Aethel.

| Plataforma | Evidencia oficial relevante | Implicación arquitectónica |
|---|---|---|
| NVIDIA H100 SXM | NVIDIA declara 80 GB de memoria HBM, 3,35 TB/s de ancho de banda de memoria y 900 GB/s de NVLink GPU-a-GPU; H100 NVL declara 94 GB de memoria por GPU y 600 GB/s de NVLink. | Adecuada para validar kernels Triton, entrenar pilotos pequeños/medianos y ejecutar ajuste distribuido limitado, siempre que el modelo, activaciones y optimizador quepan en la memoria disponible. |
| NVIDIA DGX B200 | NVIDIA especifica 8 GPU Blackwell, 1.440 GB de memoria GPU total, 64 TB/s de ancho de banda HBM3e agregado, 14,4 TB/s de NVLink agregado, hasta 4×400 Gb/s de red y aproximadamente 14,3 kW máximos del sistema. | Representa una unidad coherente para entrenamiento y evaluación distribuidos de modelos más grandes; requiere software distribuido, almacenamiento de alto rendimiento, red y operación de infraestructura, no sólo GPU. |
| NVIDIA GB200 NVL72 | NVIDIA especifica 72 GPU Blackwell, 13,4 TB HBM3e agregados, 576 TB/s de ancho de banda HBM agregado y 130 TB/s de NVLink en un dominio de 72 GPU. | Es infraestructura de rack para modelos de escala muy grande y MoE distribuido; no es necesaria para demostrar Aethel, pero ilustra el orden de recursos requerido para perseguir preentrenamiento de frontera. |
| NVIDIA DGX Cloud | NVIDIA describe una plataforma gestionada para entrenamiento y operación a escala, y señala uso de patrones de operación multi-nodo sobre decenas de miles de GPU para sus propias cargas. | Una alternativa futura a poseer infraestructura es una plataforma administrada; aun así exige control de datos, presupuesto, reproducibilidad y planificación de checkpoints. |

## Fuentes oficiales

1. [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/)
2. [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
3. [NVIDIA GB200 NVL72](https://www.nvidia.com/en-us/data-center/gb200-nvl72/)
4. [NVIDIA DGX Cloud](https://www.nvidia.com/en-us/data-center/dgx-cloud/)
