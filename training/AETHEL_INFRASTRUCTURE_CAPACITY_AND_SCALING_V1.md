# Capacidad y escalamiento de infraestructura de Aethel v1

**Estado:** especificación de planificación. No representa una reserva de hardware, coste comprometido ni rendimiento medido de Aethel.

## Límite de la infraestructura actual

El entorno de desarrollo actual no dispone de CUDA y no es persistente. Es apropiado para desarrollar código, validar Dataset, ejecutar pruebas CPU y preparar artefactos, pero no puede entrenar Aethel Edge ni alojar un servicio de inferencia sostenido. La GPU gratuita se reserva para una calibración Seed de corta duración; la operación comercial requiere un entorno separado y persistente.

| Escenario | Unidad de cómputo | Trabajo admisible | Evidencia requerida antes de avanzar |
|---|---|---|---|
| Desarrollo | CPU local temporal | Dataset, pruebas de contratos, simulación prohibida, validación de manifests. | Pruebas unitarias y hashes de paquetes. |
| Seed E0 | Una GPU gratuita temporal | Línea base pequeña, preflight CUDA y medición de pérdida/VRAM. | Checkpoint recuperable, evaluación en/en-es y recibo de persistencia. |
| Edge | GPU persistente con VRAM suficiente para pesos, activaciones y caché. | Entrenamiento escalonado, evaluación repetible e inferencia de piloto. | Coste/latencia reales, guardas de datos y recuperación probada. |
| Pro | Varias GPU con interconexión y almacenamiento rápido. | MoE top-2 distribuido, FSDP/experts y validación Triton. | Pruebas multi-GPU, trazas de router y recuperación distribuida. |
| Research | Clúster multi-nodo. | Investigación de familias grandes y comparación controlada. | Gobierno de datos, presupuesto, red y operación explícitos. |

## Criterio de memoria

Un plan no debe tratar la VRAM como si fuese únicamente el tamaño de los pesos. Para entrenamiento se presupuestan pesos, gradientes, estados de optimizador, activaciones, comunicación distribuida, Dataset prefetch y margen de recuperación. Para inferencia se presupuestan pesos, KV-cache, lote/concurrencia y memoria de los servicios de recuperación. La cifra real se registra por corrida; no se fija por anticipado.

## Referencia de escala

NVIDIA publica que H100 SXM ofrece 80 GB de HBM y 900 GB/s de NVLink GPU-a-GPU, mientras que DGX B200 integra ocho GPU y 1.440 GB de memoria GPU agregada. Esas capacidades describen hardware, no el desempeño de Aethel, pero sirven para separar un piloto de una carga distribuida.[1] [2]

> Una unidad de hardware mayor sólo se justifica después de que Aethel Seed haya producido evidencia reproducible. Escalar una receta no validada incrementa coste y riesgo; no convierte por sí mismo una arquitectura en producto.

## Referencias

[1] [NVIDIA H100 GPU](https://www.nvidia.com/en-us/data-center/h100/)  
[2] [NVIDIA DGX B200](https://www.nvidia.com/en-us/data-center/dgx-b200/)
