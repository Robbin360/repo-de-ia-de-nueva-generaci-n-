# Estado de preparación Triton — CPU

**Fecha:** 2026-08-19  
**Alcance:** verificación local sin GPU, sin entrenamiento y sin datos de corpus.

## Resultado observado

La ejecución de `engine/test_triton_bridge.py` confirmó los fallbacks CPU para SwiGLU, atención de decodificación causal y selección top-2 del router. La ejecución de `engine/test_triton_gpu.py` se detuvo correctamente con estado `SKIPPED`: CUDA y Triton no estaban disponibles en este entorno.

| Ruta | Estado CPU | Estado GPU |
|---|---|---|
| SwiGLU fusionado | Fallback PyTorch verificado | Pendiente de equivalencia y medición real. |
| Atención de decodificación con KV-cache | Fallback SDPA verificado | Pendiente de equivalencia y medición real. |
| Router top-2 MoE | Fallback PyTorch verificado | Pendiente de equivalencia y medición real. |
| Atención causal de prefill por bloques | No declarada como kernel validado | Requiere implementación y validación real. |
| Dispatch/combina de expertos MoE | No declarada como kernel validado | Requiere implementación y validación real. |

> Ninguna de estas pruebas demuestra aceleración. Una validación posterior deberá exigir CUDA disponible, Triton instalado, paridad numérica contra referencias PyTorch, prueba de gradientes donde corresponda, memoria pico y tokens por segundo en la GPU objetivo.

## Bloqueo vigente

La tarea de ruta Triton permanece abierta. No debe afirmarse que Aethel tiene FlashAttention propio ni dispatch MoE Triton de producción hasta que las pruebas GPU anteriores estén realizadas y registradas.
