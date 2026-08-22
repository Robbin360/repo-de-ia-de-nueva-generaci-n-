# Auditoría de brechas Triton de Aethel

**Fecha:** 22 de agosto de 2026  
**Estado:** revisión estática y CPU; no contiene validación CUDA ni métricas de rendimiento GPU.

## Hallazgos confirmados

| Ruta | Estado en código | Protección actual | Brecha restante |
|---|---|---|---|
| SwiGLU | Kernel Triton `_swiglu_kernel` y fallback PyTorch. | `require_triton` rechaza el fallback en modo estricto. | Falta prueba numérica/rendimiento CUDA por configuración representativa. |
| Decode causal | Kernel `_causal_decode_kernel` para un token contra KV-cache, con límites de secuencia y head dimension. | `causal_decode_attention` rechaza fallback estricto. | Falta equivalencia CUDA y no cubre prefill por bloques. |
| Router top-2 | Kernel `_top2_router_kernel` para seleccionar y normalizar dos expertos. | `top2_router` rechaza fallback estricto. | Falta validación CUDA de índices/gates y no hace dispatch ni combinación. |
| Prefill causal | Referencia CPU causal y kernel Triton experimental con softmax online. | `enforce_triton_prefill_contract` continúa bloqueando prefill CUDA con `require_triton=True`. | Falta validación numérica, gradientes, límites y perfilado CUDA antes de habilitar producción. |
| Dispatch/combina MoE | Referencias CPU de capacidad determinista y combinación por experto con gates. | `enforce_triton_moe_dispatch_contract` bloquea CUDA estricta. | Falta pipeline Triton de capacidad, agrupación, dispatch, expert compute y combine. |

## Invariantes que deben preservarse

El kernel futuro no puede cambiar la semántica de selección top-2, la normalización de gates, el cálculo de carga, la telemetría o la pérdida auxiliar. La implementación actual selecciona top-2 para inferencia, mientras entrenamiento usa `torch.topk` a fin de preservar los gradientes del router. Esa separación no debe eliminarse sin una prueba de gradiente CUDA.

El modo `require_triton=True` debe seguir fallando de forma explícita, en vez de degradarse silenciosamente a SDPA o al loop PyTorch. Un fallback autorizado sólo puede etiquetarse como E0 de laboratorio y no puede certificar producción.

## Referencia ejecutable incorporada

Se incorporaron `causal_prefill_reference()`, `causal_prefill_experimental()` y `moe_dispatch_combine_reference()` en `engine/triton_bridge.py`. La primera conserva máscara triangular inclusiva y coincide con SDPA causal en CPU. La segunda contiene un kernel Triton experimental de softmax online, inaccesible desde `Attention.forward` bajo la ruta estricta. La tercera preserva agrupación por experto, acumulación con `index_add`, gates por token/slot, expertos sin tokens y gradientes de tokens/gates.

`engine/test_moe_dispatch_reference.py` cubre equivalencia frente a la ruta legacy, experto vacío, combinación ponderada, gradientes, rechazo de índices fuera de rango y capacidad token-major/slot-major con overflow explícito. `engine/test_triton_prefill_reference.py` cubre equivalencia contra SDPA causal, bloqueo de futuros tokens, errores de forma, fallback experimental sin CUDA y rechazo estricto sin CUDA. La validación CPU no valida CUDA.

**Evidencia local del 22 de agosto de 2026:** `python3 engine/test_triton_prefill_reference.py`, `python3 engine/test_moe_dispatch_reference.py`, `python3 engine/test_triton_bridge.py` y `python3 engine/test_model_budget.py` pasaron en CPU. La búsqueda de `Attention.forward` confirmó que conserva `enforce_triton_prefill_contract` y que sólo llama SDPA en la ruta no estricta. `pnpm test` pasó 5/5. Ninguna de estas comprobaciones ejecutó Triton/CUDA.

> Esta evidencia define el contrato del futuro kernel de dispatch/combina; no valida CUDA ni reduce la brecha de producción descrita en la tabla.
