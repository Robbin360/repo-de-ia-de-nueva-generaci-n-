# Matriz de aceptación CUDA para Triton v1

**Estado:** protocolo de validación futuro. No ha sido ejecutado en GPU y no habilita kernels de producción.  
**Propósito:** convertir los contratos CPU y kernels experimentales de Aethel en una decisión binaria, verificable y reversible antes de permitir `require_triton=True` en prefill o dispatch MoE.

## Regla de decisión

Un kernel sólo puede pasar de “experimental” a “habilitable” cuando todas las filas obligatorias de su ruta tienen evidencia emitida por la misma familia de hardware, versión de CUDA, PyTorch, Triton, commit de código y configuración de prueba. Un resultado parcial no permite relajar los bloqueos de `enforce_triton_prefill_contract()` ni `enforce_triton_moe_dispatch_contract()`.

| Estado | Significado | Efecto en producción |
|---|---|---|
| `NOT_RUN` | No existe ejecución CUDA válida. | Bloqueado. |
| `FAILED` | Existe divergencia, error, OOM o límite incumplido. | Bloqueado; conservar artefactos y abrir diagnóstico. |
| `PASSED_EXPERIMENTAL` | Cumple paridad en un rango definido, sin guardas completas. | Sólo laboratorio, no promoción. |
| `PASSED_PRODUCTION` | Cumple todos los criterios, límites y repetición definidos. | Puede proponerse cambio de contrato con revisión humana. |

## Información que debe acompañar a cada corrida

```json
{
  "commit": "<git_sha>",
  "device": "<nombre gpu>",
  "driver": "<versión>",
  "cuda": "<versión>",
  "pytorch": "<versión>",
  "triton": "<versión>",
  "dtype": "float16|bfloat16|float32",
  "seed": 1729,
  "shape": "<B,H,S,D u otra forma>",
  "timestamp_utc": "<ISO-8601>"
}
```

No registrar únicamente una cifra de tokens/s sin forma, dtype, calentamiento, referencia y medición de memoria. Eso no es evidencia comparable.

## Matriz: prefill causal experimental

| Puerta | Referencia | Casos mínimos | Criterio de aceptación | Artefacto |
|---|---|---|---|---|
| Disponibilidad | `torch.cuda.is_available()` y `triton` importable. | GPU objetivo. | CUDA/Triton visibles y versiones registradas. | `environment.json`. |
| Forma | `causal_prefill_reference`. | B∈{1,2}, H∈{1,8}, S∈{1,31,64,127,256,1024,2048}, D∈{32,64,128}. | Salida con misma forma/dtype permitido. | `prefill_shapes.json`. |
| Paridad | SDPA causal / referencia CPU. | Todas las formas admisibles. | Error absoluto y relativo definidos por dtype y almacenados, sin NaN/Inf. | `prefill_numerics.jsonl`. |
| Causalidad | Referencia con valores futuros distinguibles. | Tokens inicial, medio y final. | Ningún token observa valor de posición futura. | `prefill_causality.json`. |
| Gradiente | Autograd PyTorch de referencia. | Formas pequeñas representativas. | Q/K/V gradientes finitos y dentro de tolerancia, o kernel declarado inferencia-only y bloqueado en entrenamiento. | `prefill_gradients.json`. |
| Memoria | `torch.cuda.max_memory_allocated`. | S=256,1024,2048. | Pico medido; sin OOM y sin fuga creciente tras repeticiones. | `prefill_memory.json`. |
| Rendimiento | SDPA con mismo stream/dtype/forma. | Warmup + iteraciones medidas. | Mediana y percentiles registrados; no se exige aceleración a priori. | `prefill_perf.json`. |
| Límites | Validación de entradas. | D no potencia de dos, D>128, S>2048. | Rechazo explícito y legible, no salida incorrecta. | `prefill_limits.json`. |

El kernel actual es inferencia experimental. Hasta completar una estrategia de backward o confirmar que la ruta se limita a inferencia, el entrenamiento debe conservar las rutas PyTorch permitidas por el contrato de laboratorio.

## Matriz: router, capacidad y dispatch/combina MoE

| Puerta | Referencia | Casos mínimos | Criterio de aceptación | Artefacto |
|---|---|---|---|---|
| Router top-2 | `top2_router` PyTorch. | Tokens vacíos, 1, 2, 63, 64, 65, 1024; expertos 2, 8. | Índices idénticos bajo desempate definido; gates normalizados, finitos. | `router_numerics.jsonl`. |
| Capacidad | `moe_capacity_reference`. | Capacidad 1,2; overflow; expertos sin tokens. | Posiciones, `accepted`, cargas y overflow idénticos. | `moe_capacity.json`. |
| Dispatch | `moe_dispatch_combine_reference`. | top-1/top-2, tokens repetidos, expertos vacíos. | Agrupación y orden lógico preservados. | `moe_dispatch.json`. |
| Combinación | Referencia `index_add`. | Gates no uniformes y top-2. | Suma ponderada dentro de tolerancia y sin pérdida de contribuciones. | `moe_combine.json`. |
| Gradiente | Ruta PyTorch de entrenamiento. | Batch pequeño, expertos 2/8. | Gradientes de router/experts/tokens finitos y dentro de tolerancia, si la ruta CUDA se habilita para entrenamiento. | `moe_gradients.json`. |
| Balance | Telemetría existente. | Router uniforme, colapsado y aleatorio. | Cargas/entropía/overflow preservados. | `moe_balance.json`. |
| Rendimiento/memoria | Implementación PyTorch actual. | Shapes de Seed y luego Edge autorizadas. | Métricas comparables; sin promesa previa de mejora. | `moe_perf_memory.json`. |

## Criterios de seguridad y rollback

1. Un fallo numérico, NaN/Inf, fallo de gradiente, violación causal o overflow no contabilizado fuerza `FAILED`.
2. La decisión no se toma sólo con una forma pequeña ni sólo con `float32`; se repite en los dtype de despliegue.
3. Si una ruta pasa inferencia pero no backward, se marca explícitamente `inference_only`; no se usa en entrenamiento.
4. El cambio que habilite el contrato estricto requiere revisión humana, commit separado, pruebas de regresión y checkpoint de código antes de despliegue.
5. Un rollback vuelve a `require_triton=True` bloqueado para la ruta afectada; no altera La Roca, Dataset ni candidatos de Sueño.

## Orden de ejecución al existir GPU autorizada

```text
GPU/entorno → decode + router ya existentes → prefill experimental →
capacidad MoE → dispatch/combina experimental → gradientes →
memoria/rendimiento → revisión humana → posible cambio de contrato
```

No iniciar entrenamiento Seed para justificar una validación de kernel incompleta. Las pruebas de kernel se realizan primero con tensores sintéticos de validación numérica; esos tensores no son Dataset ni métricas de modelo. El entrenamiento de datos reales comienza sólo después de sus autorizaciones y gates correspondientes.
