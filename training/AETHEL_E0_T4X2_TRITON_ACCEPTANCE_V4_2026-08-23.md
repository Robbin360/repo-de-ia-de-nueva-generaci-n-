# Evidencia de aceptación CUDA/Triton — Aethel Seed E0 V4

## Alcance y procedencia

Este registro conserva la salida real comunicada desde el notebook privado de Kaggle **Aethel Seed E0 — Offline Preflight** el 23 de agosto de 2026. La ejecución seleccionó el release de código exacto `e0-v4-triton-constexpr-fix` desde `aethel-nextgen-source (4)` y ejecutó únicamente la aceptación CUDA/Triton. No ejecutó entrenamiento, no habilitó las autorizaciones E0 y no creó un checkpoint de modelo.

| Campo | Evidencia observada |
|---|---|
| GPU | 2 × Tesla T4, 15,360 MiB por dispositivo |
| Driver | `580.159.04` |
| CUDA / PyTorch | CUDA `12.8`; PyTorch `2.10.0+cu128` |
| Triton | Importable |
| Release fuente | `e0-v4-triton-constexpr-fix` |
| Estado del ejecutor | `PASSED_EXPERIMENTAL` |
| Autorización de entrenamiento | `AETHEL_RUN_AUTHORIZED=false`; fallback PyTorch `false` |

## Resultados experimentales reales

El prefill causal experimental produjo salidas finitas en las dos formas de prueba: `[1, 1, 31, 32]` y `[1, 2, 64, 64]`. Los errores absolutos máximos registrados frente a la referencia fueron `0.001953125` y `0.001220703125`, respectivamente. Los errores relativos máximos fueron `0.5311004519462585` y `28.496599197387695`; por tanto, **no se usan como evidencia de paridad estricta**.

El control de capacidad del router top-2 mantuvo posiciones válidas, índices iguales a la referencia y puertas normalizadas. Registró `122` asignaciones aceptadas, `8` descartadas por capacidad y un error absoluto máximo de compuertas de `8.940696716308594e-08`.

> `PASSED_EXPERIMENTAL` significa que la ruta experimental compiló y produjo evidencia básica en esa configuración. No implica que las rutas Triton estrictas estén aceptadas ni que E0 esté autorizado.

## Límites que permanecen bloqueados

La aceptación no validó gradientes del prefill, un kernel Triton de dispatch/combine MoE, límites completos de memoria/rendimiento ni la matriz CUDA estricta. En consecuencia, el entrenamiento Seed E0, si recibe autorización posterior, debe seguir declarando explícitamente el fallback PyTorch como **experimental y no promocionable**. Este registro tampoco acredita un modelo entrenado, métricas de holdout, generación ni checkpoint.

## Archivos de evidencia en Kaggle

La celda V4 indicó la escritura de los siguientes archivos de sesión en el almacenamiento de trabajo efímero:

```text
/kaggle/working/aethel-e0-cuda-triton-acceptance/output/triton_cuda_acceptance.json
/kaggle/working/aethel-e0-cuda-triton-acceptance/output/acceptance_session.json
```

Antes de cerrar la sesión, estos archivos deben descargarse o incluirse explícitamente en una versión de notebook autorizada si se requiere su conservación externa. No se incorporan al repositorio como sustituto de la salida original.
