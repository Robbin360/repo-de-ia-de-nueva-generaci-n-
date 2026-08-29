# Aethel — índice técnico del repositorio

Este índice organiza la documentación necesaria para entender, ejecutar y auditar Aethel. La fuente de verdad para el comportamiento implementado es el código y sus pruebas; las especificaciones de investigación describen decisiones futuras y deben validarse con experimentos.

## Lectura recomendada

| Orden | Documento | Para qué sirve |
|---:|---|---|
| 1 | [`AETHEL_SYSTEM_ARCHITECTURE_V1.md`](./AETHEL_SYSTEM_ARCHITECTURE_V1.md) | Explica el sistema completo, el flujo de inferencia y el papel de cada módulo. |
| 2 | [`AETHEL_BASE_CAPABILITY_SPEC.md`](./AETHEL_BASE_CAPABILITY_SPEC.md) | Define el nivel mínimo de conversación EN/ES, razonamiento y matemáticas antes de escalar. |
| 3 | [`AETHEL_BASE_TRAINING_PLAN_100M_300M.md`](./AETHEL_BASE_TRAINING_PLAN_100M_300M.md) | Describe la campaña base 100M–300M con Kaggle/T4, sesiones y puertas. |
| 4 | [`AETHEL_PRO_SPEC.md`](./AETHEL_PRO_SPEC.md) | Presenta el diseño Pro y sus cálculos de parámetros/VRAM, sin confundirlo con pesos entrenados. |
| 5 | [`training/AETHEL_TRAINING_RESUME_CONTRACT_V1.md`](./training/AETHEL_TRAINING_RESUME_CONTRACT_V1.md) | Fija el contrato de checkpoints y reanudación fiel. |
| 6 | [`training/AETHEL_COGNITIVE_EXPERIMENT_CONTRACT_V1.md`](./training/AETHEL_COGNITIVE_EXPERIMENT_CONTRACT_V1.md) | Define cómo medir módulos cognitivos sin claims no verificables. |
| 7 | [`AETHEL_EXTERNAL_BLOCKERS_RUNBOOK_V1.md`](./AETHEL_EXTERNAL_BLOCKERS_RUNBOOK_V1.md) | Explica los bloqueos de Kaggle/GPU/FSDP/Triton/Rust/Mojo y la reanudación segura. |

## Código ejecutable

| Dominio | Punto de entrada |
|---|---|
| Modelo Transformer | [`engine/aethel_model.py`](./engine/aethel_model.py) |
| Ensamblaje NextGen | [`engine/aethel_nextgen.py`](./engine/aethel_nextgen.py) |
| Entrenamiento GPU | [`engine/train_aethel_gpu.py`](./engine/train_aethel_gpu.py) |
| Entrenamiento NextGen | [`engine/train_nextgen.py`](./engine/train_nextgen.py) |
| Preparación bilingüe | [`engine/prepare_bilingual_corpus.py`](./engine/prepare_bilingual_corpus.py) |
| Presupuesto | [`engine/report_model_budget.py`](./engine/report_model_budget.py) |
| Reanudación | [`engine/aethel_resume.py`](./engine/aethel_resume.py) |
| Evaluación | [`engine/evaluate_nextgen.py`](./engine/evaluate_nextgen.py), [`engine/evaluate_benchmarks.py`](./engine/evaluate_benchmarks.py) |
| Router | [`engine/router_health.py`](./engine/router_health.py), [`engine/router_assignment_health.py`](./engine/router_assignment_health.py) |
| Triton/CUDA | [`engine/triton_bridge.py`](./engine/triton_bridge.py) |

## Mapa de evidencia

Los resultados de experimentos se conservan en `training/` y las fuentes metodológicas en `research/`. Un archivo de resultado debe indicar semilla, datos, configuración, hardware, pasos, métricas, hashes y limitaciones. Las pruebas CPU verifican contratos y formas; no sustituyen una medición de calidad o throughput en T4.

## Regla de promoción

Una mejora entra en la rama estable sólo después de una comparación contra baseline con los mismos datos, semilla y presupuesto. Debe mejorar la métrica objetivo sin superar el límite de regresión en EN/ES, razonamiento, estabilidad del router, memoria y coste. Si no pasa, se conserva como evidencia experimental y no se activa por defecto.

## Estados del proyecto

| Estado | Interpretación |
|---|---|
| Implementado | Existe código ejecutable y pruebas o contratos asociados. |
| Experimental | Existe una implementación o sonda, pero falta validación completa en hardware objetivo. |
| Diseñado | Está especificado, pero no hay evidencia de ejecución suficiente. |
| Bloqueado | Requiere GPU, datos, cuenta, cuota o autorización externa. |
| No afirmado | No debe presentarse como capacidad de Aethel. |
