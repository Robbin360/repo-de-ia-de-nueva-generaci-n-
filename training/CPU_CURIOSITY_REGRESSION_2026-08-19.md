# Regresión CPU de curiosidad — 2026-08-19

**Alcance:** validación local, sin GPU, sin red, sin Dataset de Kaggle y sin entrenamiento.

## Propósito

Esta ejecución confirma que la incorporación del controlador de curiosidad y su ledger líquido no rompió contratos ya existentes del núcleo Aethel NextGen. No demuestra calidad lingüística, razonamiento humano, aprendizaje continuo ni rendimiento GPU.

## Resultado de ejecución

Todos los comandos siguientes finalizaron correctamente con `PYTHONPATH=engine python3`:

| Prueba | Evidencia comprobada |
|---|---|
| `test_memory_reasoning.py` | Inmutabilidad de La Roca durante observación, trazabilidad líquida y ledger de curiosidad no elegible para Sueño. |
| `test_lora_adapters.py` | Adaptadores LoRA aislados con la base congelada y fracción entrenable medida. |
| `test_adaptive_refinement.py` | Refinamiento adaptativo bajo presupuesto. |
| `test_arc_checkpoint_compatibility.py` | Compatibilidad y reanudación explícita de checkpoints ARC/baseline. |
| `test_arc_comparison.py` | Comparación ARC–baseline reproducible. |
| `test_checkpoint_resilience.py` | Resiliencia de checkpoint local. |
| `test_kv_cache.py` | Caché KV autoregresiva. |
| `test_model_budget.py` | Presupuesto de parámetros calculado. |
| `test_nextgen_stability.py` | Estabilidad del núcleo NextGen. |
| `test_training_schedule.py` | Contrato de calendario de entrenamiento. |

## Límites explícitos

La curiosidad vigente es **funcional y acotada**: puede clasificar señales y registrar una propuesta local. Sus eventos se escriben con `eligible_for_sleep: false` y `external_action_enabled: false`. Por tanto, esta prueba no autoriza ingestión de datos, acciones externas, cambio de objetivos, sueño, ajuste LoRA, promoción ni modificación de La Roca.

Las pruebas CUDA/FSDP/Triton siguen fuera de este registro y requieren hardware GPU real conforme a sus guardas de preflight.
