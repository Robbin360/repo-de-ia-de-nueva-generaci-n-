# Aethel — Resultados de contratos de componentes

**Fecha:** 2026-08-28  
**Entorno:** CPU local; no se usó Kaggle, GPU, corpus ni checkpoint Edge.

## Ejecución

Se ejecutaron directamente, sin `pytest`:

```text
python3 engine/test_model_budget.py
python3 engine/test_memory_reasoning.py
python3 engine/test_nextgen_stability.py
python3 engine/test_sleep_candidate.py
python3 engine/test_compact_telemetry.py
python3 engine/test_moe_dispatch_reference.py
python3 engine/test_resume_contract.py
python3 engine/test_training_resume_e2e.py
python3 engine/test_sleep_orchestrator.py
python3 engine/test_sleep_preflight.py
python3 engine/test_sleep_replay_admission.py
python3 engine/test_sleep_state_machine.py
python3 engine/test_liquid_device_alignment.py
```

## Resultado

| Área | Resultado |
|---|---|
| Presupuesto de modelo | PASS |
| Memoria y trazabilidad de razonamiento | PASS |
| Estabilidad NextGen y replay diverso | PASS |
| Aislamiento de candidato frente a La Roca | PASS |
| Telemetría compacta | PASS |
| Dispatch/combine MoE de referencia CPU | PASS |
| Contrato de reanudación | PASS |
| Equivalencia de entrenamiento/reanudación | PASS |
| Orquestador de sueño | PASS |
| Preflight de sueño | PASS |
| Admisión de replay | PASS |
| Máquina de estados de sueño | PASS |
| Alineación de dispositivo Liquid | SKIPPED_NO_CUDA |

## Interpretación

Los contratos locales CPU de La Roca, El Líquido, memoria, Ciclo de Sueño, MoE, neuromodulación y reanudación pasaron las pruebas disponibles. Esto verifica invariantes de software en casos pequeños; no demuestra calidad del modelo, entrenamiento estable a escala, rendimiento CUDA/Triton ni inteligencia.

La prueba de alineación específica de dispositivo no se ejecutó porque CUDA no estaba disponible en este entorno. Debe repetirse en una GPU antes de declarar la ruta CUDA validada.
