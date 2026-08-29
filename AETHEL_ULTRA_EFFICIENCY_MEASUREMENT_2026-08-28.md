# Aethel — medición de eficiencia disponible

**Fecha:** 2026-08-28  
**Estado:** evidencia local CPU y presupuestos analíticos; no es una validación CUDA ni un coste de producción.

## Conclusión

La evidencia disponible no demuestra todavía «ultra-eficiencia» en sentido general. Sí permite medir tres cosas de forma reproducible: parámetros totales, una aproximación de parámetros activos en la ruta MoE y rendimiento CPU de una comparación emparejada. El coste monetario por token no se calcula porque no se ha fijado un proveedor, hardware, precio ni utilización real.

## Presupuestos analíticos

El reporte `engine/report_model_budget.py` cuenta formas de tensores sin instanciar los pesos grandes. La cifra de parámetros activos es una aproximación de la ruta que usa `active_experts` por capa; embeddings, atención y módulos NextGen permanecen activos.

| Preset | Parámetros totales | Parámetros activos aprox. | Fracción de expertos | Estado AdamW estimado* |
|---|---:|---:|---:|---:|
| `pilot-100m` | 97,155,076 | 40,531,972 | 2/8 | 1.09 GiB |
| `research-300m` | 344,337,412 | 117,844,996 | 2/8 | 3.85 GiB |
| `scale-1b` | 1,192,677,380 | 362,205,188 | 2/8 | 13.33 GiB |
| `adaptive-research-300m` | 347,883,269 | 121,390,853 | 2/8 | 3.89 GiB |

\* Estimación estática de pesos BF16, gradientes BF16 y dos estados Adam FP32; no incluye activaciones, buffers, comunicaciones ni fragmentación.

## Comparación CPU emparejada

Se ejecutó `engine/compare_arc_baseline.py` con la misma semilla, corpus sintético de prueba, batch, longitud y 128 tokens. La prueba es local y reproducible, pero deliberadamente pequeña.

| Métrica | Baseline | ARC | Diferencia ARC − baseline |
|---|---:|---:|---:|
| Parámetros | 167,428 | 192,581 | +25,153 |
| Pérdida media | 5.4572 | 5.4132 | -0.0440 |
| Pérdida final | 5.3373 | 5.3113 | -0.0260 |
| Desequilibrio medio del router | 0.1953 | 0.5000 | +0.3047 |
| Tokens/s CPU | 1,964.95 | 1,655.80 | -309.14 |
| RSS pico | 357,404,672 B | 360,255,488 B | +2,850,816 B |

ARC bajó ligeramente la pérdida, pero añadió parámetros, redujo el throughput CPU aproximadamente 15.7% y empeoró el desequilibrio del router en esta muestra. No se activa por defecto a partir de este resultado.

## Qué falta para una afirmación de ultra-eficiencia

| Métrica | Estado | Requisito |
|---|---|---|
| Parámetros totales/activos | Disponible analíticamente | Confirmar con trazas de routing en una corrida real |
| VRAM | Sólo estimación | Medir pico/reservada en CUDA con el kernel y configuración finales |
| Tokens/s | Disponible sólo en CPU para el baseline | Medir prefill y decode en hardware comparable, con warm-up y varias semillas |
| Coste por token | No disponible | Fijar proveedor, precio/hora, batch, utilización y tokens/s sostenidos |
| Calidad por FLOP | No disponible | Ejecutar la batería EN/ES, matemáticas y razonamiento frente a baseline |
| Eficiencia energética | No disponible | Medir energía del dispositivo durante una ventana estable |

## Regla de promoción

Una optimización sólo se promoverá si conserva o mejora la calidad en el conjunto de evaluación, reduce el coste definido por token o mantiene la calidad con menos latencia/VRAM, y no empeora estabilidad del router, retención ni seguridad de memoria. Una mejora aislada de pérdida no basta.

## Reproducibilidad

```bash
python3 engine/report_model_budget.py --preset all
python3 engine/compare_arc_baseline.py
```

Estos comandos no entrenan un modelo grande, no cargan checkpoints Edge y no requieren GPU. Los resultados no deben extrapolarse automáticamente a CUDA, producción, bilingüismo o razonamiento.

## Referencias internas

- `AETHEL_ARC_BASELINE_RESULT_2026-08-28.md`
- `engine/report_model_budget.py`
- `AETHEL_DYNAMIC_CAPACITY_EVALUATION_PLAN.md`
- `AETHEL_EFFICIENCY_INTELLIGENCE_ROADMAP.md`
