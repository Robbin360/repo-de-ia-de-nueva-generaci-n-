# Especificación escalable de Aethel

> **Principio:** Aethel debe escalar por evidencia, no por proclamación. Cada tamaño solo avanza si supera los controles de pérdida, retención, salud del router, calidad del corpus y coste por token de la etapa anterior.

## Núcleo de la nueva arquitectura

La base es un Transformer decoder causal con **RoPE** para codificación posicional y **GQA** para reducir la presión de la caché de claves/valores. Cada bloque combina atención eficiente con un MoE disperso: se activan pocos expertos por token, mientras la carga de todos los expertos se monitoriza y corrige lentamente mediante sesgos de routing persistentes. Esto conserva capacidad condicional sin activar toda la red en cada paso.

Los cinco pilares actúan sobre ese núcleo. **La Roca** preserva la trayectoria parametrizada estable mediante regularización de replay. **El Líquido** guarda trazas plásticas versionadas que responden a sorpresa y neuromodulación. El **Espacio de Trabajo Global** mezcla contexto actual con recuperación episódica. El **Ciclo de Sueño** conserva experiencias salientes, selecciona replay diverso y devuelve una pérdida real al entrenamiento. La **Neuromodulación** regula la fuerza de adaptación de las trazas en vez de modificar pesos de forma opaca.

## Familias iniciales

| Familia | Uso | Parámetros totales medidos | Estado AdamW estimado | Contexto | Regla operativa |
|---|---|---:|---:|---|
| `pilot-100m` | Primera sesión gratuita y validación de estabilidad | 97.16 M | 1.09 GiB | 1,024 | Una P100/T4 de 16 GB con FP16, microbatch 2 y acumulación 16 es el punto de partida |
| `research-300m` | Investigación con GPU persistente | 344.34 M | 3.85 GiB | 2,048 | Requiere al menos 24 GB, BF16 si la GPU lo permite, microbatch conservador y evaluación congelada |
| `scale-1b` | Escalado multi-GPU | 1,192.68 M | 13.33 GiB | 2,048 | Requiere 48 GB o sharding de parámetros/optimizador; DDP por sí solo replica el estado completo |

El comando siguiente calcula parámetros y memoria de optimizador de los presets desde el código, evitando estimaciones manuales:

```bash
python engine/report_model_budget.py --preset all
```

La columna de AdamW comprende pesos BF16, gradientes BF16 y los dos estados FP32; no incluye activaciones, caché KV, memoria temporal de MoE ni la sobrecarga del runtime. Por ello no se debe elegir hardware usando solo esta cifra. El runner soporta **DDP** para replicación de datos; la familia `scale-1b` permanece bloqueada hasta incorporar FSDP o ZeRO, que distribuyan estados y parámetros en lugar de replicarlos.

El runner ahora incorpora **FSDP** (`--strategy fsdp`) para repartir parámetros, gradientes y estados del optimizador entre varias GPU, guardando un checkpoint completo solo en rango 0 y reconstruyéndolo al reanudar. El preset `scale-1b` debe invocarse con `torchrun`, al menos dos GPU CUDA, una mezcla de datos aprobada y FSDP; la estrategia DDP se mantiene para investigación de hasta 300M parámetros cuando la VRAM permita replicar el modelo completo.

## Presupuesto de tokens y puertas de avance

| Familia | Presupuesto de tokens | Currículo | Puertas antes de avanzar |
|---|---:|---|---|
| `pilot-100m` | 10 M | 1 M de calentamiento; 8.5 M de texto educativo/técnico y español curado; 0.5 M de consolidación con replay | Pérdida holdout no divergente, `router_health.healthy=true`, replay registrado, exportación verificable y coste/token medido |
| `research-300m` | 1 B | 5% calentamiento, 70% conocimiento educativo multilingüe, 15% razonamiento/código y 10% consolidación | Repetir holdout, MMLU/GSM8K/HumanEval con predicciones reales, auditoría de contaminación y revisión de olvido |
| `scale-1b` | 20 B como mínimo inicial | Currículo por dominios con mezcla congelada por versión y evaluación intermedia cada 250 M tokens | FSDP/ZeRO activo, métricas de coste, pruebas de seguridad y evaluación externa reproducible; no iniciar sin estos controles |

Estos presupuestos son **hitos de investigación**, no promesas de calidad frontier. El valor por token depende de la mezcla, el tokenizador, la estabilidad del router, el contexto efectivo y la ejecución de evaluaciones no contaminadas.

### Evidencia inicial de estabilidad

La corrida reproducible `engine/run_quality_experiment.py` se ejecutó durante 12 pasos sobre `input.txt` del repositorio original. La pérdida pasó de **6.2967** a **6.0911**, hubo **6** eventos de replay, el desequilibrio máximo del router fue **0.25** y todos los eventos cumplieron el umbral de salud configurado. El informe versionado en `training/experiments/quality_experiment_2026-08-16.json` es evidencia de integración y estabilidad inicial; no mide capacidad general, razonamiento ni desempeño frente a benchmarks.

## Criterios de calidad obligatorios

| Señal | Qué mide | Criterio inicial |
|---|---|---|
| `loss` y pérdida holdout | Ajuste y generalización local | La holdout no puede empeorar de forma sostenida mientras baja la train loss |
| `router_health` | Desbalance y entropía del MoE | Debe mantenerse saludable después del calentamiento configurado |
| `replay_loss` | Retención de recuerdos consolidados | Debe registrarse cuando hay episodios disponibles; no puede ser omitida silenciosamente |
| versiones de El Líquido | Adaptación plástica auditable | Cada consolidación debe poder rastrearse y restaurarse |
| MMLU, GSM8K y HumanEval | Conocimiento, razonamiento y código | Solo se publican cuando las predicciones provienen de un modelo y dataset reales |

## Límite científico

Esta arquitectura puede investigar **especialización condicional, adaptación controlada y retención medible**. No demuestra ni garantiza consciencia, un cerebro humano real, AGI ni superioridad sobre modelos de frontera. Alcanzar resultados competitivos requiere muchas más iteraciones, datos de alta calidad, computación sostenida, control de contaminación de benchmarks y evaluación externa reproducible.
