# Experimento Aethel ARC: refinamiento adaptativo presupuestado

## Hipótesis

**Aethel ARC** (Adaptive Refinement Controller) añade una ruta de refinamiento recurrente sólo a los estados de trabajo cuya dificultad estimada supera un umbral. La ruta está apagada por defecto y, por tanto, no modifica familias ni checkpoints baseline. Cuando se habilita, cada estado seleccionado recibe entre uno y `max_steps` pasos de una `GRUCell` compartida y normalizada. El router empieza en dificultad 0.5 para no quedar inactivo por una inicialización aleatoria; la salida refinada se mezcla con esa dificultad para que la pérdida pueda ajustar el router, mientras el trabajo materializado sigue sujeto al umbral.

El diseño toma el principio de presupuesto dinámico de Mixture-of-Depths y la reutilización de bloques de Mixture-of-Recursions, pero no afirma sus resultados para Aethel. Las fuentes y limitaciones se encuentran en [`research/adaptive_compute_sources.md`](research/adaptive_compute_sources.md).

## Telemetría y criterio de aceptación

El motor registra `selected`, `fraction`, `effective_token_steps`, `mean_difficulty`, el umbral y el máximo de pasos. `effective_token_steps` describe trabajo de la ruta recurrente y **no sustituye** una medición de FLOPs, latencia o VRAM.

La variante se acepta sólo si, frente a la misma receta baseline, mantiene o mejora pérdida de validación, no degrada la salud del router MoE y reduce trabajo medible o mejora calidad bajo el presupuesto acordado. Se rechaza si el router selecciona casi todos los estados sin beneficio o si aumenta coste sin ganancia verificable.

## Activación

```bash
python3 engine/train_nextgen.py \
  --corpus engine/corpora/aethel_repo_corpus.txt \
  --adaptive-refinement-steps 2 \
  --adaptive-refinement-threshold 0.35 \
  --adaptive-compute-penalty 0.001
```

La primera comparación se realiza localmente; el veredicto de rendimiento sólo se toma con GPU CUDA, el preflight Triton/FSDP aprobado y artefactos persistentes.

## Resultado de humo reproducible: 16 de agosto de 2026

La ejecución guardada en [`training/experiments/arc_baseline_comparison_2026-08-16.json`](training/experiments/arc_baseline_comparison_2026-08-16.json) comparó 20 pasos sobre el corpus del repositorio, con la misma semilla, lotes, contexto y configuración base. ARC añadió 25,153 parámetros a esta configuración mínima y seleccionó todos los estados; su pérdida final fue **5.0199** frente a **4.9870** del baseline, el desbalance medio del router fue **0.4781** frente a **0.3820**, y la tasa observada en CPU fue **6,732.9** frente a **7,498.3 tokens/s**. En procesos aislados, el pico RSS fue **362,311,680 bytes** para ARC frente a **361,168,896 bytes** para baseline: una diferencia de **1,142,784 bytes** en esta ejecución local.

> El experimento no supera los criterios de aceptación. ARC queda **desactivado por defecto** y no se propone para la primera corrida GPU. La ruta permanece como hipótesis instrumentada para una búsqueda posterior de umbrales y presupuestos en GPU; estos valores no son una medición de FLOPs, VRAM ni rendimiento GPU.

Las pruebas `engine/test_arc_checkpoint_compatibility.py` también verifican guardado, carga y un paso de reanudación tanto de baseline como de ARC. La carga cruzada se permite sólo con `strict=False`: al pasar baseline a ARC faltan exclusivamente pesos `adaptive_refinement.*`, y al pasar ARC a baseline sobran exclusivamente esos pesos. Así se mantiene explícita la incompatibilidad intencional de la variante, en lugar de ocultarla.

## Protocolo de comparación reproducible

`engine/compare_arc_baseline.py` ejecuta baseline y ARC con el **mismo corpus**, los **mismos lotes preconstruidos**, la misma semilla y configuración compartida. Cada variante corre en un proceso aislado para registrar su **pico RSS real** en Linux. El informe registra pérdida, parámetros, desbalance MoE, fracción seleccionada, pasos de refinamiento, rendimiento observado y RAM; deja explícito que el resultado de humo no demuestra calidad, FLOPs, VRAM ni rendimiento de GPU.

```bash
python3 engine/compare_arc_baseline.py \
  --corpus engine/corpora/aethel_repo_corpus.txt \
  --output training/experiments/arc_baseline_comparison.json
```
