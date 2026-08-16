# Protocolo de benchmarks de Aethel

Los paneles de MMLU, HumanEval y GSM8K permanecen vacíos hasta que exista un informe creado por `engine/evaluate_benchmarks.py`. El protocolo evita tanto puntuaciones de marketing como ejecuciones no reproducibles.

| Benchmark | Entrada de referencia | Predicción requerida | Medición |
|---|---|---|---|
| MMLU | JSONL con `task=mmlu`, `id`, `answer` (A–D) | JSONL con el mismo `id` y `answer` | Exactitud de elección múltiple |
| GSM8K | JSONL con `task=gsm8k`, `id`, `answer` | JSONL con la respuesta final de Aethel | Exactitud numérica final |
| HumanEval | JSONL con `task=humaneval`, `id` | Resultado de un sandbox aislado: `pass: true/false` | `pass@1` sobre ejecuciones seguras |

## Ejecución

```bash
python3 engine/evaluate_benchmarks.py \
  --reference /data/evals/reference.jsonl \
  --predictions /data/evals/aethel_predictions.jsonl \
  --output /data/evals/aethel_benchmark_report.json
```

No se ejecuta código generado por el modelo dentro de este script. Para HumanEval, use un entorno aislado con límites de CPU, memoria, red, tiempo y sistema de archivos; almacene solo el resultado de esa ejecución.

Cada informe debe conservar la revisión del modelo, hash del checkpoint, tokenizador, prompt, semilla, versión de dataset y fecha de ejecución.
