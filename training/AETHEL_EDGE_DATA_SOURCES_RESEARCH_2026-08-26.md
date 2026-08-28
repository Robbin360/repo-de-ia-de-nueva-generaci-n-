# Investigación inicial de fuentes de datos para Aethel Edge

Este documento registra fuentes candidatas; **no autoriza descargas, creación de datasets Kaggle ni su incorporación al entrenamiento**. Cada fuente deberá volver a verificarse, fijarse por revisión y pasar filtros de calidad, deduplicación y separación de holdout antes de su uso.

| Fuente candidata | Rol posible | Licencia expuesta | Decisión provisional |
|---|---|---|---|
| [FineWeb2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) | Preentrenamiento general EN/ES con muestreo pequeño y trazable | ODC-BY | Candidata para texto general; seleccionar sólo fragmentos EN/ES, filtrar calidad y excluir cualquier texto de evaluación. |
| [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Matemática supervisada en inglés | Apache-2.0 | Candidata para matemática; no supone competencia matemática en español sin una fuente española separada y evaluación propia. |
| [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | Evaluación matemática y posible entrenamiento con control estricto | MIT | Reservar primero como benchmark/holdout; no mezclar en entrenamiento si se usará para medir progreso. |
| [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | Matemática a gran escala | CC-BY-4.0 | Candidata condicionada a revisión de calidad, atribución y submuestreo; no descargar sin aprobación específica. |
| [HPLT2.0_cleaned](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned) | Texto multilingüe filtrado, incluido español | CC0 para el empaquetado de texto | Candidata para una fracción ES trazable; confirmar el alcance de licencia de cada componente antes de mezclar. |
| [Common Corpus](https://huggingface.co/datasets/PleIAs/common_corpus) | Texto multilingüe de fuente abierta, incluido español | Revisar por subset antes de uso | Candidata para preentrenamiento ético; someter a muestreo, filtros y documentación de procedencia. |
| [Nemotron Post-Training Dataset v2](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2) | Razonamiento e instrucción multilingüe, incluido español | CC-BY-4.0 | Candidata para una fracción pequeña de postentrenamiento; verificar la procedencia de cada subconjunto y no asumir que sustituye el preentrenamiento general. |
| [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Razonamiento matemático supervisado | Apache-2.0 | Candidata alternativa a NuminaMath; elegir una fuente y reservar benchmarks independientes para evitar contaminación de evaluación. |

## Criterios no negociables

El corpus Edge debe tener manifiesto de procedencia y licencia, hashes de archivos, proporción EN/ES declarada, deduplicación dentro y entre fuentes, filtros de longitud/calidad, exclusión del holdout y una mezcla explícita por objetivo. Los datos de evaluación no se mezclarán con el entrenamiento.

## Revisiones verificadas el 26 de agosto de 2026

| Fuente | Revisión fijable | Estado de acceso | Observación operativa |
|---|---|---|---|
| FineWeb2 | `af9c13333eb981300149d5ca60a8e9d659b276b9` | Pública | Elegir configuraciones EN/ES concretas antes de descargar. |
| HPLT2.0_cleaned | `d1324a5283f762ee62c2a5c81de08fc6450ea540` | Pública | El catálogo expone EN y ES; emplear sólo una muestra declarada. |
| OpenR1-Math-220k | `e4e141ec9dea9f8326f4d347be56105859b2bd68` | Pública | Inglés, Apache-2.0; reservar MGSM/GSM8K como evaluación separada. |
| Nemotron Post-Training v2 | `5c89e01dd720ae0f4058445ed49c5fb68a03c76e` | Gated automático | No incorporar hasta que el acceso y el subconjunto se confirmen explícitamente. |

## Esquema verificado de la fuente matemática

La API pública de filas de [OpenR1-Math-220k](https://datasets-server.huggingface.co/rows?dataset=open-r1%2FOpenR1-Math-220k&config=default&split=train&offset=0&length=1), consultada el 26 de agosto de 2026, expone `problem`, `solution`, `answer`, `problem_type`, `question_type`, `source`, `uuid`, `is_reasoning_complete`, `generations`, `correctness_math_verify`, `correctness_llama`, `finish_reasons`, `correctness_count` y `messages`. El adaptador Edge debe formar texto sólo con `problem`, `solution` y `answer`, y filtrar por señal de completitud/verificación cuando esté disponible; no debe incorporar las generaciones auxiliares como si fueran referencias curadas.

## Fuentes

1. [HuggingFaceFW, "FineWeb2"](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2).
2. [AI-MO, "NuminaMath-CoT"](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT).
3. [OpenAI, "GSM8K"](https://huggingface.co/datasets/openai/gsm8k).
4. [NVIDIA, "OpenMathInstruct-2"](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2).
5. [HPLT, "HPLT2.0_cleaned"](https://huggingface.co/datasets/HPLT/HPLT2.0_cleaned).
6. [PleIAs, "Common Corpus"](https://huggingface.co/datasets/PleIAs/common_corpus).
7. [NVIDIA, "Nemotron Post-Training Dataset v2"](https://huggingface.co/datasets/nvidia/Nemotron-Post-Training-Dataset-v2).
8. [Open R1, "OpenR1-Math-220k"](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k).

## Corrección posterior a la primera construcción autorizada

El 26 de agosto de 2026, la CELDA 3 falló antes de construir el corpus porque el manifiesto solicitaba `eng_Latn` en `HuggingFaceFW/fineweb-2`. La ficha y el repositorio oficial de FineWeb2 explican que FineWeb2 se construye sobre la porción no inglesa del FineWeb original; la lista de configuraciones publicada no contiene `eng_Latn`. Por tanto, FineWeb2 se conserva como fuente española `spa_Latn`, pero no puede suministrar la fuente inglesa. [9] [10]

La ficha oficial de `HuggingFaceFW/fineweb` identifica ese dataset como inglés y expone una configuración `sample-10BT`, con 14,9 millones de filas. Es suficientemente grande para el límite propuesto de 100.000 documentos sin solicitar el subconjunto global completo. La API oficial del dataset confirma que `sample-10BT` está disponible y publica la revisión `9bb295ddab0e05d785b879661af7260fed5140fc`. Tras la autorización explícita recibida, se sustituirá sólo `fineweb2-en-proposed` por `fineweb-en-sample-10bt-proposed`, con `dataset: HuggingFaceFW/fineweb`, `config: sample-10BT`, `split: train`, columna `text`, esa revisión fija y licencia ODC-BY. [10] [11]

9. [FineWeb2: repositorio oficial y nota de datos no ingleses](https://github.com/huggingface/fineweb-2), consultado el 26 de agosto de 2026.
10. [FineWeb: ficha oficial y configuraciones disponibles](https://huggingface.co/datasets/HuggingFaceFW/fineweb), consultada el 26 de agosto de 2026.
11. [FineWeb: API oficial de configuración](https://huggingface.co/api/datasets/HuggingFaceFW/fineweb), consultada el 26 de agosto de 2026.
