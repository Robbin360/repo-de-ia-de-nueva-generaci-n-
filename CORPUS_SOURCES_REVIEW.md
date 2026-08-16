# Revisión de fuentes para el corpus de Aethel

Este documento registra la revisión previa a cualquier descarga. **No activa ni descarga ninguna fuente**: cada incorporación exige revisión jurídica y aprobación explícita del responsable del entrenamiento.

| Fuente | Idiomas relevantes | Escala declarada | Licencia o condición verificada | Decisión inicial |
|---|---:|---:|---|---|
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Inglés | 1.53 mil millones de filas en la ficha | `odc-by` | Elegible solo como fuente de inglés y con atribución; empezar por una muestra aprobada. |
| [RedPajama-Data-V2](https://github.com/togethercomputer/RedPajama-Data) | Inglés, alemán, francés, italiano y español | 30.4T tokens deduplicados en su porción anotada `head_middle`; 2.8T en español | El repositorio del pipeline usa `Apache-2.0`; el contenido rastreado debe revisarse por fuente y jurisdicción | Candidato para la fase multilingüe a escala, con filtros, exclusiones y trazabilidad de procedencia. |
| [OSCAR 23.01](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301) | 151 idiomas | Más de 1T según la ficha | Metadatos/annotaciones `CC0-1.0`; el contenido derivado de Common Crawl está restringido y el acceso aparece temporalmente suspendido | No incorporar: fuente bloqueada hasta resolver condiciones de acceso, licencia y cumplimiento aplicable. |

## Controles obligatorios

1. Mantener una lista exacta de shards, revisiones, hashes, idioma, licencia y fecha de obtención.
2. Ejecutar deduplicación exacta y aproximada antes de tokenizar; conservar el informe de descartes.
3. Filtrar datos personales, contenido ilícito, spam, baja calidad y duplicados contra evaluación.
4. Mantener sets de validación y benchmarks fuera del corpus de entrenamiento.
5. Publicar resultados solamente con la configuración, semilla, manifiesto y checkpoints que los reproduzcan.

## Fuentes

1. Hugging Face, FineWeb-Edu dataset card: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
2. Together Computer, RedPajama-Data-V2: https://github.com/togethercomputer/RedPajama-Data
3. OSCAR 23.01 dataset card: https://huggingface.co/datasets/oscar-corpus/OSCAR-2301
