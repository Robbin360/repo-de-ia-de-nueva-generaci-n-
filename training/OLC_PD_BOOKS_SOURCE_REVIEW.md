# Revisión de fuente inglesa: Open License Corpus

Fecha de verificación: 2026-08-17

## Fuente

- Dataset: `kernelmachine/open-license-corpus`
- Configuración propuesta: `pd_books`
- Split: `train`
- Campo de texto: `text`
- Revisión fijada verificada por API de Hugging Face: `384d5e19d19361803630ce4d382604267d3951d2`
- URL: https://huggingface.co/datasets/kernelmachine/open-license-corpus
- Licencia declarada para el subconjunto `pd_books`: dominio público, según la ficha del dataset.

## Compatibilidad

El preparador existente soporta `hf_rows_api` y el esquema de filas con columna `text`, por lo que no requiere un adaptador de formato para esta fuente. El subconjunto `pd_books` figura con aproximadamente 13.000 filas en la ficha pública, suficiente para aportar hasta 10.000 documentos al piloto si las peticiones remotas y los filtros los aceptan.

## Decisión provisional

Añadir la fuente con `document_limit` 10000 y `minimum_documents` 7000. Mantener la puerta global de 14.000 documentos ingleses; no bajar el umbral automáticamente. La entrada debe conservar `revision` con el SHA anterior y la procedencia del repositorio.

## Riesgos

La fuente contiene libros de dominio público y puede tener textos largos o distribución temática distinta a Wikipedia. Deben mantenerse los hashes, el manifiesto, el subconjunto exacto y los filtros de longitud. La ejecución debe detenerse si no alcanza los mínimos o si falla la verificación de procedencia.

## Fuentes consultadas

1. https://huggingface.co/datasets/kernelmachine/open-license-corpus
2. https://huggingface.co/api/datasets/kernelmachine/open-license-corpus
3. https://huggingface.co/api/datasets/kernelmachine/open-license-corpus/commits/main
