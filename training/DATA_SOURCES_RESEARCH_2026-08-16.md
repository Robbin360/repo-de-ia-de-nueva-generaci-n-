# Investigación de fuentes para Aethel NextGen

Este registro preserva fuentes externas revisadas el 16 de agosto de 2026. Es una preselección técnica, no una aprobación jurídica ni una afirmación de que los datos ya están descargados o listos para entrenar.

| Fuente | Uso propuesto | Evidencia y limitación operativa |
|---|---|---|
| Wikipedia `20231101.en` y `20231101.es` | Base enciclopédica y exposición nativa equilibrada a ambos idiomas. | La revisión de dataset publicada por Wikimedia en Hugging Face es `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`; la tarjeta declara los subconjuntos `20231101.en` y `20231101.es`, con licencia CC BY-SA 3.0 y GFDL. Se fijará este commit en el manifiesto y se guardarán hashes de los shards preparados. [1] [2] [3] [9] |
| Diccionarios estructurados Kaikki/Wiktextract en español e inglés | Vocabulario, definiciones, lemas y ejemplos, con separación por idioma de origen. | Kaikki publica JSONL extraído con Wiktextract y documenta la fecha del dump y commits de extractores. El archivo comprimido inglés es de 2.6 GB y el español de 94.7 MB; se seleccionará un subconjunto determinista de definiciones, con fecha, URL y SHA-256 propios. El contenido derivado conserva las condiciones de atribución/compartir igual de Wiktionary. [1] [2] [13] [14] [15] |
| Pares es–en de Tatoeba | Complemento paralelo pequeño y de alta trazabilidad para instrucciones de traducción y contraste de idiomas. | La tarjeta `Helsinki-NLP/tatoeba` declara CC BY 2.0 y fija la revisión `00476f0f7e251c934e14f6e88c42a15e1b67c5a5`. Tatoeba publica exportaciones semanales bajo CC BY 2.0 FR y CC0 para parte de las oraciones. La descarga se realizará con adaptador explícito que conserve fecha, hash y licencia; no se usará audio. [4] [10] |
| ParaCrawl en–es release 9 | Fuente paralela auxiliar para una fase posterior, solo tras filtrado de calidad y revisión de sus condiciones. | La fuente declara una versión 9 con limpieza neuronal y ofrece 269,394,967 pares en–es en su variante TXT, pero su tamaño excede una primera sesión gratuita. El piloto debe usar una muestra determinista y documentar la procedencia/condiciones antes de habilitarla. [5] |
| MGSM-Rev2 (español e inglés) | Evaluación retenida de razonamiento matemático multietapa; no se mezcla con el preentrenamiento ni con el ajuste. | El repositorio documenta 250 problemas traducidos al español y otros nueve idiomas, con licencia CC BY-SA 4.0. La copia de evaluación se fija en la revisión `b2f13d426afe3be8d69a7e739b36724db8b66bbc`. Se conserva como conjunto de evaluación íntegro y separado. [6] [11] |
| Belebele (spa_Latn, eng_Latn) | Evaluación retenida de comprensión lectora comparable entre español e inglés; no usar para entrenar ni validar. | El proyecto contiene 900 preguntas por variante lingüística, está diseñado como test y advierte expresamente que no debe utilizarse para entrenamiento ni validación. La copia de evaluación se fija en la revisión `7899cdfa4e1e0d733fd77c848e2c273cb1d32be2`, bajo CC BY-SA 4.0. [7] [12] |
| FLORES-200 (spa_Latn ↔ eng_Latn) | Evaluación retenida de traducción; no se mezcla con el corpus ni el tokenizador. | Documenta 3,001 oraciones en dev, devtest y test, y el repositorio recomienda versiones más nuevas en Open Language Data. Se debe fijar una revisión concreta antes de evaluación. [8] |

> **Decisión provisional para el piloto:** usar una mezcla controlada de Wikipedia y Wiktionary en español e inglés, más un subconjunto licenciado y trazable de Tatoeba. ParaCrawl queda fuera de la primera ejecución hasta fijar versión, licencia aplicable y un método de muestreo/filtrado revisado. Esta decisión privilegia procedencia y control de calidad sobre volumen bruto.

Para medir capacidades sin contaminar el aprendizaje, el piloto reservará MGSM-Rev2 para razonamiento matemático en español e inglés, Belebele para comprensión en ambos idiomas y FLORES-200 para traducción español–inglés. No se incluirán sus ítems de prueba en el tokenizer, en el corpus de preentrenamiento ni en el ajuste. [6] [7] [8]

La licencia CC BY-SA 4.0 permite compartir y adaptar, incluso comercialmente, pero exige atribución, enlace a licencia, aviso de cambios y compartir las adaptaciones bajo licencia equivalente; por tanto, una eventual distribución de un dataset derivado o modelo entrenado requiere revisión específica. [3]

La preparación de evaluación copiará únicamente la partición `test` de MGSM en `en` y `es`: sus 250 ítems de prueba por idioma incluyen `question` y `answer_number`, mientras que los ocho ejemplos `train` se conservarán solo como contexto de *few-shot*, separados de cualquier texto del corpus. Belebele se copiará solo desde sus configuraciones `eng_Latn` y `spa_Latn`, ambas con 900 preguntas de `test`, usando `flores_passage`, opciones de respuesta y `correct_answer_num` exclusivamente en archivos de referencia. [11] [12]

La revisión `b04c8d1ceb2f5cd4588862100d08de323dccfbaa` de `wikimedia/wikipedia` expone los subconjuntos `20231101.en` y `20231101.es` como archivos Parquet grandes; además, el servidor de filas público limita las consultas y respondió HTTP 429 al intentar una extracción amplia sin autenticación. Por ello, el piloto ajusta las cuotas a 5.000 documentos de Wikipedia y 10.000 entradas de Kaikki por idioma, aplica espera entre páginas y deja para una fase posterior con infraestructura aprobada la mezcla enciclopédica de mayor volumen. El artefacto generado incorpora hashes del contenido filtrado de Kaikki, pues sus URLs de exportación son una instantánea actual y no publican un identificador de revisión inmutable en la misma forma que Hugging Face. [13] [14] [15]

## Referencias

[1]: https://dumps.wikimedia.org/ "Wikimedia Downloads"
[2]: https://dumps.wikimedia.org/legal.html "License information about Wikimedia dump downloads"
[3]: https://creativecommons.org/licenses/by-sa/4.0/deed.en "Creative Commons Attribution-ShareAlike 4.0 International"
[4]: https://tatoeba.org/en/downloads "Tatoeba downloads and licenses"
[5]: https://paracrawl.eu/ "ParaCrawl release 9"
[6]: https://github.com/google-research-datasets/MGSM-Rev2 "MGSM-Rev2"
[7]: https://github.com/facebookresearch/belebele "Belebele benchmark"
[8]: https://github.com/facebookresearch/flores/blob/main/flores200/README.md "FLORES-200 documentation"
[9]: https://huggingface.co/api/datasets/wikimedia/wikipedia "Wikimedia Wikipedia dataset API metadata"
[10]: https://huggingface.co/api/datasets/Helsinki-NLP/tatoeba "Helsinki-NLP Tatoeba dataset API metadata"
[11]: https://huggingface.co/api/datasets/juletxara/mgsm "MGSM dataset API metadata"
[12]: https://huggingface.co/api/datasets/facebook/belebele "Belebele dataset API metadata"
[13]: https://kaikki.org/dictionary/rawdata.html "Kaikki raw Wiktextract downloads"
[14]: https://kaikki.org/dictionary/English/index.html "Kaikki English dictionary"
[15]: https://kaikki.org/dictionary/Spanish/index.html "Kaikki Spanish dictionary"


## Adición para la recuperación del piloto inglés

La ejecución remota mostró que `olc-pd-books-en` quedó en 6.523 documentos ingleses después de varios HTTP 502, por lo que el mínimo de 14.000 no se alcanzó. Como fuente de recuperación se incorpora `manu/project_gutenberg`, usando exclusivamente el split `en`, que la tarjeta del dataset describe con 61,3 mil filas inglesas dentro de un total de 75.570 filas y texto de libros de Project Gutenberg. La revisión inmutable usada en el manifiesto es `164853d214065df26a630ee1ab91a0c39e461caf`, verificada mediante la API del Hub el 17 de agosto de 2026. El endpoint de filas devolvió dos filas reales con columnas `id` y `text`, y la primera contenía 358.423 caracteres; por tanto, es compatible con `hf_rows_api` y con el límite determinista de 10.000 documentos del piloto.

Esta adición no elimina OLC, Wikipedia ni Kaikki y no rebaja la puerta de 14.000 documentos. La ejecución seguirá fallando si alguna fuente no alcanza su mínimo declarado o si el total inglés no llega a 14.000. La licencia de Project Gutenberg exige conservar su aviso y verificar por obra la situación jurídica en la jurisdicción de redistribución; el piloto conserva la procedencia y no autoriza por sí solo la distribución pública del corpus o del modelo.

[16]: https://huggingface.co/datasets/manu/project_gutenberg "Project Gutenberg dataset card"
[17]: https://huggingface.co/api/datasets/manu/project_gutenberg/revision/main "Project Gutenberg immutable revision metadata"
[18]: https://www.gutenberg.org/policy/license.html "Project Gutenberg license policy"

## Respaldo oficial fuera de Hugging Face: fragmento de Wikipedia inglesa

El 19 de agosto de 2026 se comprobó que el servidor oficial de Wikimedia responde `200 OK`, acepta descargas parciales mediante `Range` y publica el fragmento `enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2` en `https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles-multistream1.xml-p1p41242.bz2`. La respuesta inspeccionada declara 299.138.062 bytes, `Last-Modified: Wed, 05 Aug 2026 18:41:29 GMT` y `ETag: "6a7383d9-11d47c0e"`.

La documentación oficial describe los dumps `pages-articles-multistream.xml.bz2` como revisiones actuales sin páginas de discusión ni de usuario y explica que cada flujo multistream contiene aproximadamente 100 páginas. El adaptador nuevo reanuda el archivo comprimido con `Range`, lee XML de forma incremental, conserva sólo espacio principal, descarta redirecciones y reduce marcado de MediaWiki sin inventar contenido. Después aplica los filtros, hashes, deduplicación y puertas bilingües ya existentes. El texto de los dumps se conserva bajo CC BY-SA y GFDL conforme a la documentación oficial; su incorporación no equivale a autorizar una distribución pública posterior sin atribución y revisión de licencia.

[19]: https://dumps.wikimedia.org/enwiki/latest/ "Índice oficial de dumps de Wikipedia inglesa"
[20]: https://en.wikipedia.org/wiki/Wikipedia:Database_download "Guía oficial de descargas de Wikipedia"
