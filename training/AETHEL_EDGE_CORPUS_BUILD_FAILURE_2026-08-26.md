# Incidente de construcción de corpus Aethel Edge

**Estado:** construcción incompleta; no promover, no reutilizar ni borrar la salida Kaggle existente.

La ejecución autorizada de la CELDA 3 llegó a `prepare_bilingual_corpus.py`, pero se detuvo con `ValueError: BuilderConfig 'eng_Latn' not found`. La lista que devolvió el proveedor no contiene `eng_Latn`; sí muestra configuraciones de idiomas en el formato ISO 639-3 con escritura, como `spa_Latn`.

La revisión del repositorio oficial de FineWeb2 confirma que su punto de partida son datos no ingleses del FineWeb original. Por ello, `HuggingFaceFW/fineweb-2` no ofrece una configuración `eng_Latn` y el problema pertenece a la entrada **FineWeb2 EN**, no a HPLT ES. La fuente inglesa debe sustituirse por un corte explícito del FineWeb original, cuya configuración se verificará antes de reemitir el bundle.

**Fuente:** [HuggingFace FineWeb2](https://github.com/huggingface/fineweb-2), consultada el 26 de agosto de 2026.

No se inició entrenamiento ni se usó GPU. La advertencia de Hugging Face sobre solicitudes sin autenticación no fue la causa del fallo. El fallo corresponde a una configuración declarativa del manifiesto que no existe en el dataset actualmente resuelto.

## Segundo intento: incompatibilidad del preflight

Tras sustituir la fuente inglesa por FineWeb `sample-10BT`, la nueva CELDA 3 se detuvo antes de descargar textos, crear shards o producir un manifiesto de corpus. El `datasets` instalado en Kaggle rechazó el argumento `trust_remote_code` cuando el preflight llamó a `get_dataset_config_names`, propagándolo a `DownloadConfig`, cuya firma no lo admite: `TypeError: DownloadConfig.__init__() got an unexpected keyword argument 'trust_remote_code'`.

La evidencia confirma que la revisión corregida, las cuatro fuentes autorizadas y las celdas 1–2 se resolvieron correctamente. Este incidente es una incompatibilidad de argumentos del preflight, no una nueva decisión de fuentes ni un fallo de las configuraciones FineWeb/HPLT. El arreglo deberá llamar a `get_dataset_config_names` sólo con `path` y `revision`, pues el preflight no ejecuta código remoto ni necesita ese argumento.

## Regla de recuperación

No vuelva a ejecutar la CELDA 3 con el mismo `EDGE_DATA_OUTPUT` y no borre su contenido. La corrección deberá emitirse en un bundle de código nuevo y cualquier nuevo intento deberá usar un directorio de salida Kaggle distinto, preservando la evidencia de este intento fallido.

## Tercer intento: filtro vectorial de OpenR1 rechazó todos los ejemplos matemáticos

El siguiente intento superó el preflight de las cuatro configuraciones —FineWeb `sample-10BT`, FineWeb2 `spa_Latn`, HPLT `spa_Latn` y OpenR1-Math `default`— y alcanzó el límite autorizado de 100.000 documentos ingleses. Después se detuvo con `RuntimeError: Datos insuficientes para en: 100000 < 120000`.

La investigación inicial atribuyó el error al umbral, pero la inspección de una fila pública de OpenR1 confirmó la causa completa: los campos `is_reasoning_complete` y `correctness_math_verify` son listas de booleanos alineadas por traza. El adaptador los comparaba incorrectamente contra booleanos escalares y rechazó los 20.000 ejemplos matemáticos que debían completar el mínimo inglés. Por tanto, el mínimo de 120.000 sí es factible con los límites autorizados: 100.000 textos FineWeb más 20.000 ejemplos OpenR1 que cumplan ambas validaciones en la misma traza. La estructura publicada del conjunto `default` documenta ambos campos como secuencias booleanas y declara 93.733 ejemplos; véase [OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k/blob/main/README.md), consultado el 26 de agosto de 2026.

El arreglo debe conservar el mínimo, aceptar una fila matemática sólo si alguna posición emparejada de ambas listas es `true`, y verificar localmente que los límites declarados por idioma son factibles antes de cualquier solicitud de red. La salida actual sigue siendo incompleta: no se promueve, reutiliza ni borra; el siguiente intento debe usar un `EDGE_DATA_OUTPUT` distinto.
