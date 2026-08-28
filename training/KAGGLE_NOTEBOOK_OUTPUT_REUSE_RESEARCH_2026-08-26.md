# Reutilización de salida Kaggle para Aethel Edge

## Evidencia verificada

La documentación oficial de Kaggle indica que, al crear un dataset desde **Datasets → New Dataset**, se puede escoger una de cuatro fuentes de carga, incluida **Notebook Outputs**. El selector permite buscar archivos producidos por notebooks.[1]

La misma documentación indica que el menú de compartición controla la visibilidad y que la configuración por defecto para un dataset es **Private**.[1]

La salida preservada de `Aethel — Construcción de Corpus Edge V2` contiene la carpeta `aethel-edge-corpus-v1-retry-openr1-aligned-flags`, con `prepared_manifest.json`, `tokenizer.json`, diez shards `train-*.jsonl.gz` y `validation.jsonl.gz`. La captura compartida por el usuario confirma que la versión de commit guarda dichos archivos.

## Decisión operativa

El cuaderno principal de entrenamiento usará exactamente dos inputs: el bundle de código `aethel-direct-train-source-v1` y el nuevo dataset privado `aethel-edge-corpus-v1`. Este último sustituye a `aethel-nextgen-data-v1` para Edge; no se añade como un tercer input.

La interfaz puede rotular el selector como **Notebook Outputs** y pedir buscar el notebook del usuario. Si no muestra el notebook privado en el buscador, no se descargarán ni se modificarán los archivos: se solicitará una captura del diálogo y se seguirá la ruta alternativa permitida por la interfaz.

El entrenamiento permanece bloqueado hasta que el dataset se cree, se adjunte y se verifique de forma independiente.

## Referencias

[1] [Kaggle, "How to Use Kaggle — Datasets"](https://www.kaggle.com/docs/datasets#creating-a-dataset)
