# Aethel Knowledge & Reasoning Bilingual v1

Este paquete contiene un **corpus real, bilingüe y trazable** para la fase de preentrenamiento de Aethel NextGen. No incluye texto generado, ejemplos sintéticos ni métricas de entrenamiento. Cada registro conserva su idioma, título, identificador de revisión, URL de origen, licencia, hash del texto y etiquetas de dominio para auditoría.

| Componente | Contenido | Uso permitido en la canalización |
|---|---|---|
| `corpus/` | Shards JSONL comprimidos, con `split=train` y `split=holdout` | Entrenamiento y evaluación de pérdida; el holdout no debe entrar en actualizaciones de pesos. |
| `tokenizer-input/` | Sólo documentos `train` en shards JSONL comprimidos | Entrenar un BPE reproducible sin contaminar el holdout. |
| `metadata.json` | Inventario de fuentes, URLs, licencias y hashes de dumps descargados | Trazabilidad y reproducibilidad. |
| `validation_report.json` | Resultado de la validación sin red | Puerta obligatoria antes del entrenamiento. |
| `package_manifest.json` | Hashes de cada shard empaquetado | Verificación al montar el Dataset. |

## Alcance y licencia

Los textos proceden de los dumps oficiales de Wikipedia en inglés y español y están señalados en cada registro como **CC BY-SA 4.0**. El usuario que redistribuya, entrene o publique artefactos derivados debe evaluar y respetar las obligaciones de atribución y compartir-igual que resulten aplicables.[1] [2] [3]

El corpus proporciona exposición a prosa enciclopédica y a conceptos de lenguaje, matemáticas, ciencias, ingeniería y programación. Por sí solo, **no demuestra razonamiento humano, conciencia, autonomía ni seguridad de un modelo**. Es necesario medir esos aspectos mediante entrenamiento real, evaluación retenida, pruebas de seguridad y revisión humana posteriores.

## Reglas de entrenamiento offline

El trabajo GPU deberá montar este Dataset privado como entrada y no ejecutar descargas de corpus ni peticiones a APIs. Antes de iniciar cualquier sesión, se debe ejecutar la validación local del paquete y comprobar los hashes del manifiesto. Los checkpoints y métricas se guardarán exclusivamente como salida versionada de la ejecución para poder reanudar sin reconstruir ni volver a descargar datos.

## Referencias

[1]: https://dumps.wikimedia.org/enwiki/latest/ "Wikimedia Dumps — English Wikipedia"
[2]: https://dumps.wikimedia.org/eswiki/latest/ "Wikimedia Dumps — Spanish Wikipedia"
[3]: https://creativecommons.org/licenses/by-sa/4.0/ "Creative Commons Attribution-ShareAlike 4.0 International"
