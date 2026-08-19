# Congelación local del Dataset Aethel v1

**Estado:** congelado localmente por decisión del usuario el 19 de agosto de 2026. Este documento no autoriza ninguna carga a Kaggle, inicio de notebook, reserva de GPU ni entrenamiento. El siguiente artefacto sólo podrá publicarse o usarse para entrenamiento tras una nueva autorización explícita.

| Propiedad | Valor verificado |
|---|---:|
| Identificador | `aethel-knowledge-reasoning-bilingual-v1` |
| Directorio del paquete | `/home/ubuntu/aethel-knowledge-corpus-v1-package` |
| Tamaño distribuible | 194 MB |
| Shards comprimidos | 22 |
| Documentos totales | 40.000 |
| Entrenamiento inglés / español | 19.011 / 19.012 |
| Holdout inglés / español | 989 / 988 |
| Hash lógico de documentos | `94089b8d4776b2e49f483d74fef87ccd328c878c26e3567f62a65932e21756c6` |
| Tokenizador BPE | 32.000 entradas, derivado sólo del split `train` |
| SHA-256 del tokenizador | `4a3608e4e45c9117415d1f4fa236aebe20771dc3a3ce85760d9fb9d218fa0815` |

## Controles vigentes

La procedencia procede de fragmentos oficiales de Wikipedia en inglés y español, conservando el manifiesto de fuentes, metadatos, hashes por shard y el informe de validación. Los registros se filtran y etiquetan con dominio; se rechazan duplicados por contenido y el conjunto `holdout` permanece separado de las entradas del tokenizador. La validación final verificó los 22 shards, sus hashes, el esquema de cada registro, los conteos por idioma y split, y el hash del tokenizador sin efectuar solicitudes de red.

> **Límite de interpretación:** el corpus representa una base enciclopédica bilingüe real y verificable. No basta por sí solo para demostrar razonamiento humano, aprendizaje autónomo o rendimiento de frontera. Cualquier afirmación futura deberá basarse en entrenamiento y evaluación reproducibles.

## Activación futura

Para levantar esta congelación serán necesarias dos decisiones separadas: la autorización para publicar el paquete como Dataset privado y, posteriormente, una autorización independiente para una corrida GPU offline que valide el paquete antes de entrenar. Ninguna decisión está implícita en este archivo.
