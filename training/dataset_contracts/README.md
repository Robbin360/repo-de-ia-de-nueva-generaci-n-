# Contratos de Dataset versionados

Este directorio contiene **metadatos reproducibles**, no datos de entrenamiento. Sirve para que un operador o un chat posterior valide que el paquete privado montado corresponde exactamente al Dataset congelado esperado antes de ejecutar cualquier preflight o corrida Seed.

| Archivo | Propósito | Contiene datos de entrenamiento |
|---|---|---|
| `aethel-knowledge-reasoning-bilingual-v1.manifest.json` | Rutas relativas, tamaños, hashes, conteos y contrato del tokenizador. | No |
| `aethel-knowledge-reasoning-bilingual-v1.validation.json` | Resultado de la validación offline del paquete local congelado. | No |

El paquete de bytes continúa deliberadamente fuera de GitHub en una ubicación privada. El operador debe aportar un directorio privado cuyo contenido reproduzca las 22 rutas `corpus/*.jsonl.gz`, el tokenizador y los manifiestos, y verificar sus hashes contra este contrato. No se debe sustituir el Dataset por una descarga de red ni generar shard alguno durante la corrida.

> **Estado:** el contrato documenta el paquete local `aethel-knowledge-reasoning-bilingual-v1`; no acredita que exista una copia en Kaggle ni que se haya iniciado una corrida GPU.
