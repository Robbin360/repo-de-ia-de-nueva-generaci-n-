# Contrato de inferencia local — Mojo

Mojo es la **ruta objetivo** de inferencia local token a token; todavía no está instalado ni validado en este entorno. Este documento define el contrato que deberá cumplir una implementación Mojo antes de que Aethel la anuncie como disponible.

| Operación | Entrada | Salida obligatoria | Verificación contra PyTorch |
|---|---|---|---|
| `load_artifact` | Directorio firmado con `model.safetensors`, `config.json`, tokenizador y manifiesto SHA-256 | Identificador de modelo y configuración validada | Hashes idénticos y rechazo de manifiesto incompleto. |
| `prefill` | Tokens `[batch, prompt]` | Logits del último token y KV-cache por capa | Error máximo de logits dentro de la tolerancia FP16/BF16 acordada. |
| `decode` | Token `[batch, 1]` + KV-cache | Logits y KV-cache extendida | Igualdad de tokens greedy y longitud correcta de caché. |
| `health` | Ninguna | Backend, dtype, memoria, versión de artefacto | Publicación local de p50/p95, tokens/s y uso de memoria. |

El runtime no obtiene corpus, claves ni conversaciones del usuario sin una política local explícita. No se medirá ni prometerá rendimiento hasta ejecutar esta matriz sobre el hardware objetivo.
