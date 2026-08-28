# Protocolo de carga y generación mínima del checkpoint `router-selection-jitter-v1`

## Propósito

Este protocolo verifica solamente que el checkpoint recuperable producido por `DIRECT_TRAIN_ROUTER_JITTER_V1` puede cargarse de forma estricta y ejecutar una generación autoregresiva mínima. No entrena, no reanuda el optimizador, no modifica pesos, no lee corpus ni holdout y no constituye una evaluación de calidad lingüística, razonamiento, matemáticas, bilingüismo ni eficiencia relativa.

La prueba utiliza el checkpoint existente `/kaggle/working/aethel-direct-train-router-jitter-v1/latest.pt` y el tokenizador que quedó junto a él en la misma salida. Si cualquiera de los dos falta, la prueba debe detenerse sin crear una nueva salida ni intentar sustituirlos.

## Límites obligatorios

| Elemento | Regla fija |
|---|---|
| Pesos y optimizador | Se cargan desde `latest.pt`; no se crea optimizador y no se llama a `backward`, `step` ni `observe`. |
| Modo del modelo | `model.eval()` y `torch.inference_mode()` durante toda la generación. |
| Router | El jitter queda inactivo porque sólo opera en `model.train()`; se registra el valor observado en cada paso. |
| Estado de parámetros | Se calculan huellas SHA-256 antes y después de la inferencia; una diferencia bloquea el recibo exitoso. |
| Memorias persistentes | El modelo se construye contra una ruta temporal nueva; no se llama a `observe`, `flush` ni `consolidate`. |
| Datos | No se abre `train-*`, `holdout-*`, corpus, métricas de entrenamiento ni datasets adicionales. |
| Salida | Debe ser inédita: `/kaggle/working/aethel-direct-train-router-jitter-v1-inference-check-v1`. No se borra, reutiliza ni reanuda una salida existente. |
| GPU | Sólo inferencia de una secuencia por prompt; no se usa para entrenamiento. |

## Contrato de carga estricta

El evaluador debe validar que el payload tiene `model`, `config`, `step`, `tokenizer` y `tokenizer_sha256`; que el hash del tokenizador encontrado coincide con el checkpoint; y que `model.load_state_dict(..., strict=True)` no informa tensores faltantes ni inesperados. La configuración se reconstruye únicamente desde `checkpoint["config"]`, con `router_jitter_noise` preservado para auditarlo pero con el modelo en evaluación.

La ejecución no debe cargar ni utilizar el campo `optimizer` salvo que esté presente como parte del payload. Una discrepancia de configuración, hash, forma o estado detiene la prueba con un error explícito.

## Generación mínima controlada

Los prompts son fijos, cortos y sólo prueban el recorrido técnico de tokens:

| ID | Idioma | Prompt | Máximo de tokens nuevos |
|---|---|---|---:|
| `es_control` | Español | `Aethel:` | 32 |
| `en_control` | Inglés | `Aethel:` | 32 |

Cada prompt se codifica con el tokenizador del checkpoint, se limita a 32 tokens de contexto y se decodifica por *greedy decoding* (`argmax`), sin muestreo, herramientas, red, memoria persistente ni contenido holdout. El evaluador registra los IDs generados, la decodificación resultante, si los logits son finitos y la telemetría básica de ruta; no interpreta la salida como respuesta correcta ni como competencia lingüística.

## Criterios de resultado

| Estado | Condición |
|---|---|
| `CHECKPOINT_GENERATION_READY` | Carga estricta correcta, hash de tokenizador coincidente, todos los logits finitos, al menos un token por prompt, router sin jitter activo, huellas de parámetros iguales y recibo escrito. |
| `CHECKPOINT_GENERATION_BLOCKED` | Falta checkpoint/tokenizador, existe una salida previa, falla hash, falla carga estricta, aparecen logits no finitos, se activa jitter en evaluación o cambian parámetros. |

Un resultado `READY` demuestra sólo recuperabilidad e inferencia mínima. El checkpoint conserva su clasificación **MEASURED_NOT_PROMOTED** hasta completar evaluación holdout independiente, medición de estabilidad temporal del router y un baseline comparable de eficiencia.
