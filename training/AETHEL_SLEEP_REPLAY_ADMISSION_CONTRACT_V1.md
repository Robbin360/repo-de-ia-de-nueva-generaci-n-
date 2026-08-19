# Contrato V1 de admisión de replay para Sueño

**Estado:** validado por CPU únicamente. No activa entrenamiento, GPU, red, acceso al holdout, optimizador, ajuste LoRA ni promoción.

## Finalidad

La puerta de admisión recibe **metadatos** de eventos líquidos. Un evento sólo puede cruzar a una lista de replay en cuarentena si ya fue curado y aprobado. La salida no contiene el texto del evento; guarda solamente su identificador, procedencia, hash, idioma, dominio, prioridad, TTL y revisor.

| Regla | Requisito | Resultado al fallar |
|---|---|---|
| Procedencia | `source`, `source_sha256` hexadecimal y `event_id` no vacíos. | Rechazo. |
| Alcance | Idioma `en` o `es`, dominio declarado y TTL positivo. | Rechazo. |
| Curación | `curation_status=approved` y `approved_by` explícito. | Rechazo. |
| Cuarentena | `eligible_for_sleep=true` sólo después de curación. | Rechazo. |
| Separación | El hash no puede pertenecer al holdout ni llevar `holdout_member=true`. | Rechazo. |
| Deduplicación | No se repiten `event_id` ni `source_sha256`. | Rechazo. |

## Salida deliberadamente limitada

El manifiesto `aethel_sleep_replay_quarantine` está ordenado por prioridad y queda ligado al hash de La Roca padre. Aun así declara todas las capacidades siguientes como falsas:

| Campo | Valor |
|---|---|
| `eligible_for_training` | `false` |
| `eligible_for_promotion` | `false` |
| `holdout_access_enabled` | `false` |
| `external_action_enabled` | `false` |
| `optimizer_creation_enabled` | `false` |

> La admisión a replay no equivale a admisión a entrenamiento. Es una selección trazable que debe superar una puerta posterior de presupuesto, datos `train`, evaluación y autorización antes de cualquier ajuste LoRA.

La prueba CPU cubre admisión de eventos aprobados, ordenamiento por prioridad, exclusión de contenido, rechazo de falta de curación, TTL vencido, colisión con holdout y duplicados.
