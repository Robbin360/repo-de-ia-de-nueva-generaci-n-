# Contrato V1 del candidato de Sueño de Aethel

**Estado:** implementación CPU validada; no es un ciclo de entrenamiento ni una promoción de modelo.  
**Prohibiciones activas:** sin GPU, red, ingestión de datos, acceso al holdout, optimizador, ajuste LoRA, acciones externas o actualización de La Roca.

## Propósito

El contrato crea una frontera comprobable entre el checkpoint activo, llamado **La Roca**, y cualquier mejora hipotética producida durante Sueño. La primera operación del ciclo no aprende todavía: copia La Roca, instala adaptadores LoRA de bajo rango en la copia y declara el resultado como **candidato en cuarentena**. La mejora, si llegara a existir, vive únicamente dentro de esa rama.

| Artefacto | Contenido | Autoridad | Estado inicial |
|---|---|---|---|
| `rock_manifest.json` | Configuración canónica, hash de configuración y SHA-256 del estado completo de La Roca. | Referencia inmutable. | `active_reference`. |
| Candidato LoRA | Copia exacta de La Roca más adaptadores LoRA congelando pesos base. | Sólo permite gradientes futuros en `lora_a`/`lora_b`. | `quarantined_candidate`. |
| Manifiesto candidato | Hash de La Roca padre, hash de la copia base, parámetros LoRA y prohibiciones. | Trazabilidad y verificación. | No promovible. |
| Rollback | Descarte de la rama candidata. | Restaura la referencia por no haberla tocado. | Disponible inmediatamente. |

## Invariantes comprobables

> Un candidato de Sueño no es una versión nueva de La Roca: es una rama aislada que puede descartarse sin escribir pesos en la referencia.

| Invariante | Verificación implementada | Fallo que bloquea |
|---|---|---|
| **Referencia exacta** | El SHA-256 del `state_dict` de La Roca coincide antes y después de crear/descartar un candidato. | La Roca se modificó. |
| **Base clonada exacta** | El estado base del candidato, excluyendo `lora_a` y `lora_b`, tiene el mismo hash que La Roca. | La copia introdujo una variación base. |
| **Entrenabilidad mínima** | Las proyecciones base de módulos LoRA no requieren gradiente; existen parámetros LoRA entrenables. | El candidato podría reescribir La Roca o no tener adaptador. |
| **Cuarentena** | `training_started=false`, `eligible_for_promotion=false`, `holdout_access_enabled=false` y `external_action_enabled=false`. | El candidato reclama autoridad que no tiene. |
| **Rollback** | `discard_candidate` informa que no cambió La Roca. | La reversión no es verificable. |

## Secuencia autorizada hoy

La implementación disponible permite los pasos siguientes:

1. Crear `rock_manifest.json` desde una instancia sin LoRA activo.
2. Clonar el `state_dict` de La Roca en un modelo nuevo.
3. Habilitar LoRA con la base congelada en la copia, sin crear un optimizador.
4. Ejecutar `verify_candidate_isolation` antes y después de una modificación de prueba en matrices LoRA.
5. Ejecutar `rollback_candidate`, que simplemente descarta la rama y verifica el hash de referencia.

La prueba `engine/test_sleep_candidate.py` también realiza una alteración maliciosa controlada de una proyección base del candidato y confirma que la verificación la rechaza. Esta es una prueba de integridad de software, no una simulación de mejora del modelo.

## Puertas que siguen cerradas

Para pasar de la rama aislada a un verdadero ciclo de Sueño faltan puertas que **no están habilitadas** por este contrato:

| Puerta posterior | Requisito antes de abrirla |
|---|---|
| Curación | Eventos líquidos con procedencia, permisos, deduplicación, TTL y una revisión de admisión. Los eventos de curiosidad siguen con `eligible_for_sleep=false`. |
| Replay | Selección estratificada exclusivamente desde `train`; auditoría de intersección cero contra `holdout`. |
| Ajuste | Autorización explícita, presupuesto de cómputo, configuración inmutable, optimizador y logs de pérdida reales. |
| Evaluación | Baseline y candidato evaluados con idéntico tokenizador, protocolo e inputs retenidos. |
| Promoción | No regresión por idioma/dominio, integridad, reversión, coste registrado y aprobación explícita. |

Por tanto, crear un candidato **no inicia entrenamiento** y no permite al modelo aprender por sí mismo todavía. Sólo establece la estructura técnica que hará posible evaluar una mejora futura sin perder el control de la versión activa.
