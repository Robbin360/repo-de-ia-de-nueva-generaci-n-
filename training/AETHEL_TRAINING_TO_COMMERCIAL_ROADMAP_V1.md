# Ruta de entrenamiento autorizado a operación comercial de Aethel v1

La ruta comercial no salta de un Dataset a una afirmación de producto. Cada fase produce una evidencia concreta y puede detenerse sin modificar La Roca de producción.

| Fase | Objetivo | Evidencia de salida | Decisión |
|---|---|---|---|
| 0. Preparación | Congelar datos, código y gates. | Hashes, validación offline, runbook y pruebas. | Preparada. |
| 1. Seed E0 | Obtener una línea base real pequeña. | Checkpoint, pérdida/perplejidad en/en-es, router, VRAM y restauración. | Requiere GPU y autorización. |
| 2. Ablaciones | Medir memoria, Líquido, workspace y cómputo adaptativo contra E0. | Comparaciones con mismas semillas, datos y presupuesto. | Sólo si E0 es reproducible. |
| 3. Edge v1 | Escalar arquitectura y Dataset tras corregir hallazgos. | Evaluación, coste, seguridad y operación de piloto. | Requiere GPU persistente. |
| 4. Workspace privado | Integrar modelo Edge con conocimiento del cliente. | Control de acceso, recuperación citable, observabilidad y rollback. | Piloto comercial limitado. |
| 5. Pro/Sueño | Añadir MoE distribuido y adaptación gobernada. | Triton, FSDP, gates P0–P6 y revisión humana. | Decisión de inversión separada. |

El ciclo de Sueño nunca sustituye la fase 1: sólo recibe candidatos curados y aprobados, entrena adaptadores aislados y exige evaluación antes de promoción. La curiosidad propone recuperación, aclaración o replay; no inicia compras de infraestructura, acciones externas ni actualizaciones de pesos base.

La condición de “comercial” es operacional, no retórica: un servicio debe ser recuperable, medible, seguro, limitado a su dominio y útil para usuarios reales. Sin esa evidencia, Aethel permanece como investigación y laboratorio.
