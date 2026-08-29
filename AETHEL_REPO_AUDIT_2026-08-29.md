# Auditoría de posible pérdida de archivos — 2026-08-29

## Hallazgos iniciales

La copia local contiene estos archivos críticos: `AETHEL_SYSTEM_ARCHITECTURE_V1.md`, `AETHEL_DOCUMENTATION_INDEX.md`, `AETHEL_EXTERNAL_BLOCKERS_RUNBOOK_V1.md`, `AETHEL_BASE_TRAINING_PLAN_100M_300M.md`, `AETHEL_BASE_CAPABILITY_SPEC.md`, `engine/aethel_model.py`, `engine/aethel_nextgen.py` y `todo.md`.

La rama `main` visible mediante GitHub CLI en `Robbin360/repo-de-ia-de-nueva-generaci-n-` está en el commit `43a52b929d31059761f9019e809ecca589532d56`, fechado el 2026-08-28, con mensaje `Document Aethel Pro architecture and VRAM budget`. En ese árbol remoto están `engine/aethel_model.py`, `engine/aethel_nextgen.py` y `todo.md`, pero no aparecen los cinco documentos nuevos de arquitectura/plan/índice/runbook comprobados localmente.

## Interpretación provisional

Esto no demuestra que Google AI Studio haya borrado archivos de la copia local. La evidencia apunta primero a una **desincronización**: los documentos existen localmente, mientras que el GitHub seleccionado todavía muestra un estado anterior. Además, el remoto Git configurado en el proyecto no pudo autenticarse durante `git fetch`, porque apunta a un endpoint de artefactos con credenciales no disponibles en esa operación. No se debe ejecutar `git reset`, `git checkout` destructivo ni restaurar desde GitHub hasta resolver qué copia es la fuente de verdad.

## Próximas comprobaciones

Se debe confirmar el HEAD local, su rama, si la copia local está limpia, qué commits locales no están en GitHub y si existe una copia alternativa en `Robbin360/katalog-ai`. Después se compararán hashes de los archivos críticos. La acción segura probable será preservar la copia local y sincronizarla mediante el flujo autorizado, no descargar el árbol remoto antiguo sobre ella.

## Resultado de la auditoría

La copia local está en la rama `main`, HEAD corto `59f1b1d`, correspondiente al checkpoint más reciente de la documentación y el runbook. `git status` mostró únicamente dos cambios producidos por esta auditoría: la línea nueva de `todo.md` y este informe no versionado. No se detectaron archivos eliminados ni modificaciones externas en el estado local.

El GitHub visible está en `43a52b929d`, anterior a los checkpoints locales `95ab7804`, `db79fe9f` y `59f1b1d`. Por eso faltan en GitHub la arquitectura maestra, el índice, el runbook, el plan base 100M–300M y la especificación de capacidades base, mientras sí existen localmente. Esto es una divergencia de sincronización, no evidencia de que Google AI Studio haya borrado esos archivos de la copia local.

`engine/aethel_nextgen.py` coincide por hash entre local y GitHub. `engine/aethel_model.py` y `todo.md` difieren porque la copia local contiene cambios posteriores al estado remoto; no deben reemplazarse con la versión remota antigua. El remoto Git configurado por WebDev es un endpoint interno de artefactos y no pudo hacer `fetch` autenticado desde la shell; el repositorio público se consultó de forma independiente mediante GitHub CLI.

## Decisión de preservación

No se restaura desde GitHub ni se ejecuta `reset`. La copia local de Aethel se considera la fuente más completa de esta sesión. Antes de cualquier sincronización pública se debe crear un checkpoint del informe y decidir explícitamente si se desea publicar los commits locales posteriores al GitHub visible. Los datos, checkpoints y pesos continúan fuera del repositorio público.
