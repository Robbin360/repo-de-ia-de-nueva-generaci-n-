# Runbook local de GPU para Aethel v1

**Estado:** guía de preparación local; no certifica que exista una GPU conectada, no instala paquetes, no ejecuta CUDA y no inicia entrenamiento.  
**Propósito:** ofrecer una alternativa reproducible a sesiones temporales de Kaggle para Aethel Seed y una base de preparación para Edge.

## Principio de operación

La ruta local sirve cuando el usuario dispone de una computadora con GPU NVIDIA y puede mantenerla encendida durante la corrida. El agente sólo ve los directorios que el usuario conecta explícitamente mediante My Computer; no obtiene acceso general al disco, a credenciales ni a otras cuentas. El código fuente, el Dataset congelado y los artefactos de salida deben residir en un directorio montado y respaldado por el usuario.

> Esta ruta evita el límite de sesión de Kaggle, pero no elimina los controles de Aethel: el Dataset sigue congelado, holdout continúa aislado, La Roca no se modifica durante inferencia y el contrato Triton sigue bloqueando promoción comercial si sus kernels GPU no han sido validados.

| Recurso | Requisito para Seed local | Requisito adicional para Edge |
|---|---|---|
| GPU | Al menos una NVIDIA CUDA visible para una línea base E0. | VRAM, estabilidad térmica y horas de cómputo suficientes para la configuración elegida; deben medirse, no suponerse. |
| Procesador/RAM | Capacidad para descomprimir/leer shards y preparar lotes. | Mayor margen para prefetch, evaluación, recuperación y herramientas de operación. |
| Almacenamiento | Copia del repositorio, paquete congelado y checkpoints fuera del árbol de código. | Copias de Dataset, checkpoints y logs con política de backup. |
| Persistencia | El equipo debe permanecer encendido y el directorio montado debe sobrevivir a la sesión del agente. | Supervisión, restart, almacenamiento redundante y operación documentada. |

## Ámbito de directorios

El usuario debe conectar una carpeta de trabajo mediante My Computer. Como ejemplo conceptual, el agente ve esa carpeta bajo `/mnt/desktop/`; la ruta exacta se confirma cuando la computadora esté realmente conectada. La organización propuesta mantiene los datos, código y resultados separados.

```text
/mnt/desktop/aethel-local/
├── source/                         # clon o copia de aethel-platform
├── data/aethel-knowledge-corpus-v1-package/
│   ├── corpus/                     # 22 shards congelados
│   ├── tokenizer.json
│   ├── metadata.json
│   ├── package_manifest.json
│   └── validation_report.json
└── runs/aethel-seed-e0/            # checkpoints y evidencia de una corrida
```

No mezclar `runs/` dentro de `data/`, no editar shards congelados y no guardar Dataset o checkpoints en un directorio efímero. El destino de backup se define antes de la corrida y se verifica por hash tras cada exportación importante.

## Puertas de preparación

La primera ejecución no debe comenzar entrenando. Se ejecutan puertas en el siguiente orden y cada una conserva su reporte en `runs/aethel-seed-e0/`.

Antes del preflight puede ejecutarse `training/inspect_local_aethel_host.py`. El inspector no entrena ni reserva GPU: verifica archivos de código, integridad del Dataset, capacidad de escritura, espacio libre, CUDA y disponibilidad de Triton; después emite `host_inspection.json`. `READY_FOR_AUTHORIZATION` sólo significa que el host pasó sus controles locales; sigue siendo necesaria una autorización humana separada para Seed.

| Orden | Puerta | Resultado aceptable | Motivo de bloqueo |
|---:|---|---|---|
| 1 | Identidad de hardware | El sistema expone CUDA y registra nombre/VRAM. | No hay GPU CUDA o drivers compatibles. |
| 2 | Dataset | `validate_aethel_knowledge_package.py` produce informe válido sin red. | Faltan shard, hash, manifiesto o tokenizador. |
| 3 | Separación de evaluación | Existen holdout inglés y español y no entran a `train`. | Ruta equivocada o contaminación de split. |
| 4 | Kernel | Las pruebas CUDA/Triton pertinentes registran resultado, no un éxito simulado. | Error numérico, kernel ausente o dispositivo no compatible. |
| 5 | Autorización | El usuario autoriza el alcance, pasos y uso de GPU. | No existe autorización explícita. |
| 6 | Checkpoint | Directorio de salida es escribible y tiene backup definido. | No hay persistencia/reanudación verificable. |

## Preflight existente

El repositorio ofrece `training/run_gpu_preflight.sh`. Registra disponibilidad y número de dispositivos CUDA. Si hay menos de dos GPU CUDA, devuelve un estado `SKIPPED` para las validaciones Triton/FSDP avanzadas; no lo convierte en éxito. Con dos o más GPU, intenta validar Triton y FSDP con checkpoint/reanudación distribuida.

Esto no impide ejecutar Aethel Seed en una sola GPU: el lanzador `training/run_kaggle_seed_offline.sh` acepta una GPU CUDA para su línea base. Sin embargo, el nombre del script no limita su uso a Kaggle: las rutas se controlan mediante variables de entorno y pueden apuntar a directorios locales. El contrato sigue siendo el mismo.

## Configuración local de Seed E0

El siguiente bloque es una **plantilla**. No debe ejecutarse hasta que el usuario confirme la GPU, la ubicación de salida y el alcance de E0. Las rutas deben ajustarse al directorio realmente conectado.

```bash
export AETHEL_SOURCE_DIR=/mnt/desktop/aethel-local/source
export AETHEL_DATA_DIR=/mnt/desktop/aethel-local/data/aethel-knowledge-corpus-v1-package
export AETHEL_OUTPUT_DIR=/mnt/desktop/aethel-local/runs/aethel-seed-e0

# Sólo tras aprobación explícita de una corrida GPU real:
export AETHEL_RUN_AUTHORIZED=YES

# Sólo para E0 experimental si Triton completo continúa pendiente:
export AETHEL_LAB_FALLBACK_AUTHORIZED=YES

cd "$AETHEL_SOURCE_DIR"
bash training/run_kaggle_seed_offline.sh
```

El lanzador valida el paquete antes de cargar el modelo, exige CUDA, ejecuta las pruebas disponibles, guarda `latest.pt`, snapshots, métricas, recibo de recuperación e inspección de checkpoint. Luego evalúa holdout inglés y español por separado. Si `AETHEL_LAB_FALLBACK_AUTHORIZED` permanece distinto de `YES`, el contrato Triton estricto bloquea la actualización de pesos en CUDA.

## Evidencia y reanudación

Una Seed local sólo se considera recuperable cuando aparecen y se inspeccionan estos archivos:

| Artefacto | Uso |
|---|---|
| `package_preflight.json` | Integridad y procedencia del paquete montado. |
| `latest.pt` | Punto de reanudación con pesos, optimizador, paso y configuración. |
| `step_*.pt` | Snapshots portátiles de contingencia. |
| `recovery_receipt.json` | Confirmación del último paso seguro. |
| `metrics_rank_0.jsonl` | Pérdida, tokens/s, consumo y salud de routing observados. |
| `checkpoint_inspection.json` | Verificación de compatibilidad y datos reproducibles. |
| `evaluation_holdout_en.json` y `evaluation_holdout_es.json` | Evaluación retenida y separada por idioma. |

Una interrupción no justifica reiniciar desde cero ni extrapolar métricas. La siguiente sesión usa `latest.pt` sólo si el inspector confirma arquitectura y tokenizador compatibles. Si el chequeo falla, se conserva el artefacto para diagnóstico y no se fuerza una carga estricta.

## Diferencia entre Seed, Edge y operación comercial

Seed E0 demuestra la ruta técnica; no es el modelo comercial. Edge requiere un presupuesto de hardware y Dataset mayor, evaluación repetida, latencia/coste medidos, control de acceso y recuperación operativa. Los servicios El Líquido, curiosidad, memoria/RAG y gobierno de Sueño deben mantenerse fuera del bucle de token y con permisos separados.

La computadora local es una alternativa de coste inicial cero para desarrollo y Seed si ya existe hardware. Para operación comercial que deba sobrevivir apagados, actualizaciones y usuarios concurrentes, se necesitará un host persistente con backups, monitoreo y una política de incidentes. Este documento no recomienda ni activa una compra; sólo evita confundir un PC de experimento con un servicio de producción.

## Bloqueos conocidos al momento de redactar

1. El entorno actual de desarrollo no tiene GPU CUDA; este runbook no se ha ejecutado contra hardware local.
2. La validación CUDA completa de prefill causal y dispatch/combina MoE en Triton sigue pendiente.
3. FSDP requiere al menos dos procesos/GPU reales; el preflight actual lo registra como `SKIPPED` si no están disponibles.
4. Aethel Edge de ≈2,2 B parámetros no debe iniciarse como sustituto de Seed sin evidencias reales de E0 y planificación de memoria/datos.
5. La conexión My Browser no llegó a exponer una sesión personal a este chat; My Computer es una ruta distinta y requiere que el usuario conecte una carpeta explícitamente.
