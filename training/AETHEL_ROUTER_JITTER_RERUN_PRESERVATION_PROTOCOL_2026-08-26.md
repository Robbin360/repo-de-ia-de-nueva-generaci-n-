# Protocolo de repetición y preservación — Router Jitter V1

## Motivo

La corrida real anterior `router-selection-jitter-v1` terminó con **446/768** pasos saludables, pero su checkpoint estuvo únicamente en la ruta efímera de Kaggle. La inspección de sólo lectura posterior confirmó que no existe un checkpoint Aethel recuperable ni en `/kaggle/working` ni en los inputs de la nueva sesión. La evidencia métrica se conserva, pero no permite probar carga o generación.

## Hipótesis y perfil invariables

La repetición ejecuta la misma sonda que produjo la mejora: inicio fresco, seed 17, 768 pasos, Dataset v1 train-only, secuencia 1024, batch 1, acumulación 16, BF16, dimensión 512, 4 capas, 8 expertos, top-2, `router_aux_loss_weight=0.05`, `router_entropy_loss_weight=0.03` y `router_jitter_noise=0.01`. No reanuda pesos, no abre holdout, no cambia arquitectura ni corpus.

## Salida inédita y preservación

La única salida permitida es `/kaggle/working/aethel-direct-train-router-jitter-v1-rerun-v1`. Si existe, el flujo se detiene antes de leer datos o iniciar GPU. Después de que el validador complete, el empaquetador crea en `/kaggle/working` el archivo `aethel-direct-train-router-jitter-v1-rerun-v1-preservation.tar.gz`, un recibo JSON con SHA-256 del checkpoint, tokenizador, métricas, diagnóstico, `recovery_receipt.json`, `aethel_direct_validation.json` y paquete, y la compuerta incluida `SAVE_KAGGLE_VERSION_NOW.txt`. El lanzador ejecuta `sync` y sólo entonces declara éxito. El operador debe usar **Save Version en la misma sesión y antes de cualquier otra acción**; la compuerta no automatiza la subida ni concede promoción.

> El paquete en `/kaggle/working` sigue siendo efímero hasta que el usuario guarde explícitamente la versión de Kaggle. El recibo confirma empaquetado local, no persistencia remota, carga, generación, calidad, promoción ni eficiencia.

## Criterio de éxito operativo

La corrida debe emitir `AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_READY`; el recibo debe declarar `PERSISTENCE_ACTION_REQUIRED: SAVE_KAGGLE_VERSION`. Sólo entonces se pide guardar la versión de Kaggle antes de cerrar la sesión. La clasificación de salud sigue el protocolo jitter original y no promociona automáticamente ningún checkpoint.
