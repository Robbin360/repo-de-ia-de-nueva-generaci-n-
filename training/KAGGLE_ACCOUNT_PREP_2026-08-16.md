# Preparación observada de Kaggle — 16 de agosto de 2026

La cuenta de Kaggle de Félix Tremigual inició sesión correctamente y se revisó sin crear un cuaderno, habilitar acelerador ni ejecutar código.

| Recurso observado | Tipo | Estado visible | Uso potencial |
|---|---|---|---|
| `reanudar entrenamiento` | Cuaderno privado | Actualizado hace un día; borrador | Candidato para inspección de una receta previa, sin ejecutarlo. |
| `Puntos de control de Aethel v3` | Dataset privado | Actualizado hace dos días; 2 GB | Candidato para recuperación de checkpoints, sujeto a verificación de compatibilidad y hashes. |
| `modelo aethel`, `eahel-2`, `eathel-1` | Modelos privados | Variaciones privadas visibles | No se asumen compatibles hasta revisar sus metadatos. |

## Cuaderno inspeccionado sin modificación

El cuaderno privado [`reanudar entrenamiento`](https://www.kaggle.com/code/felixtremigual/reanudar-entrenamiento) tiene una versión 4/4, usa el dataset privado `Puntos de control de Aethel v3` como entrada y registró una ejecución fallida de 45.2 segundos con una topología visible de **GPU T4 × 2**. Se observó la primera celda de instalación de dependencias, pero no se ejecutó, editó, descargó ni reanudó el cuaderno durante esta inspección.

La presencia visible de dos T4 hace que el preflight FSDP sea potencialmente aplicable, pero no confirma que la cuota siga disponible, que la sesión pueda reasignarse con dos GPU, ni que los checkpoints sean compatibles. La causa del fallo previo debe leerse en `Registros` antes de cualquier edición o nueva ejecución.

## Diagnóstico de la ejecución fallida

Los registros muestran que la imagen de Kaggle ya tenía Python 3.12, PyTorch 2.10.0 con CUDA 12.8 y Triton 3.6.0. El cuaderno detectó CUDA, localizó el checkpoint crudo `aethel_v3_chkpt_10000.pth` y falló al realizar una carga estricta con `model.load_state_dict(loaded_data)`. Por tanto, el bloqueo no fue una instalación ni una ausencia de GPU: es una incompatibilidad entre la arquitectura declarada en el cuaderno y las claves o formas del checkpoint previo.

No se reanudó ni modificó el cuaderno. La corrección debe usar el cargador de compatibilidad actual de Aethel, inspeccionar claves y formas, y rechazar cualquier recuperación que no tenga manifiesto, hashes y configuración de modelo verificables.

## Límites de esta observación

Este inventario no confirma la disponibilidad de GPU, la cuota restante, la licencia de los pesos ni la compatibilidad de los artefactos. La próxima acción que pueda crear coste o consumir cuota requiere una confirmación explícita del usuario en la página de configuración del cuaderno.
