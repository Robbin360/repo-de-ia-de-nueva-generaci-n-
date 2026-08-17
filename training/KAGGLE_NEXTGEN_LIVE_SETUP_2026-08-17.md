# Estado del borrador Kaggle Aethel NextGen

Fecha de comprobación: 2026-08-17.

El cuaderno privado `felixtremigual/aethel-nextgen-bilingual-pilot` existe como borrador y tiene adjunto el Dataset privado `felixtremigual/aethel-nextgen-source`.

La primera celda fue sustituida manualmente. La inspección visual confirmó el inicio esperado: importaciones de `os`, `shutil`, `tarfile` y `Path`; las variables `AETHEL_DATA_DIR`, `AETHEL_PERSISTENCE_MODE`, `AETHEL_BUILD_DATA_IN_KAGGLE` y `AETHEL_RUN_AUTHORIZED`; y la verificación de que exista exactamente un archivo `.gz` bajo `/kaggle/input/aethel-nextgen-source` antes de extraerlo.

La sesión seguía apagada y no se ejecutó ninguna celda. La verificación visual del tramo final de la celda no pudo completarse porque el panel de código no expuso un contenedor desplazable a la automatización. Antes de comprometer la versión debe confirmarse visualmente que incluye la comprobación de `run_kaggle_nextgen_in_situ.sh` y termina con `!bash {launcher}`.

La configuración de sesión observada en Kaggle muestra **GPU T4 x2** con marca de selección. El menú muestra la acción `Turn off internet`, lo que indica que el acceso a Internet está activo. Estas observaciones se realizaron sin iniciar una sesión de cómputo ni ejecutar una celda.
