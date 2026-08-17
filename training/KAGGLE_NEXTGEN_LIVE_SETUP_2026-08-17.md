# Estado del borrador Kaggle Aethel NextGen

Fecha de comprobación: 2026-08-17.

El cuaderno privado `felixtremigual/aethel-nextgen-bilingual-pilot` existe como borrador y tiene adjunto el Dataset privado `felixtremigual/aethel-nextgen-source`.

La primera celda fue sustituida manualmente. La inspección visual confirmó el inicio esperado: importaciones de `os`, `shutil`, `tarfile` y `Path`; las variables `AETHEL_DATA_DIR`, `AETHEL_PERSISTENCE_MODE`, `AETHEL_BUILD_DATA_IN_KAGGLE` y `AETHEL_RUN_AUTHORIZED`; y la verificación de que exista exactamente un archivo `.gz` bajo `/kaggle/input/aethel-nextgen-source` antes de extraerlo.

La sesión seguía apagada y no se ejecutó ninguna celda. La verificación visual del tramo final de la celda no pudo completarse porque el panel de código no expuso un contenedor desplazable a la automatización. Antes de comprometer la versión debe confirmarse visualmente que incluye la comprobación de `run_kaggle_nextgen_in_situ.sh` y termina con `!bash {launcher}`.

La configuración de sesión observada en Kaggle muestra **GPU T4 x2** con marca de selección. El menú muestra la acción `Turn off internet`, lo que indica que el acceso a Internet está activo. Estas observaciones se realizaron sin iniciar una sesión de cómputo ni ejecutar una celda.

El Dataset de fuentes permanece privado y su versión 1 contiene el paquete inicial. Antes del compromiso debe publicarse una nueva versión con el paquete SHA-256 `ba64c014fe9ca51611bc0d8325ee4e5a6e738b812423a778393408bd5704d829`, que incorpora el tokenizador portátil, los recibos de recuperación y los checkpoints atómicos reforzados.

La página de Kaggle expone la acción **New Version** para el Dataset privado; la actualización debe conservar el Dataset como privado y reemplazar únicamente el paquete de fuentes, sin crear un Dataset de corpus adicional antes de que la propia corrida lo construya.

La ventana de nueva versión está abierta: contiene la nota de versión sobre checkpoints atómicos, tokenizador portátil y reanudación verificable. El archivo de la versión 1 fue retirado de la propuesta de versión 2 para que el bootstrap encuentre exactamente un paquete comprimido; aún falta añadir el paquete actualizado por URL y crear la versión.

**Estado observado:** Kaggle completó satisfactoriamente la **versión 2** privada de `felixtremigual/aethel-nextgen-source` con el paquete actualizado. El cuaderno debe usar esta versión antes de comprometerse.

Al recargar `felixtremigual/aethel-nextgen-bilingual-pilot`, Kaggle muestra el Dataset de fuentes adjunto y el estado de sesión borrador apagada; sin embargo, el lienzo del editor aparece vacío y la tabla de contenidos no detecta celdas. No se debe pulsar **Save Version** hasta restaurar la celda de bootstrap y confirmar que la entrada usa la versión 2.

## Estado observado tras regenerar el paquete de retención

La página de Kaggle `https://www.kaggle.com/datasets/felixtremigual/aethel-nextgen-source` muestra el Dataset como **PRIVATE**, con una sola entrada binaria y **Version 2**. El campo Provenance apunta al paquete `gWTAXnCagwvOipgz.gz`; la acción visible en Data Explorer es `New Version`. El nuevo paquete local de fuentes con retención acotada fue publicado temporalmente en `https://files.manuscdn.com/user_upload_by_module/session_file/310519663046068494/VdjXzANoiKeHTeds.gz` con SHA-256 `318518929932c771feeb76f5821923b13bc5d6cb1452c32e884cbd83276f5b31`. Aún no se ha ejecutado el cuaderno ni se ha iniciado GPU.

Fuente observada: página privada de Kaggle del Dataset, consultada el 17 de agosto de 2026.

---


La navegación actual sigue mostrando `Version 2`, un único archivo `gWTAXnCagwvOipgz` y la Provenance del paquete anterior. La acción `New Version` está presente en el contenido extraído, pero no queda expuesta como elemento interactivo en el viewport actual; todavía no se ha iniciado una actualización ni se ha ejecutado el cuaderno.

---


## Estado tras Save & Run All

Kaggle muestra `aethel-nextgen-bilingual-pilot Version #1 with GPU T4 x2 — Failed`. El borrador conserva la celda y el Dataset privado `aethel-nextgen-source` aparece adjunto. El intento de abrir el registro `scriptVersionId=342929998` devolvió timeout del navegador; todavía no se conoce el traceback real. No se inició una segunda ejecución ni se inventa la causa del fallo.

Siguiente acción segura: abrir manualmente el enlace de logs de la versión #1 y copiar las últimas líneas/traceback antes de modificar el cuaderno.
