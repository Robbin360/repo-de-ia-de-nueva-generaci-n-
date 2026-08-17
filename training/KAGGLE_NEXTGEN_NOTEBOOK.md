# Cuaderno limpio de Kaggle para Aethel NextGen

Este flujo sustituye el cuaderno y el Dataset de checkpoints de **Aethel V3**. No usa `aethel-v3-checkpoints`, no intenta cargar archivos `.pth` crudos y no declara resultados antes de una ejecución real.

| Entrada de Kaggle | Contenido exigido | Estado permitido |
|---|---|---|
| `aethel-nextgen-source` | `aethel-nextgen-source.tar.gz`, creado con `training/build_kaggle_nextgen_source_bundle.sh` | Necesario antes de editar el cuaderno. |
| `aethel-nextgen-data` | Ya no es entrada: el cuaderno lo prepara desde fuentes aprobadas dentro de `/kaggle/working` y conserva hashes, manifiesto y evaluación retenida. | Construido sólo durante la versión comprometida. |
| Salida del cuaderno comprometido | Checkpoints, métricas, manifiesto, tokenizador y `persistence_receipt.txt` bajo `/kaggle/working/aethel-runs/nextgen-pilot`. | Persistencia de la versión de Kaggle; un Dataset de artefactos es una opción posterior. |

> El script se detiene antes de instalar paquetes o usar CUDA si falta la autorización final, si se detecta una ruta de Aethel V3 o si las fuentes, el manifiesto o la evaluación no son verificables.

Primero, desde el repositorio Aethel local, construye el paquete de fuentes sin pesos históricos:

```bash
bash training/build_kaggle_nextgen_source_bundle.sh
```

Sube los dos archivos resultantes de `/home/ubuntu/aethel-kaggle-bundles/` a un nuevo Dataset privado denominado `aethel-nextgen-source`. El archivo comprimido debe conservar el nombre `aethel-nextgen-source.tar.gz`.

En un cuaderno nuevo de Kaggle, agrega como **Input** solamente el Dataset privado `aethel-nextgen-source`. Configura GPU T4 x2, activa Internet y pega esta única celda. No usa checkpoints ni datos de Aethel V3.

```python
import os
import shutil
import tarfile
from pathlib import Path

os.environ["AETHEL_DATA_DIR"] = "/kaggle/working/aethel-nextgen-data"
os.environ["AETHEL_PERSISTENCE_MODE"] = "notebook-output"
os.environ["AETHEL_BUILD_DATA_IN_KAGGLE"] = "YES"
os.environ["AETHEL_RUN_AUTHORIZED"] = "YES"

bundle = Path("/kaggle/input/aethel-nextgen-source/aethel-nextgen-source.tar.gz")
target = Path("/kaggle/working/aethel-nextgen-source")
if not bundle.is_file():
    raise FileNotFoundError(f"Missing source bundle: {bundle}")
shutil.rmtree(target, ignore_errors=True)
with tarfile.open(bundle, "r:gz") as archive:
    archive.extractall("/kaggle/working", filter="data")

os.environ["AETHEL_SOURCE_DIR"] = str(target)
!bash /kaggle/working/aethel-nextgen-source/training/run_kaggle_nextgen_in_situ.sh
```

Esta celda está preparada exclusivamente para **Save Version → Save & Run All (Commit)** autorizado. Antes de usarla, verifica que el cuaderno esté privado, que Internet esté activo y que GPU T4 x2 esté seleccionado. El guard conserva la verificación de manifiesto, evaluación, checkpoint empaquetado y persistencia de la salida versionada antes de declarar que el piloto ha terminado.
