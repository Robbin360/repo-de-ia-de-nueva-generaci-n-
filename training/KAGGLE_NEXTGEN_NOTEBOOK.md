# Cuaderno limpio de Kaggle para Aethel NextGen

Este flujo sustituye el cuaderno y el Dataset de checkpoints de **Aethel V3**. No usa `aethel-v3-checkpoints`, no intenta cargar archivos `.pth` crudos y no declara resultados antes de una ejecución real.

| Entrada de Kaggle | Contenido exigido | Estado permitido |
|---|---|---|
| `aethel-nextgen-source` | Un único archivo `.gz` creado con `training/build_kaggle_nextgen_source_bundle.sh`. Kaggle puede conservar el nombre remoto de la URL. | Necesario antes de editar el cuaderno. |
| `aethel-nextgen-data` | Ya no es entrada: el cuaderno lo prepara desde fuentes aprobadas dentro de `/kaggle/working` y conserva hashes, manifiesto y evaluación retenida. | Construido sólo durante la versión comprometida. |
| Salida del cuaderno comprometido | Checkpoints, métricas, manifiesto, tokenizador y `persistence_receipt.txt` bajo `/kaggle/working/aethel-runs/nextgen-pilot`. | Persistencia de la versión de Kaggle; un Dataset de artefactos es una opción posterior. |

> El script se detiene antes de instalar paquetes o usar CUDA si falta la autorización final, si se detecta una ruta de Aethel V3 o si las fuentes, el manifiesto o la evaluación no son verificables.

Primero, desde el repositorio Aethel local, construye el paquete de fuentes sin pesos históricos:

```bash
bash training/build_kaggle_nextgen_source_bundle.sh
```

Sube el archivo comprimido resultante de `/home/ubuntu/aethel-kaggle-bundles/` a un nuevo Dataset privado denominado `aethel-nextgen-source`. Si Kaggle lo importa por URL, puede asignarle un nombre remoto; la celda siguiente rechaza cero o más de un archivo `.gz` para no extraer una entrada ambigua.

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

input_dir = Path("/kaggle/input/aethel-nextgen-source")
bundles = sorted(input_dir.glob("*.gz"))
if len(bundles) != 1:
    raise RuntimeError(
        "Expected exactly one compressed Aethel NextGen source bundle under "
        f"{input_dir}, found {len(bundles)}: {[path.name for path in bundles]}"
    )
bundle = bundles[0]
target = Path("/kaggle/working/aethel-nextgen-source")
shutil.rmtree(target, ignore_errors=True)
with tarfile.open(bundle, "r:gz") as archive:
    archive.extractall("/kaggle/working", filter="data")

launcher = target / "training" / "run_kaggle_nextgen_in_situ.sh"
if not launcher.is_file():
    raise RuntimeError(f"Extracted source bundle does not contain expected launcher: {launcher}")

os.environ["AETHEL_SOURCE_DIR"] = str(target)
!bash {launcher}
```

Esta celda está preparada exclusivamente para **Save Version → Save & Run All (Commit)** autorizado. Antes de usarla, verifica que el cuaderno esté privado, que Internet esté activo y que GPU T4 x2 esté seleccionado. El guard conserva la verificación de manifiesto, evaluación, checkpoint empaquetado y persistencia de la salida versionada antes de declarar que el piloto ha terminado.
