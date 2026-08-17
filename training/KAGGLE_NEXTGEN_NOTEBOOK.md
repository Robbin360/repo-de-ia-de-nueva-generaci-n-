# Cuaderno limpio de Kaggle para Aethel NextGen

Este flujo sustituye el cuaderno y el Dataset de checkpoints de **Aethel V3**. No usa `aethel-v3-checkpoints`, no intenta cargar archivos `.pth` crudos y no declara resultados antes de una ejecución real.

| Entrada de Kaggle | Contenido exigido | Estado permitido |
|---|---|---|
| `aethel-nextgen-source` | `aethel-nextgen-source.tar.gz`, creado con `training/build_kaggle_nextgen_source_bundle.sh` | Necesario antes de editar el cuaderno. |
| `aethel-nextgen-data` | Corpus preparado y aprobado, tokenizador BPE, `prepared_manifest.json`, holdout y referencias de evaluación accesibles | Necesario antes de activar GPU. |
| `felixtremigual/aethel-nextgen-artifacts` | Dataset privado vacío o versionable para checkpoints, métricas, manifiestos y memoria | Necesario para persistir resultados. |

> El script se detiene antes de instalar paquetes o usar CUDA si falta la autorización final, si se detecta una ruta de Aethel V3 o si las fuentes, el manifiesto o la evaluación no son verificables.

Primero, desde el repositorio Aethel local, construye el paquete de fuentes sin pesos históricos:

```bash
bash training/build_kaggle_nextgen_source_bundle.sh
```

Sube los dos archivos resultantes de `/home/ubuntu/aethel-kaggle-bundles/` a un nuevo Dataset privado denominado `aethel-nextgen-source`. El archivo comprimido debe conservar el nombre `aethel-nextgen-source.tar.gz`.

En un cuaderno nuevo de Kaggle, agrega como **Input** los Datasets `aethel-nextgen-source` y `aethel-nextgen-data`. Configura la aceleración GPU antes de guardar el borrador y pega esta única celda de código. El valor de autorización se deja vacío de forma deliberada.

```python
import os
import shutil
import tarfile
from pathlib import Path

os.environ["AETHEL_DATA_DIR"] = "/kaggle/input/aethel-nextgen-data"
os.environ["AETHEL_KAGGLE_DATASET"] = "felixtremigual/aethel-nextgen-artifacts"
os.environ["AETHEL_RUN_AUTHORIZED"] = ""

bundle = Path("/kaggle/input/aethel-nextgen-source/aethel-nextgen-source.tar.gz")
target = Path("/kaggle/working/aethel-nextgen-source")
if not bundle.is_file():
    raise FileNotFoundError(f"Missing source bundle: {bundle}")
shutil.rmtree(target, ignore_errors=True)
with tarfile.open(bundle, "r:gz") as archive:
    archive.extractall("/kaggle/working", filter="data")

os.environ["AETHEL_SOURCE_DIR"] = str(target)
!bash /kaggle/working/aethel-nextgen-source/training/bootstrap_kaggle_nextgen.sh
```

Al guardar este borrador no se ejecuta la celda. Después de recibir confirmación explícita para gastar la cuota gratuita, cambia únicamente `AETHEL_RUN_AUTHORIZED` a `YES` y usa **Save Version → Save & Run All (Commit)**. El guard conserva la verificación de manifiesto, evaluación, checkpoint empaquetado y persistencia antes del entrenamiento.
