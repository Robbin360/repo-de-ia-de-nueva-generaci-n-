import os
import shutil
import tarfile
from pathlib import Path

os.environ["AETHEL_DATA_DIR"] = "/kaggle/working/aethel-nextgen-data"
os.environ["AETHEL_PERSISTENCE_MODE"] = "notebook-output"
os.environ["AETHEL_BUILD_DATA_IN_KAGGLE"] = "YES"
os.environ["AETHEL_RUN_AUTHORIZED"] = "YES"

kaggle_input = Path("/kaggle/input")
if not kaggle_input.is_dir():
    raise RuntimeError("Kaggle no montó /kaggle/input; verifica que el cuaderno tenga un Input Dataset.")

# Kaggle puede eliminar la extensión .gz y también puede montar el dataset
# con un slug distinto. Por eso se busca por firma gzip en todos los archivos.
bundles = []
for path in kaggle_input.rglob("*"):
    if not path.is_file() or path.name.startswith("."):
        continue
    try:
        with path.open("rb") as stream:
            if stream.read(2) == bytes.fromhex("1f8b"):
                bundles.append(path)
    except OSError:
        continue

if len(bundles) != 1:
    mounted = sorted(str(path) for path in kaggle_input.rglob("*") if path.is_file())
    raise RuntimeError(
        "Se esperaba exactamente un bundle gzip de Aethel NextGen, "
        f"pero se encontraron {len(bundles)}: {[str(path) for path in bundles]}. "
        "Archivos montados: " + str(mounted[:50]) + ". "
        "Si la lista está vacía, usa Add Input y adjunta el dataset privado "
        "aethel-nextgen-source antes de ejecutar de nuevo."
    )

bundle = bundles[0]
print(f"Bundle Aethel detectado: {bundle}")

target = Path("/kaggle/working/aethel-nextgen-source")
shutil.rmtree(target, ignore_errors=True)
with tarfile.open(bundle, "r:gz") as archive:
    archive.extractall("/kaggle/working", filter="data")

launcher = target / "training" / "run_kaggle_nextgen_in_situ.sh"
if not launcher.is_file():
    raise RuntimeError(
        f"El bundle se extrajo, pero falta el launcher esperado: {launcher}"
    )

os.environ["AETHEL_SOURCE_DIR"] = str(target)
!bash {launcher}
