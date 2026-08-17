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
    raise RuntimeError(
        "Kaggle no montó /kaggle/input. Adjunta aethel-nextgen-source en Add Input."
    )

files = sorted(path for path in kaggle_input.rglob("*") if path.is_file())
if not files:
    raise RuntimeError(
        "No hay archivos montados bajo /kaggle/input. "
        "Adjunta el dataset privado aethel-nextgen-source y vuelve a guardar la versión."
    )

# Kaggle puede renombrar los archivos y puede presentar más de una entrada
# (por ejemplo, el bundle y su manifiesto). Identificamos el bundle por el
# contenido interno requerido, aceptando gzip, tar sin compresión u otro modo
# reconocido por tarfile.
bundle = None
candidate_details = []
for path in files:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            names = archive.getnames()
        has_launcher = any(
            name.endswith("training/run_kaggle_nextgen_in_situ.sh")
            for name in names
        )
        candidate_details.append(
            {"path": str(path), "members": len(names), "has_launcher": has_launcher}
        )
        if has_launcher:
            if bundle is not None:
                raise RuntimeError(
                    "Se encontraron varios bundles Aethel válidos: "
                    f"{bundle} y {path}. Deja adjunto solamente "
                    "aethel-nextgen-source."
                )
            bundle = path
    except (tarfile.TarError, OSError):
        continue

if bundle is None:
    raise RuntimeError(
        "No se encontró un bundle que contenga "
        "training/run_kaggle_nextgen_in_situ.sh. "
        f"Archivos montados: {[str(path) for path in files]}. "
        f"Candidatos tar inspeccionados: {candidate_details}. "
        "Verifica que aethel-nextgen-source se haya creado a partir de "
        "build_kaggle_nextgen_source_bundle.sh."
    )

print(f"Bundle Aethel detectado por contenido: {bundle}")

target = Path("/kaggle/working/aethel-nextgen-source")
shutil.rmtree(target, ignore_errors=True)
with tarfile.open(bundle, mode="r:*") as archive:
    archive.extractall("/kaggle/working", filter="data")

launcher = target / "training" / "run_kaggle_nextgen_in_situ.sh"
if not launcher.is_file():
    raise RuntimeError(f"El bundle se extrajo, pero falta el launcher: {launcher}")

os.environ["AETHEL_SOURCE_DIR"] = str(target)
!bash {launcher}
