import hashlib
import json
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
    raise RuntimeError("No hay archivos montados bajo /kaggle/input.")

candidates = []
for path in files:
    try:
        with tarfile.open(path, mode="r:*") as archive:
            names = archive.getnames()
            manifest_bytes = b""
            manifest_members = [
                name for name in names if name.endswith(".manifest.json")
            ]
            for manifest_name in manifest_members:
                member = archive.extractfile(manifest_name)
                if member is not None:
                    manifest_bytes += member.read(2_000_000)
        required = {
            "launcher": any(
                name.endswith("training/run_kaggle_nextgen_in_situ.sh")
                for name in names
            ),
            "trainer": any(name.endswith("engine/train_aethel_gpu.py") for name in names),
            "olc_source": b"olc-pd-books-en" in manifest_bytes,
            "gutenberg_source": b"project-gutenberg-en" in manifest_bytes,
            "source_count": len(names),
            "size": path.stat().st_size,
        }
        if required["launcher"] and required["trainer"]:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            candidates.append((path, required, digest))
    except (tarfile.TarError, OSError):
        continue

if not candidates:
    raise RuntimeError(
        "No se encontró un bundle Aethel válido con launcher y trainer. "
        f"Archivos montados: {[str(path) for path in files]}"
    )

# Si Kaggle montó copias duplicadas, priorizamos el bundle que contiene
# Project Gutenberg inglés y, después, OLC pd_books. Luego usamos completitud,
# tamaño y hash. Así una copia anterior con OLC no puede ganar a la versión nueva.
candidates.sort(
    key=lambda item: (
        item[1]["gutenberg_source"],
        item[1]["olc_source"],
        item[1]["source_count"],
        item[1]["size"],
        item[2],
        str(item[0]),
    ),
    reverse=True,
)
bundle, details, digest = candidates[0]
if not details["gutenberg_source"]:
    raise RuntimeError(
        "El selector no encontró un bundle con project-gutenberg-en. "
        "No se permite continuar con una copia antigua que sólo contenga OLC."
    )
print(
    "Bundles válidos detectados: "
    f"{len(candidates)}; seleccionado: {bundle} "
            f"(project_gutenberg={details['gutenberg_source']}, "
        f"olc_pd_books={details['olc_source']}, "
        f"archivos internos={details['source_count']}, "
    f"bytes={details['size']}, sha256={digest[:16]}...)"
)

if len(candidates) > 1:
    print("Copias descartadas:")
    for discarded, discarded_details, discarded_digest in candidates[1:]:
        print(
            f"- {discarded} (project_gutenberg={discarded_details['gutenberg_source']}, "
            f"olc_pd_books={discarded_details['olc_source']}, "
            f"archivos internos={discarded_details['source_count']}, "
            f"bytes={discarded_details['size']}, sha256={discarded_digest[:16]}...)"
        )

target = Path("/kaggle/working/aethel-nextgen-source")
shutil.rmtree(target, ignore_errors=True)
with tarfile.open(bundle, mode="r:*") as archive:
    archive.extractall("/kaggle/working", filter="data")

launcher = target / "training" / "run_kaggle_nextgen_in_situ.sh"
if not launcher.is_file():
    raise RuntimeError(f"El bundle se extrajo, pero falta el launcher: {launcher}")

os.environ["AETHEL_SOURCE_DIR"] = str(target)
!bash {launcher}
