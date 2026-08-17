import os
import shutil
import tarfile
from pathlib import Path

os.environ["AETHEL_DATA_DIR"] = "/kaggle/working/aethel-nextgen-data"
os.environ["AETHEL_PERSISTENCE_MODE"] = "notebook-output"
os.environ["AETHEL_BUILD_DATA_IN_KAGGLE"] = "YES"
os.environ["AETHEL_RUN_AUTHORIZED"] = "YES"

input_dir = Path("/kaggle/input/aethel-nextgen-source")
bundles = sorted(
    path for path in input_dir.iterdir()
    if path.is_file() and not path.name.startswith(".")
)
if len(bundles) != 1:
    raise RuntimeError(
        "Expected exactly one Aethel NextGen source bundle under "
        f"{input_dir}, found {len(bundles)}: {[path.name for path in bundles]}"
    )

bundle = bundles[0]
with bundle.open("rb") as stream:
    if stream.read(2) != bytes.fromhex("1f8b"):
        raise RuntimeError(f"Source bundle is not gzip-compressed: {bundle}")

target = Path("/kaggle/working/aethel-nextgen-source")
shutil.rmtree(target, ignore_errors=True)
with tarfile.open(bundle, "r:gz") as archive:
    archive.extractall("/kaggle/working", filter="data")

launcher = target / "training" / "run_kaggle_nextgen_in_situ.sh"
if not launcher.is_file():
    raise RuntimeError(
        f"Extracted source bundle does not contain expected launcher: {launcher}"
    )

os.environ["AETHEL_SOURCE_DIR"] = str(target)
!bash {launcher}
