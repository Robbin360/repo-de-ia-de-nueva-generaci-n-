"""Verifica que la guía Kaggle Edge tenga exactamente tres celdas y un alcance de sólo datos."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    guide = (ROOT / "AETHEL_EDGE_CORPUS_BUILD_KAGGLE_CELLS_V1.md").read_text(encoding="utf-8")
    assert guide.count("## CELDA ") == 3
    assert "edge-corpus-build-openr1-aligned-flags-v1" in guide
    assert "Aethel — Construcción de Corpus Edge V1 — Reintento OpenR1" in guide
    assert "aethel-edge-corpus-v1" in guide
    assert "aethel-nextgen-data-v1" in guide
    assert "EXPECTED_SOURCES" in guide
    assert "fineweb-en-sample-10bt-proposed" in guide
    assert "/kaggle/working/aethel-edge-corpus-v1-retry-openr1-aligned-flags" in guide
    assert "las configuraciones Hugging Face" in guide
    assert "usando sólo el dataset y su revisión" in guide
    assert "valida la capacidad por idioma" in guide
    assert "AETHEL_EDGE_CORPUS_BUILD_READY" in guide
    assert "No entrena, no selecciona GPU, no carga checkpoints y no evalúa holdout." in guide
    assert "subprocess.run([\"bash\", str(SOURCE_ROOT / \"training\" / \"run_kaggle_build_edge_corpus_v1.sh\")" in guide
    assert "train_aethel_gpu.py" not in guide
    print("AETHEL_EDGE_CORPUS_BUILD_CELLS_CONTRACT_OK")


if __name__ == "__main__":
    main()
