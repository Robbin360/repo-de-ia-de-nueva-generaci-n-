"""Contrato estático de la guía de tres celdas para la primera sesión Edge."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    guide = (ROOT / "AETHEL_EDGE_LONG_TRAIN_KAGGLE_CELLS_V1.md").read_text(encoding="utf-8")
    assert guide.count("## CELDA ") == 3
    assert "edge-long-session-phase1-kaggle-jsonl-v1" in guide
    assert "Aethel — Entrenamiento Directo Dataset V1" in guide
    assert "aethel-direct-train-source-v1" in guide
    assert "aethel-edge-corpus-v1" in guide
    assert "aethel-nextgen-data-v1" in guide
    assert "exactamente **dos inputs**" in guide
    assert "EXPECTED_SHARDS = 10" in guide
    assert "EDGE_DATA_LAYOUT:" in guide
    assert "EDGE_MANIFEST_PATH" in guide
    assert "EDGE_TOKENIZER_PATH" in guide
    assert "kaggle-descomprimido-jsonl" in guide
    assert "GZIP_HASH_COMPARISON" in guide
    assert "no se afirma verificación independiente del contenido" in guide
    assert "torch.cuda.is_available()" in guide
    assert "single / world_size=1" in guide
    assert "SESSION_TARGET_STEP\": \"183680\"" in guide
    assert "SCHEDULE_TOTAL_STEPS\": \"734720\"" in guide
    assert "run_kaggle_edge_long_session_v1.sh" in guide
    assert "RESUME_CHECKPOINT no debe estar definido" in guide
    assert "Save Version" in guide
    assert "no demuestra bilingüismo, razonamiento, matemáticas ni eficiencia" in guide
    print("AETHEL_EDGE_LONG_TRAIN_CELLS_CONTRACT_OK")


if __name__ == "__main__":
    main()
