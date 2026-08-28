"""Contrato estático de la preparación Kaggle para la evaluación aislada."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
TRANSFER = ROOT.parent / "aethel-private-transfer"


def main() -> None:
    marker = json.loads((TRAINING / "aethel_direct_train_source_release.json").read_text(encoding="utf-8"))
    assert marker["revision"] == "router-selection-jitter-rerun-preserve-v3-save-version-gate"
    assert marker["checkpoint_loading_authorized"] is False
    assert marker["holdout_read_authorized"] is False
    assert marker["promotion_authorized"] is False
    runner = (TRAINING / "run_kaggle_router_jitter_checkpoint_eval.sh").read_text(encoding="utf-8")
    assert "evaluate_router_jitter_checkpoint.py" in runner
    assert "--device cuda" in runner
    assert "--max-new-tokens 32" in runner
    assert "train_aethel_gpu.py" not in runner
    guide = TRANSFER / "AETHEL_ROUTER_JITTER_CHECKPOINT_EVALUATION_3_CELLS.md"
    assert guide.is_file()
    text = guide.read_text(encoding="utf-8")
    assert "CELDA 1" in text and "CELDA 2" in text and "CELDA 3" in text
    assert "router-jitter-checkpoint-evaluation-v1" in text
    assert "aethel-nextgen-source-e0-v1" in text
    assert "CHECKPOINT_GENERATION_READY" in text
    print("AETHEL_ROUTER_JITTER_CHECKPOINT_EVALUATION_CONTRACT_OK")


if __name__ == "__main__":
    main()
