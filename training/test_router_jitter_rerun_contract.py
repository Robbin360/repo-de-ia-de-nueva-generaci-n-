"""Contrato estático de la repetición jitter que debe preservar su checkpoint."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TRAINING = ROOT / "training"
TRANSFER = ROOT.parent / "aethel-private-transfer"


def main() -> None:
    marker = json.loads((TRAINING / "aethel_direct_train_source_release.json").read_text(encoding="utf-8"))
    assert marker["revision"] == "router-selection-jitter-rerun-preserve-v3-save-version-gate"
    assert marker["fresh_initialization"] is True
    assert marker["checkpoint_loading_authorized"] is False
    assert marker["holdout_read_authorized"] is False
    assert marker["promotion_authorized"] is False
    profile = marker["training_profile"]
    assert profile["seed"] == 17 and profile["steps"] == 768
    assert profile["router_jitter_noise"] == 0.01
    runner = (TRAINING / "run_kaggle_router_jitter_rerun_v1.sh").read_text(encoding="utf-8")
    assert "train_aethel_gpu.py" in runner
    assert "package_router_jitter_rerun.py" in runner
    assert "--router-jitter-noise 0.01" in runner
    assert "--max-steps 768" in runner
    assert "--output" in runner
    assert "PRESERVATION_PACKAGE" in runner
    assert "sync" in runner
    assert "holdout" not in runner.lower()
    guide = TRANSFER / "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_3_CELLS.md"
    assert guide.is_file()
    text = guide.read_text(encoding="utf-8")
    assert "CELDA 1" in text and "CELDA 2" in text and "CELDA 3" in text
    assert "router-selection-jitter-rerun-preserve-v3-save-version-gate" in text
    assert "SAVE_KAGGLE_VERSION_NOW.txt" in text
    assert "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_READY" in text
    assert "SAVE_KAGGLE_VERSION" in text
    recovery = TRANSFER / "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_RECOVERY_CELL.md"
    assert recovery.is_file()
    recovery_text = recovery.read_text(encoding="utf-8")
    assert "AETHEL_ROUTER_JITTER_RERUN_PRESERVATION_READY" in recovery_text
    assert "SAVE_KAGGLE_VERSION" in recovery_text
    assert "train_aethel_gpu.py" not in recovery_text
    assert "torch.load" not in recovery_text
    assert "subprocess" not in recovery_text
    print("AETHEL_ROUTER_JITTER_RERUN_CONTRACT_OK")


if __name__ == "__main__":
    main()
