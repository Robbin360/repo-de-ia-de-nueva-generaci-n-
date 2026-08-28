"""Contrato estático de la selección del checkpoint Edge canónico en Kaggle."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "training" / "AETHEL_EDGE_PHASE1_EVALUATION_KAGGLE_CELLS_V1.md"


def main() -> None:
    source = GUIDE.read_text(encoding="utf-8")
    required = (
        'ARTIFACT_DATASET = "aethel-edge-phase1-artifacts-v1"',
        'CANONICAL_ARTIFACT_FOLDER = "aethel-edge-phase-1-183680-v1"',
        'if any("preservation" in part for part in root.parts):',
        "excluded_tar_copies.append(root)",
        "TAR_COPIES_EXCLUDED:",
    )
    forbidden = (
        'artifact_candidates = [path.parent for path in INPUT_ROOT.rglob("latest.pt")',
        "len(artifact_candidates) != 1:\n    raise RuntimeError(f\"Se esperaba exactamente un input de artefactos Edge",
    )
    for expected in required:
        assert expected in source, expected
    for unexpected in forbidden:
        assert unexpected not in source, unexpected
    print("AETHEL_EDGE_EVALUATION_CELLS_CANONICAL_ARTIFACT_CONTRACT_VALIDATED")


if __name__ == "__main__":
    main()
