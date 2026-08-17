"""Pruebas del calendario de aprendizaje sin iniciar entrenamiento ni usar GPU."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from train_aethel_gpu import learning_rate_at_step


def test_warmup_and_cosine_floor_are_bounded() -> None:
    peak, floor = 3e-4, 3e-5
    assert learning_rate_at_step(1, 1000, peak, floor, 100) == peak / 100
    assert learning_rate_at_step(100, 1000, peak, floor, 100) == peak
    assert floor <= learning_rate_at_step(500, 1000, peak, floor, 100) <= peak
    assert learning_rate_at_step(1000, 1000, peak, floor, 100) == floor


if __name__ == "__main__":
    test_warmup_and_cosine_floor_are_bounded()
    print("OK")
