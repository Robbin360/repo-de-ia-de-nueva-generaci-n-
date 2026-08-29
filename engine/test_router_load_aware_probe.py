from __future__ import annotations

import math

from router_load_aware_probe import load_aware_logits, top_k_indices


def test_load_penalty_changes_top_k_when_one_pair_is_saturated() -> None:
    logits = [1.0, 0.99, 0.2, 0.1]
    load = [10.0, 9.0, 0.0, 0.0]
    assert top_k_indices(logits) == (0, 1)
    adjusted = load_aware_logits(logits, load, strength=0.2)
    assert top_k_indices(adjusted) == (2, 3)


def test_zero_strength_preserves_logits() -> None:
    logits = [0.5, -0.2, 0.1]
    load = [3.0, 1.0, 2.0]
    assert load_aware_logits(logits, load, strength=0.0) == logits


def test_rejects_non_finite_and_mismatched_inputs() -> None:
    try:
        load_aware_logits([1.0, 2.0], [1.0], strength=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched lengths must fail")
    try:
        load_aware_logits([1.0, math.inf], [1.0, 2.0], strength=0.1)
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite logits must fail")


if __name__ == "__main__":
    test_load_penalty_changes_top_k_when_one_pair_is_saturated()
    test_zero_strength_preserves_logits()
    test_rejects_non_finite_and_mismatched_inputs()
    print("ROUTER_LOAD_AWARE_PROBE_TEST_OK")
