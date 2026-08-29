from __future__ import annotations

from router_assignment_health import analyze_hard_assignments


def test_uniform_assignments_have_full_coverage_and_zero_overflow() -> None:
    result = analyze_hard_assignments([0, 1, 2, 3] * 2, expert_count=4, capacity=2)
    assert result["coverage"] == 1.0
    assert result["max_density"] == 0.25
    assert result["overflow_count"] == 0
    assert result["entropy"] == 1.0


def test_collapsed_assignments_expose_concentration_and_missing_experts() -> None:
    result = analyze_hard_assignments([0] * 7 + [1], expert_count=4, capacity=4)
    assert result["coverage"] == 0.5
    assert result["max_density"] == 0.875
    assert result["overflow_count"] == 3
    assert result["overflow_rate"] == 0.375


def test_invalid_assignments_fail_closed() -> None:
    for invalid in ([], [-1], [4], [True]):
        try:
            analyze_hard_assignments(invalid, expert_count=4)
        except (TypeError, ValueError):
            pass
        else:
            raise AssertionError("La entrada inválida no fue rechazada")
