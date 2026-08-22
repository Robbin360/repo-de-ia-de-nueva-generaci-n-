"""Pruebas CPU del contrato de dispatch/combina que un kernel Triton debe igualar."""
from __future__ import annotations

import torch

from triton_bridge import moe_capacity_reference, moe_dispatch_combine_reference


def legacy_dispatch(tokens, selected_experts, gates, experts):
    output = torch.zeros_like(tokens)
    for token_index in range(tokens.shape[0]):
        for slot in range(selected_experts.shape[1]):
            expert_index = int(selected_experts[token_index, slot])
            output[token_index] += experts[expert_index](tokens[token_index : token_index + 1])[0] * gates[token_index, slot]
    return output


def test_matches_legacy_route_and_preserves_gate_combination() -> None:
    tokens = torch.tensor([[1.0, 2.0], [3.0, 4.0], [-1.0, 5.0]])
    selected = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=torch.long)
    gates = torch.tensor([[0.75, 0.25], [0.40, 0.60], [0.50, 0.50]])
    experts = [lambda x: x * 2.0, lambda x: x - 3.0]

    actual = moe_dispatch_combine_reference(tokens, selected, gates, experts)
    expected = legacy_dispatch(tokens, selected, gates, experts)
    assert torch.allclose(actual, expected)


def test_ignores_empty_expert_and_keeps_gradients() -> None:
    tokens = torch.tensor([[2.0, -1.0], [4.0, 3.0]], requires_grad=True)
    selected = torch.tensor([[1, 1], [1, 1]], dtype=torch.long)
    gates = torch.tensor([[0.2, 0.8], [0.6, 0.4]], requires_grad=True)
    calls = [0, 0]

    def empty_expert(x):
        calls[0] += 1
        return x * 99

    def active_expert(x):
        calls[1] += 1
        return x.square()

    output = moe_dispatch_combine_reference(tokens, selected, gates, [empty_expert, active_expert])
    assert calls == [0, 1]
    assert torch.allclose(output, tokens.square())
    output.sum().backward()
    assert tokens.grad is not None and torch.isfinite(tokens.grad).all()
    assert gates.grad is not None and torch.isfinite(gates.grad).all()


def test_rejects_invalid_routing_contracts() -> None:
    tokens = torch.ones((1, 2))
    experts = [lambda x: x]
    try:
        moe_dispatch_combine_reference(tokens, torch.tensor([[1]], dtype=torch.long), torch.ones((1, 1)), experts)
    except ValueError as error:
        assert "fuera" in str(error)
    else:
        raise AssertionError("un índice fuera de rango debe bloquearse")


def test_capacity_reference_is_token_slot_deterministic_and_reports_overflow() -> None:
    selected = torch.tensor([[0, 1], [0, 0], [1, 0]], dtype=torch.long)
    positions, accepted, loads = moe_capacity_reference(selected, n_experts=2, capacity=2)
    assert torch.equal(positions, torch.tensor([[0, 0], [1, -1], [1, -1]]))
    assert torch.equal(accepted, torch.tensor([[True, True], [True, False], [True, False]]))
    assert torch.equal(loads, torch.tensor([2, 2]))


def test_capacity_reference_rejects_invalid_expert_index() -> None:
    try:
        moe_capacity_reference(torch.tensor([[2]], dtype=torch.long), n_experts=2, capacity=1)
    except ValueError as error:
        assert "rango" in str(error)
    else:
        raise AssertionError("la capacidad debe rechazar índices de expertos inválidos")


if __name__ == "__main__":
    test_matches_legacy_route_and_preserves_gate_combination()
    test_ignores_empty_expert_and_keeps_gradients()
    test_rejects_invalid_routing_contracts()
    test_capacity_reference_is_token_slot_deterministic_and_reports_overflow()
    test_capacity_reference_rejects_invalid_expert_index()
    print("PASS: referencia CPU de dispatch/combina MoE verificada")
