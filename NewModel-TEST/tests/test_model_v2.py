import torch

from fh_transport.model_v2 import CompetingChannelFranckHertz


def test_competing_history_probability_is_conserved_for_h1_and_h4s() -> None:
    va = torch.tensor([0.0, 15.0, 40.0, 80.0])
    for hypothesis in ("h1", "h4s"):
        model = CompetingChannelFranckHertz(
            hypothesis=hypothesis, n_layers=8, max_collisions=3, quadrature_order=4
        )
        probability = model.collision_state_probabilities(va)
        assert torch.all(probability >= 0.0)
        assert torch.allclose(
            probability.sum(dim=2), torch.ones_like(probability[:, :, 0]), atol=2.0e-5
        )


def test_h4s_contains_nonzero_mixed_collision_history() -> None:
    model = CompetingChannelFranckHertz(
        hypothesis="h4s", n_layers=12, max_collisions=3, quadrature_order=4
    )
    probability = model.collision_state_probabilities(torch.tensor([80.0]))
    mixed_index = model.history_index((1, 1, 0, 0))
    mixed_mass = torch.sum(probability[0, :, mixed_index] * model.thermal_weights)
    assert float(mixed_mass.detach()) > 0.0


def test_collector_is_exactly_zero_when_axial_field_is_unreachable() -> None:
    model = CompetingChannelFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=3, quadrature_order=4
    )
    gate = model.collector_state_transmission(torch.tensor([0.0]), torch.tensor([0.0]))
    current = model(torch.tensor([0.0]), torch.tensor([0.0]))
    assert torch.count_nonzero(gate) == 0
    assert float(current.detach()) == 0.0


def test_retarding_voltage_reduces_v2_current() -> None:
    model = CompetingChannelFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=3, quadrature_order=4
    )
    va = torch.linspace(0.0, 60.0, 31)
    low = model(va, torch.zeros_like(va))
    high = model(va, torch.full_like(va, 10.0))
    assert torch.all(high <= low + 1.0e-6)


def test_v2_gradients_are_finite() -> None:
    model = CompetingChannelFranckHertz(
        hypothesis="h4s", n_layers=5, max_collisions=2, quadrature_order=3
    )
    va = torch.linspace(0.0, 50.0, 21)
    loss = model(va, torch.full_like(va, 6.0)).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
