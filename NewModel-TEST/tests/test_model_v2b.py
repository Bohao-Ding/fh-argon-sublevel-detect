import torch

from fh_transport.model_v2b import LastCollisionFranckHertz


def test_v2b_probability_and_last_position_moment_are_physical() -> None:
    model = LastCollisionFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    probability, last_mass = model.collision_state_moments(
        torch.tensor([0.0, 20.0, 50.0, 80.0])
    )
    last_position = last_mass / torch.clamp(probability, min=1.0e-12)
    assert torch.allclose(
        probability.sum(dim=2), torch.ones_like(probability[:, :, 0]), atol=2.0e-5
    )
    assert torch.all(last_position >= 0.0)
    assert torch.all(last_position <= 1.0)
    assert torch.count_nonzero(last_mass[:, :, 0]) == 0


def test_collision_near_collector_has_lower_transmission_than_early_collision() -> None:
    model = LastCollisionFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.tensor([80.0])
    vr = torch.tensor([10.0])
    shape = (1, model.thermal_nodes.numel(), model.collision_counts.shape[0])
    early = torch.zeros(shape)
    late = torch.zeros(shape)
    state = model.history_index((5,))
    early[:, :, state] = 0.20
    late[:, :, state] = 0.90
    early_gate = model.collector_state_transmission(va, vr, early)
    late_gate = model.collector_state_transmission(va, vr, late)
    early_value = torch.sum(early_gate[0, :, state] * model.thermal_weights)
    late_value = torch.sum(late_gate[0, :, state] * model.thermal_weights)
    assert float(early_value.detach()) > float(late_value.detach())


def test_v2b_angle_parameter_affects_post_collision_current() -> None:
    model = LastCollisionFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(45.0, 80.0, 25)
    current = model(va, torch.full_like(va, 10.0))
    current.mean().backward()
    gradient = model.raw_angle_sigma.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert abs(float(gradient)) > 1.0e-6


def test_v2b_zero_voltage_and_retarding_monotonicity() -> None:
    model = LastCollisionFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    va = torch.linspace(0.0, 60.0, 31)
    low = model(va, torch.zeros_like(va))
    high = model(va, torch.full_like(va, 10.0))
    assert float(low[0].detach()) == 0.0
    assert float(high[0].detach()) == 0.0
    assert torch.all(high <= low + 1.0e-6)
