import torch

from fh_transport.model import ThinLayerFranckHertz


def test_collision_master_equation_conserves_probability() -> None:
    model = ThinLayerFranckHertz(n_levels=1, n_layers=8, max_collisions=5, quadrature_order=6)
    va = torch.tensor([0.0, 12.0, 30.0, 80.0])
    energy = model.excitation_parameters()[0][0]
    probability = model.collision_probabilities(va, energy)
    assert torch.all(probability >= 0.0)
    assert torch.allclose(probability.sum(dim=1), torch.ones(4), atol=1.0e-5)


def test_zero_voltage_is_zero_and_retarding_voltage_reduces_current() -> None:
    model = ThinLayerFranckHertz(n_levels=1, n_layers=8, max_collisions=5, quadrature_order=6)
    va = torch.tensor([0.0, 10.0, 20.0, 40.0])
    low = model(va, torch.zeros_like(va))
    high = model(va, torch.full_like(va, 10.0))
    assert float(low[0].detach()) == 0.0
    assert float(high[0].detach()) == 0.0
    assert torch.all(high <= low + 1.0e-6)


def test_forward_gradients_are_finite() -> None:
    model = ThinLayerFranckHertz(n_levels=2, n_layers=6, max_collisions=4, quadrature_order=5)
    loss = model(torch.linspace(0.0, 40.0, 21), torch.full((21,), 6.0)).mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
