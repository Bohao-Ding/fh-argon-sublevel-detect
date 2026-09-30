import math

import torch

from fh_transport.model_v2i import FiniteApertureFranckHertz
from fh_transport.model_v2k import TruncatedGaussianGateFranckHertz


def test_v2i_energy_gate_refactor_preserves_gaussian_cdf() -> None:
    model = FiniteApertureFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    energy = torch.tensor([[[-1.0, 0.0, 1.0]]])
    barrier = torch.tensor([0.0])
    sigma = torch.tensor(1.0)
    expected = 0.5 * (1.0 + torch.erf(energy / math.sqrt(2.0)))
    actual = model.collector_energy_gate(energy, barrier, sigma)
    assert torch.allclose(actual, expected)


def test_v2k_gate_is_zero_below_barrier_and_half_gaussian_above() -> None:
    model = TruncatedGaussianGateFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    energy = torch.tensor([[[-1.0, 0.0, 1.0, 2.0]]])
    barrier = torch.tensor([0.0])
    sigma = torch.tensor(1.0)
    actual = model.collector_energy_gate(energy, barrier, sigma)
    assert torch.equal(actual[..., :2], torch.zeros_like(actual[..., :2]))
    assert torch.all(actual[..., 2:] > 0.0)
    assert float(actual[..., 3]) > float(actual[..., 2])
    assert torch.allclose(actual[..., 2], torch.erf(torch.tensor(1.0 / math.sqrt(2.0))))


def test_v2k_forward_is_finite_and_preserves_zero_supply() -> None:
    model = TruncatedGaussianGateFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(0.0, 60.0, 25)
    prediction = model(va, torch.full_like(va, 8.0))
    assert float(prediction[0].detach()) == 0.0
    assert torch.all(torch.isfinite(prediction))
    prediction.mean().backward()
    assert model.raw_gate_sigma.grad is not None
    assert torch.isfinite(model.raw_gate_sigma.grad)
