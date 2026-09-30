import torch

from fh_transport.model_v2d import EnergyDependentScatteringFranckHertz


def test_v2d_scattering_scale_changes_high_voltage_transmission() -> None:
    model = EnergyDependentScatteringFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(50.0, 80.0, 21)
    vr = torch.full_like(va, 10.0)
    with torch.no_grad():
        model.raw_scattering_scale.fill_(-10.0)
        forward = model(va, vr)
        model.raw_scattering_scale.fill_(10.0)
        isotropic = model(va, vr)
    assert torch.mean(forward) > torch.mean(isotropic)


def test_v2d_scattering_scale_has_finite_gradient() -> None:
    model = EnergyDependentScatteringFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(45.0, 80.0, 25)
    model(va, torch.full_like(va, 8.0)).mean().backward()
    gradient = model.raw_scattering_scale.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert abs(float(gradient)) > 1.0e-7
