import torch

from fh_transport.model_v2f import ThresholdGaussianFranckHertz


def test_v2f_threshold_hazard_is_zero_below_threshold_and_bounded() -> None:
    model = ThresholdGaussianFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    params = model.physical_parameters()
    threshold = params["energies_eV"]
    kinetic = torch.tensor([10.0, float(threshold[0].detach()), 12.0, 20.0])
    cross = model._cross_sections(
        kinetic,
        threshold,
        params["cross_section_rise_eV"],
        params["cross_section_decay_eV"],
        params["cross_section_power"],
    ).squeeze(1)
    detached_cross = cross.detach()
    assert float(detached_cross[0]) == 0.0
    assert float(detached_cross[1]) == 0.0
    assert torch.all(cross >= 0.0)
    assert torch.all(cross <= 1.0)
    assert float(detached_cross[3]) > float(detached_cross[2])


def test_v2f_replaced_legacy_parameters_are_frozen() -> None:
    model = ThresholdGaussianFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    assert not model.raw_thermal_energy.requires_grad
    assert not model.raw_arrival_scale.requires_grad
    assert not model.raw_optical_depth.requires_grad
    assert not model.raw_cross_decay.requires_grad
    assert not model.raw_cross_power.requires_grad
    assert model.raw_injection_energy_v2f.requires_grad
    assert model.raw_optical_depth_v2f.requires_grad


def test_v2f_zero_voltage_retarding_monotonicity_and_finite_gradients() -> None:
    model = ThresholdGaussianFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    va = torch.linspace(0.0, 60.0, 31)
    low = model(va, torch.zeros_like(va))
    high = model(va, torch.full_like(va, 10.0))
    assert float(low[0].detach()) == 0.0
    assert torch.all(high <= low + 1.0e-6)
    high.mean().backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
