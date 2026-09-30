import torch

from fh_transport.model_v2i import FiniteApertureFranckHertz


def test_v2i_aperture_does_not_change_uncollided_state_gate() -> None:
    model = FiniteApertureFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(20.0, 60.0, 9)
    vr = torch.full_like(va, 6.0)
    with torch.no_grad():
        model.raw_aperture_sigma_v2i.fill_(-8.0)
        narrow = model.collector_state_transmission(va, vr)
        model.raw_aperture_sigma_v2i.fill_(8.0)
        wide = model.collector_state_transmission(va, vr)
    assert torch.allclose(narrow[:, :, 0], wide[:, :, 0], atol=1.0e-6, rtol=1.0e-6)


def test_v2i_narrow_aperture_reduces_collided_state_transmission() -> None:
    model = FiniteApertureFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(45.0, 75.0, 13)
    vr = torch.full_like(va, 8.0)
    with torch.no_grad():
        model.raw_aperture_sigma_v2i.fill_(-8.0)
        narrow = model.collector_state_transmission(va, vr)[:, :, 1:]
        model.raw_aperture_sigma_v2i.fill_(8.0)
        wide = model.collector_state_transmission(va, vr)[:, :, 1:]
    assert torch.mean(narrow) < torch.mean(wide)


def test_v2i_aperture_has_finite_nonzero_gradient_and_report() -> None:
    model = FiniteApertureFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(35.0, 80.0, 31)
    model(va, torch.full_like(va, 8.0)).mean().backward()
    gradient = model.raw_aperture_sigma_v2i.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert float(torch.abs(gradient)) > 0.0
    report = model.parameter_report()
    assert report["model_version"] == "v2i-finite-angular-aperture"
    assert report["additional_trainable_parameters_vs_v2a"] == 2
