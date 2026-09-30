import torch

from fh_transport.model_v2c import IsotropicLastCollisionFranckHertz


def test_v2c_scattering_is_independent_of_legacy_angle_width() -> None:
    model = IsotropicLastCollisionFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    va = torch.linspace(30.0, 70.0, 17)
    vr = torch.full_like(va, 8.0)
    with torch.no_grad():
        model.raw_angle_sigma.fill_(-8.0)
        narrow = model(va, vr)
        model.raw_angle_sigma.fill_(8.0)
        wide = model(va, vr)
    assert not model.raw_angle_sigma.requires_grad
    assert torch.allclose(narrow, wide, atol=1.0e-7)


def test_v2c_reports_only_effective_trainable_parameters() -> None:
    model = IsotropicLastCollisionFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    expected = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    report = model.parameter_report()
    assert report["trainable_parameter_count"] == expected
    assert report["post_collision_angular_distribution"] == "p(mu)=2mu, 0<=mu<=1"
