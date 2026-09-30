import torch

from fh_transport.model_v2 import _inverse_bounded
from fh_transport.model_v2g import GeneralizedSupplyFranckHertz


def _set_supply_exponent(model: GeneralizedSupplyFranckHertz, value: float) -> None:
    with torch.no_grad():
        model.raw_supply_exponent_v2g.copy_(
            torch.tensor(_inverse_bounded(value, 0.80, 1.80))
        )


def test_v2g_lower_exponent_raises_low_to_high_voltage_ratio() -> None:
    model = GeneralizedSupplyFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    va = torch.tensor([10.0, 60.0])
    vr = torch.zeros_like(va)

    _set_supply_exponent(model, 0.90)
    low_exponent_current = model(va, vr).detach()
    _set_supply_exponent(model, 1.70)
    high_exponent_current = model(va, vr).detach()

    low_ratio = low_exponent_current[0] / low_exponent_current[1]
    high_ratio = high_exponent_current[0] / high_exponent_current[1]
    assert torch.isfinite(low_ratio)
    assert torch.isfinite(high_ratio)
    assert float(low_ratio) > float(high_ratio)


def test_v2g_supply_exponent_has_finite_nonzero_gradient_and_report() -> None:
    model = GeneralizedSupplyFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    va = torch.linspace(5.0, 60.0, 12)
    prediction = model(va, torch.zeros_like(va))
    prediction.mean().backward()

    gradient = model.raw_supply_exponent_v2g.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert float(torch.abs(gradient)) > 0.0
    report = model.parameter_report()
    assert report["model_version"] == "v2g-generalized-supply"
    assert report["trainable_parameter_count"] == sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
