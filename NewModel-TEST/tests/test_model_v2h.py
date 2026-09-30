import torch

from fh_transport.model_v2g import GeneralizedSupplyFranckHertz
from fh_transport.model_v2h import ReachabilityGatedFranckHertz


def test_v2h_removes_negative_arrival_energy_tail() -> None:
    baseline = GeneralizedSupplyFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    model = ReachabilityGatedFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    model.load_state_dict(baseline.state_dict(), strict=False)
    va = torch.tensor([1.0, 2.0, 3.0])
    vr = torch.zeros_like(va)
    with torch.no_grad():
        baseline_current = baseline(va, vr)
        gated_current = model(va, vr)
    assert torch.any(baseline_current > 0.0)
    assert torch.all(gated_current == 0.0)


def test_v2h_reachability_scale_has_finite_nonzero_gradient_and_report() -> None:
    model = ReachabilityGatedFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(4.0, 30.0, 27)
    model(va, torch.zeros_like(va)).mean().backward()

    gradient = model.raw_reachability_scale_v2h.grad
    assert gradient is not None
    assert torch.isfinite(gradient)
    assert float(torch.abs(gradient)) > 0.0
    report = model.parameter_report()
    assert report["model_version"] == "v2h-reachability-gated"
    assert report["additional_trainable_parameters_vs_v2a"] == 1
