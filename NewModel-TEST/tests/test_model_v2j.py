import torch

from fh_transport.metrics import apply_v2j_single_level_gate
from fh_transport.model_v2i import FiniteApertureFranckHertz
from fh_transport.model_v2j import CollisionFieldEnsembleFranckHertz


def test_v2j_collision_quadrature_is_normalized() -> None:
    model = CollisionFieldEnsembleFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    assert model.collision_field_nodes.numel() == 5
    assert torch.allclose(
        torch.sum(model.collision_field_weights), torch.tensor(1.0), atol=1.0e-6
    )


def test_v2j_negligible_spread_recovers_v2i() -> None:
    baseline = FiniteApertureFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    model = CollisionFieldEnsembleFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    model.load_state_dict(baseline.state_dict(), strict=False)
    with torch.no_grad():
        model.raw_collision_field_spread_v2j.fill_(-20.0)
        va = torch.linspace(0.0, 60.0, 25)
        vr = torch.full_like(va, 8.0)
        expected = baseline(va, vr)
        actual = model(va, vr)
    assert torch.allclose(actual, expected, atol=2.0e-6, rtol=2.0e-6)


def test_v2j_preserves_zero_voltage_and_has_finite_new_gradients() -> None:
    model = CollisionFieldEnsembleFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(0.0, 60.0, 25)
    prediction = model(va, torch.full_like(va, 8.0))
    assert float(prediction[0].detach()) == 0.0
    prediction.mean().backward()
    for parameter in (
        model.raw_collision_field_spread_v2j,
        model.raw_collision_spread_decay_v2j,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad)
        assert float(torch.abs(parameter.grad)) > 0.0


def test_v2j_gate_adds_precision_and_early_high_retarding_checks() -> None:
    summary = {
        "mean_curve_nrmse": 0.04,
        "cutoff_mae_V": 0.4,
        "max_cutoff_error_V": 1.5,
        "zero_region_mae_uA": 0.001,
        "peak_position_mae_V": 0.4,
        "trough_position_mae_V": 0.4,
        "peak_recall": 1.0,
        "trough_recall": 1.0,
        "peak_drift_rmse_V": 0.4,
        "vr0_spurious_feature_count": 0.0,
        "vr0_low_voltage_mae_uA": 0.05,
        "late_trough_current_mae_uA": 0.20,
        "late_voltage_mae_uA": 0.09,
        "trough_precision": 0.90,
        "early_high_retarding_mae_uA": 0.08,
    }
    gate = apply_v2j_single_level_gate({"summary": summary})
    failed = {check["metric"] for check in gate["checks"] if not check["passed"]}
    assert failed == {"trough_precision", "early_high_retarding_mae_uA"}
