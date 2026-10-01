from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from sublevel_detect import cli, model
from sublevel_detect.spectral_models import (EffectiveSpectrum, LocalSpectrum,
    NIST_4S, main_peak_window, spline_grid, refine_measure)
from sublevel_detect.spectral_pipeline import (Experiment, curves_hash, fit_unit,
    load_curves, load_fit, one_se_selection, predict, quadrature_check, residual_sample,
    run, settings, training_curves, training_initialization)


def curves():
    va = np.arange(161, dtype=np.float32) * 0.5
    return [dict(curve_id=i + 1, curve_idx=i, Va=va, Vr=float(vr),
                 Ip=(0.5 + va / 50 + 0.3 * np.cos(va * 2 * np.pi / 11.6) - 0.01 * vr).astype(np.float32))
            for i, vr in enumerate((0, 4, 6, 8, 10))]


def peaked_model():
    net = EffectiveSpectrum(4, "C")
    x, quad, basis, _, _ = spline_grid()
    centers = (quad * x) @ basis / (quad @ basis)
    logits = -0.5 * ((centers - 11.65) / 0.25) ** 2
    with torch.no_grad():
        net.spectral_logits.copy_(torch.tensor(logits[:-1] - logits[-1], dtype=torch.float32))
    return net


@pytest.mark.parametrize("family", ["C", "G1", "H1"])
def test_nonnegative_normalized_measure(family):
    net = EffectiveSpectrum(4, family, seed=2)
    levels = net.level_params()
    assert torch.all(levels["weights"] >= 0)
    assert float(levels["weights"].detach().sum()) == pytest.approx(1, abs=1e-6)
    assert net.spline_basis.shape == (351, 31)
    if family != "H1":
        assert float(net.density().detach() @ net.quadrature) == pytest.approx(1, abs=1e-6)


def test_discrete_limit_matches_existing_kernel():
    reference = model.PoissonRateFHCoreMultiLevel(4, n_levels=1, V_exc_init=11.65)
    candidate = EffectiveSpectrum(4, "H1", excitation=11.65)
    va = torch.arange(161) * 0.5
    vr = torch.full_like(va, 6)
    assert torch.max(torch.abs(reference(va, vr) - candidate(va, vr))) < 1e-6


@pytest.mark.parametrize("family", ["d1", "g1", "d2", "d3", "d4", "h4s", "equal4"])
def test_local_replacement_keeps_outside_mass_and_width(family):
    coarse = peaked_model()
    window = main_peak_window([coarse])
    net = LocalSpectrum(coarse, family, window)
    assert float(net.outside_mass.sum() + net.inside_mass) == pytest.approx(1, abs=1e-6)
    assert torch.all((net.outside_energy <= window["lo"]) | (net.outside_energy >= window["hi"]))
    before = net.outside_mass.clone()
    for name, parameter in net.named_parameters():
        if name.startswith("local_"):
            parameter.data.add_(0.2)
    assert torch.equal(before, net.outside_mass)
    assert not net.raw_width.requires_grad
    assert float(net.level_params()["weights"].detach().sum()) == pytest.approx(1, abs=1e-6)
    energy, mass = net.local_measure()
    assert energy.min() >= window["lo"] and energy.max() <= window["hi"]
    assert float(mass.detach().sum()) == pytest.approx(float(net.inside_mass), abs=1e-6)
    if family.startswith("d") and len(energy) > 1:
        assert torch.all(torch.diff(energy) >= 0.02 - 1e-6)


def test_window_mass_uses_half_endpoint_weights():
    coarse = EffectiveSpectrum(4, "C")
    window = dict(status="eligible", center=12.0, lo=11.5, hi=12.5)
    net = LocalSpectrum(coarse, "d1", window)
    assert float(net.inside_mass) == pytest.approx(1 / 7, abs=1e-7)
    assert float(net.outside_mass.sum()) == pytest.approx(6 / 7, abs=1e-7)


@pytest.mark.parametrize("family", ["d1", "g1", "d2", "d3", "d4", "h4s", "equal4"])
def test_local_quadrature_converges_with_frozen_mass(family):
    coarse = peaked_model()
    net = LocalSpectrum(coarse, family, main_peak_window([coarse]))
    fine = refine_measure(net, 0.01)
    assert torch.equal(net.inside_mass, fine.inside_mass)
    assert max(np.max(np.abs(a - b)) for a, b in zip(predict(net, training_curves(curves())), predict(fine, training_curves(curves())))) < 1e-4


def test_mixture_kernel_is_linear_in_response_weights():
    coarse = peaked_model()
    net = LocalSpectrum(coarse, "d2", main_peak_window([coarse]))
    va, vr = torch.arange(161) * 0.5, torch.zeros(161)
    levels = net.level_params()
    p = net.phys_params()
    phase = (va[:, None] + p["phase"]) / (levels["energies"][None, :] + 1e-6)
    residual = (phase - phase.round()) * levels["energies"][None, :]
    expected = (torch.exp(-0.5 * (residual / (p["width"] + 1e-6)) ** 2) * levels["weights"]).sum(1)
    assert torch.allclose(net.kernel_components(va, vr)["weighted_dip"], expected, atol=1e-7)


def test_shared_axis_optimization_preserves_predictions_and_gradients():
    fast = peaked_model()
    reference = copy.deepcopy(fast)
    va = torch.stack([torch.arange(161) * 0.5] * 4)
    vr = torch.stack([torch.full((161,), float(v)) for v in (0, 4, 6, 8)])
    values = fast(va, vr, curve_idx=torch.arange(4))
    expected = torch.stack([reference(va[i], vr[i], curve_idx=i) for i in range(4)])
    assert torch.allclose(values, expected, atol=1e-7)
    values.square().mean().backward()
    expected.square().mean().backward()
    for name in ("raw_width", "raw_phase", "spectral_logits"):
        assert torch.allclose(getattr(fast, name).grad, getattr(reference, name).grad, atol=1e-7, rtol=1e-5)


def test_unqualified_peak_and_outside_nist_are_explicit():
    coarse = EffectiveSpectrum(4, "C")
    assert main_peak_window([coarse])["status"] == "no_interior_main_peak"
    with pytest.raises(ValueError, match="no_interior_main_peak"):
        LocalSpectrum(coarse, "d1", main_peak_window([coarse]))
    with pytest.raises(ValueError, match="outside_current_main_peak_candidates"):
        LocalSpectrum(peaked_model(), "h4s", dict(status="eligible", center=14, lo=13.5, hi=14.5))


def test_nist_and_equal_spacing_have_matched_span_and_complexity():
    coarse = peaked_model()
    window = main_peak_window([coarse])
    nist, equal = [LocalSpectrum(coarse, family, window) for family in ("h4s", "equal4")]
    assert float(torch.diff(nist.template[[0, -1]])) == pytest.approx(NIST_4S[-1] - NIST_4S[0], abs=1e-6)
    assert torch.equal(nist.template[[0, -1]], equal.template[[0, -1]])
    assert sum(p.numel() for p in nist.parameters() if p.requires_grad) == sum(p.numel() for p in equal.parameters() if p.requires_grad)


@pytest.mark.parametrize("width", [1.0, 2.0, 3.0])
def test_integral_grid_convergence(width):
    coarse = peaked_model()
    coarse.raw_width.data.fill_(model.inv_bounded(width, 0.25, 5))
    fine = EffectiveSpectrum(4, "C", step=0.01)
    with torch.no_grad():
        for name, p in fine.named_parameters():
            p.copy_(dict(coarse.named_parameters())[name])
    assert max(np.max(np.abs(a - b)) for a, b in zip(predict(coarse, training_curves(curves())), predict(fine, training_curves(curves())))) < 1e-4


def test_training_and_initialization_do_not_read_stress_or_holdout(tmp_path):
    original = curves()
    changed = copy.deepcopy(original)
    changed[0]["Ip"] += 100
    changed[-1]["Ip"][:] = -999
    a, b = training_curves(original, 0), training_curves(changed, 0)
    assert curves_hash(a) == curves_hash(b)
    assert training_initialization(a) == training_initialization(b)
    exp = Experiment(tmp_path, settings("smoke"), 0.02, "cpu")
    ta, tb = exp.task(a, "C", 0, 1e-4), exp.task(b, "C", 0, 1e-4)
    assert ta["identity"] == tb["identity"]
    first = fit_unit(ta)
    ta["directory"] = str(tmp_path / "independent")
    second = fit_unit(ta)
    assert first["best_objective"] == second["best_objective"]
    for name, value in load_fit(ta["directory"]).state_dict().items():
        assert torch.equal(value, load_fit(tb["directory"]).state_dict()[name])
    assert all(row["passed"] for row in quadrature_check([tb], a, 0.02))


def test_one_se_rule_prefers_smoothing_and_keeps_matched_ties():
    rows = [dict(fold=i, **{"lambda": penalty}, nrmse=score)
            for penalty in (1e-6, 1e-4, 1e-2) for i, score in enumerate((0.1, 0.2, 0.3))]
    assert one_se_selection(rows, kind="lambda")[0] == [1e-2]
    rows = [dict(fold=i, family=family, nrmse=score + offset)
            for family, offset in (("d1", 0.5), ("g1", 0.5), ("h4s", 0), ("equal4", 0), ("d4", 0))
            for i, score in enumerate((0.1, 0.2, 0.3))]
    assert one_se_selection(rows, kind="local")[0] == ["equal4", "h4s"]
    assert one_se_selection(rows[:-1], kind="local", expected_folds=4)[0] == []


def test_incompatible_checkpoints_and_missing_curve_fail_explicitly(tmp_path):
    (tmp_path / "fit.json").write_text(json.dumps({"spec": {"schema": "legacy"}}))
    with pytest.raises(ValueError, match="incompatible_checkpoint_schema"):
        load_fit(tmp_path)
    exp = Experiment(tmp_path / "units", settings("smoke"), 0.02, "cpu")
    task = exp.task(training_curves(curves()), "H1", 0)
    fit_unit(task)
    task["identity"] = "wrong"
    with pytest.raises(ValueError, match="incompatible_unit_identity"):
        fit_unit(task)
    import pandas as pd
    rows = [dict(curve_id=c["curve_id"], Va=float(v), Vr=c["Vr"], IuA=float(y))
            for c in curves()[:-1] for v, y in zip(c["Va"], c["Ip"])]
    pd.DataFrame(rows).to_csv(tmp_path / "missing.csv", index=False)
    with pytest.raises(ValueError, match="missing_or_unexpected_retarding_curve"):
        run(mode="smoke", input_path=tmp_path / "missing.csv", output_root=tmp_path / "bad")


def test_resampling_is_repeatable_and_centered():
    x = np.arange(161, dtype=float)
    a = residual_sample(x, np.random.default_rng(7))
    b = residual_sample(x + 10, np.random.default_rng(7))
    assert np.array_equal(a, b)
    assert a.shape == x.shape


def test_cli_remains_legacy_by_default():
    assert cli.build_parser().parse_args([]).experiment == "legacy"
    assert cli.build_parser().parse_args(["--experiment", "spectrum"]).experiment == "spectrum"
