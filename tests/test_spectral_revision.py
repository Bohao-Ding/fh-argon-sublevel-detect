from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from sublevel_detect import cli
from sublevel_detect.spectral_pipeline import training_curves
from sublevel_detect.spectral_revision import (RevisedSpectrum, concentration_status,
    quantiles, revision_grid, spectrum_summary)
from sublevel_detect.spectral_revision_pipeline import Experiment, choose, fit_unit, load_fit, settings


def curves():
    net = RevisedSpectrum(4, "H1", excitation=11.65)
    va = np.arange(161, dtype=np.float32) * 0.5
    result = []
    for i, vr in enumerate((0, 4, 6, 8, 10)):
        with torch.no_grad():
            ip = net(torch.tensor(va), torch.full((161,), vr), nuisance_mode="neutral").numpy().astype(np.float32)
        result.append(dict(curve_id=i, curve_idx=i, Va=va, Vr=float(vr), Ip=ip))
    return result


@pytest.mark.parametrize("phase", [False, True])
@pytest.mark.parametrize("family", ["H1", "G1", "C"])
def test_prediction_is_invariant_to_subsets_and_batches(family, phase):
    net = RevisedSpectrum(4, family, phase_response=phase, seed=1)
    va = torch.arange(161, dtype=torch.float64) * 0.5
    vr = torch.full_like(va, 6)
    with torch.no_grad():
        full = net(va, vr, nuisance_mode="neutral")
        part = net(va[:81], vr[:81], nuisance_mode="neutral")
        single = net(va[40:41], vr[40:41], nuisance_mode="neutral")
    assert torch.max(torch.abs(full[:81] - part)) < 1e-12
    assert torch.abs(full[40] - single[0]) < 1e-12


def test_continuous_and_atomic_quantiles_have_distinct_correct_cdfs():
    e = np.linspace(9, 16, 701)
    d = np.exp(-0.5 * ((e - 11.65) / 0.15) ** 2)
    q = quantiles(e, None, [0.5], density=d)
    assert abs(q[0] - 11.65) < 1e-10
    assert quantiles([10, 12], [0.8, 0.2], [0.5, 0.9]).tolist() == [10, 12]


def test_normalization_and_fine_knots_are_separate_from_quadrature():
    for step in (0.01, 0.005):
        e, q, basis, _, _, _ = revision_grid(step=step)
        assert np.all(basis >= 0)
        assert np.max(np.abs(q @ basis - 1)) < 1e-12
    e, q, b, _, _, _ = revision_grid(knot_step=0.1, step=0.005)
    means = q @ (b * e[:, None])
    sd = np.sqrt(q @ (b * (e[:, None] - means) ** 2))
    assert sd[np.argmin(abs(means - 11.6))] < 0.06


def test_flat_and_boundary_spectra_are_not_qualified_peaks():
    uniform = RevisedSpectrum(4, "C")
    assert spectrum_summary(uniform)["peak_status"] in ("diffuse_spectrum", "boundary_peak")
    narrow = RevisedSpectrum(4, "G1", seed=1)
    assert spectrum_summary(narrow)["peak_status"] == "concentrated"
    a = spectrum_summary(narrow)
    b = dict(a, mode_eV=a["mode_eV"] + 0.8)
    assert concentration_status([a, b])["status"] == "unstable_concentration"
    assert concentration_status([a, spectrum_summary(uniform)])["status"] == "no_robust_concentration"


def test_phase_shift_changes_periodic_positions_with_correct_sign():
    net = RevisedSpectrum(4, "H1", excitation=11.65, phase_response=True, seed=1)
    beta = float(net.phys_params()["phase_beta"].detach())
    va = torch.arange(161, dtype=torch.float64) * 0.5
    with torch.no_grad():
        a = net.kernel_components(va, torch.full_like(va, 6))["weighted_dip"]
        b = net.kernel_components(va + 2 * beta, torch.full_like(va, 8))["weighted_dip"]
    assert torch.max(abs(a - b)) < 1e-12


def test_corrective_penalty_is_centered_on_physical_zero():
    net = RevisedSpectrum(4, "H1")
    neutral = {"raw_offset": (0, -0.5, 0.8), "raw_slope": (0, -0.02, 0.04),
               "raw_vr_contrast": (0, -0.25, 0.35), "raw_late_contrast": (0, -0.35, 0.25),
               "raw_vr_late_contrast": (0, -0.20, 0.45), "raw_vr_late_baseline": (0, -0.05, 0.03)}
    from sublevel_detect.model import inv_bounded
    with torch.no_grad():
        for name, args in neutral.items():
            getattr(net, name).fill_(inv_bounded(*args))
        net.raw_high_energy_loss_strength.fill_(-50)
    assert float(net.common_penalty().detach()) < 1e-12


def test_training_identity_does_not_read_stress_or_heldout(tmp_path):
    data = curves()
    cfg = settings("smoke")
    exp = Experiment(tmp_path, cfg)
    original = exp.task(training_curves(data, heldout=6), "C", 0)
    changed = copy.deepcopy(data)
    for c in changed:
        if c["Vr"] in (6, 10):
            c["Ip"] += 100
    other = exp.task(training_curves(changed, heldout=6), "C", 0)
    assert original["identity"] == other["identity"]
    assert original["spec"] == other["spec"]


def test_one_se_prefers_no_phase_and_smoothing_when_eligible():
    rows = [dict(phase_response=phase, **{"lambda": penalty}, nrmse=.1 + .001 * phase)
            for phase in (False, True) for penalty in (0, 1e-6) for _ in range(3)]
    selected, _ = choose(rows)
    assert selected == {"phase_response": False, "lambda": 1e-6}


@pytest.mark.parametrize("freeze_kernel", [False, True])
def test_revision_fit_receipt_and_source_identity(tmp_path, freeze_kernel):
    exp = Experiment(tmp_path, settings("smoke"))
    fixed = {}
    if freeze_kernel:
        reference = RevisedSpectrum(4, "H1")
        fixed = {name: p.detach().numpy().tolist() for name, p in reference.named_parameters()
                 if name.startswith("raw_")}
    task = exp.task(training_curves(curves()), "C", 0, penalty=1e-6, fixed_parameters=fixed)
    receipt = fit_unit(task)
    net = load_fit(task["directory"])
    assert np.isfinite(spectrum_summary(net)["mean_eV"])
    assert receipt["status"] in ("stationary", "iteration_limit")
    assert receipt["simplex_kkt"] >= 0
    altered = dict(task, identity="wrong")
    with pytest.raises(ValueError, match="incompatible_revision_identity"):
        fit_unit(altered)
    path = tmp_path / "units" / task["identity"][:20] / "fit.json"
    old = json.loads(path.read_text())
    old["spec"]["schema"] = "unrelated"
    path.write_text(json.dumps(old))
    with pytest.raises(ValueError, match="incompatible_revision_schema"):
        load_fit(path.parent)


def test_cli_keeps_historical_paths_and_exposes_revision():
    parser = cli.build_parser()
    assert parser.parse_args([]).experiment == "legacy"
    assert parser.parse_args(["--experiment", "spectrum"]).experiment == "spectrum"
    assert parser.parse_args(["--experiment", "spectrum-revised"]).experiment == "spectrum-revised"
