from __future__ import annotations

import copy

import numpy as np
import pytest

from sublevel_detect.affine_readout import apply_readout, fit_readout, select_readout
from sublevel_detect.spectral_pipeline import training_curves


def sample():
    x = np.linspace(0, 80, 161, dtype=np.float32)
    predictions = [0.2 + x / 40 for _ in range(4)]
    curves = [dict(Vr=vr, Va=x, Ip=1.04 * y + 0.03)
              for vr, y in zip((0., 4., 6., 8.), predictions)]
    return curves, predictions


def test_recovers_shared_affine_readout():
    curves, predictions = sample()
    calibration = fit_readout(curves, predictions, "global")
    assert calibration["coefficients"] == pytest.approx([1.04, 0.03], abs=1e-7)
    assert apply_readout(predictions[0], 4, calibration) == pytest.approx(curves[0]["Ip"])


def test_positive_current_affine_does_not_move_unclipped_extrema():
    values = 1 + np.cos(np.arange(161) * 0.2)
    calibrated = apply_readout(values, 0, dict(mode="global", coefficients=[1.1, 0.2]))
    assert np.argmax(values) == np.argmax(calibrated)
    assert np.argmin(values) == np.argmin(calibrated)


def test_voltage_affine_is_degenerate_in_periodic_kernel():
    va = np.linspace(0, 80, 161)
    energy, width, phase, gain, offset = 12.04, 2.28, 0.35, 0.97, 1.0
    def response(voltage, e, w, phi):
        cycles = (voltage + phi) / e
        return np.exp(-0.5 * ((cycles - np.round(cycles)) * e / w) ** 2)
    actual = response(gain * va + offset, energy, width, phase)
    equivalent = response(va, energy / gain, width / gain, (offset + phase) / gain)
    assert np.max(np.abs(actual - equivalent)) < 1e-14
    # A voltage origin changes every peak equally; it does not change the period.
    assert np.diff((np.arange(7) * energy - phase - offset) / gain) == pytest.approx(energy / gain)


def test_holdout_and_stress_observations_cannot_change_readout():
    curves, predictions = sample()
    stress = copy.deepcopy(curves[-1]); stress["Vr"] = 10.
    original = curves + [stress]
    changed = copy.deepcopy(original)
    changed[0]["Ip"] += 100
    changed[-1]["Ip"][:] = -999
    a, b = training_curves(original, 0), training_curves(changed, 0)
    assert fit_readout(a, predictions[1:], "global") == fit_readout(b, predictions[1:], "global")


def test_one_se_rule_can_reject_or_retain_common_calibration():
    rows = [dict(mode=mode, nrmse=value) for mode, values in
        (("none", (.1, .11, .12)), ("global", (.099, .11, .119))) for value in values]
    assert select_readout(rows)[0] == "none"
    for row in rows:
        if row["mode"] == "global":
            row["nrmse"] *= .5
    assert select_readout(rows)[0] == "global"


def test_vr_trend_is_recovered_but_not_selected_as_common_calibration():
    curves, predictions = sample()
    for curve, values in zip(curves, predictions):
        z = (curve["Vr"] - 4) / 4
        curve["Ip"] = (1 + .1 * z) * values + .02 * z
    calibration = fit_readout(curves, predictions, "linear_vr_diagnostic")
    assert calibration["coefficients"] == pytest.approx([1, 0, .1, .02], abs=1e-7)
    with pytest.raises(ValueError, match="unknown_readout_mode"):
        fit_readout(curves, predictions, "voltage")
