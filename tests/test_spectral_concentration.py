import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "spectral_concentration", Path(__file__).parents[1] / "scripts/summarize_spectral_concentration.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_interval_uses_fractional_endpoints():
    energy = np.array([0., 1., 2.])
    density = np.array([0., 1., 0.])
    assert module.interval_mass(energy, density, 0.25, 1.75) == pytest.approx(0.9375)


def test_halfheight_is_connected_to_main_peak():
    energy = np.arange(7, dtype=float)
    density = np.array([0., 2., 0., 4., 0., 3., 0.])
    assert module.peak_interval(energy, density) == pytest.approx((2.5, 3.5))
