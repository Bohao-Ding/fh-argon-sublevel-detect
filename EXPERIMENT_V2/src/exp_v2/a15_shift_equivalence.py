"""A15: test what the variable-phase result can identify.

At a fixed period, a sine/cosine fit per band is exactly one periodic
waveform with a band-specific amplitude and voltage shift. This diagnostic
checks that equivalence on every held-out fold and reports the implied
relative shifts. It cannot determine whether those shifts are physical or
arose during acquisition.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .common import load_pivot, original_blocks, timed, write_outputs

PERIOD = 11.55
BANDS = ("0-4", "4-6", "6-8", "8-10", ">10")


def _shift_form(va: np.ndarray, predicted_z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Recover trend, amplitude and phase from a rank-2 prediction."""
    x = (va - 0.5 * (va.min() + va.max())) / (0.5 * (va.max() - va.min()))
    omega_v = 2.0 * np.pi * va / PERIOD
    design = np.column_stack((np.ones_like(x), x, x**2, x**3, np.sin(omega_v), np.cos(omega_v)))
    coef = np.linalg.lstsq(design, predicted_z, rcond=None)[0]
    amplitude = np.hypot(coef[-2], coef[-1])
    phase = np.arctan2(coef[-1], coef[-2])
    reconstructed = design[:, :4] @ coef[:4] + amplitude[None, :] * np.sin(omega_v[:, None] + phase[None, :])
    return reconstructed, amplitude, phase


def run() -> dict:
    from fh_retry.analysis import _predict_fold
    from fh_retry.phase2 import _retarding_fractions

    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        va, fractions, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
        rows = []
        for block, (low, high) in enumerate(original_blocks(), start=1):
            test = (va >= low) & (va < high)
            train = ~test
            _, rank1_z, observed_z = _predict_fold(va, fractions, train, test, period=PERIOD, rank=1, trend_degree=3)
            _, rank2_z, _ = _predict_fold(va, fractions, train, test, period=PERIOD, rank=2, trend_degree=3)
            # Request the same train-fitted prediction on the full voltage grid.
            _, rank2_all, _ = _predict_fold(va, fractions, train, np.ones_like(train, dtype=bool), period=PERIOD, rank=2, trend_degree=3)
            shifted_all, _, _ = _shift_form(va, rank2_all)
            e1 = float(np.sqrt(np.mean((rank1_z - observed_z) ** 2)))
            e2 = float(np.sqrt(np.mean((rank2_z - observed_z) ** 2)))
            rows.append({
                "block": block,
                "rank1_rmse": e1,
                "rank2_rmse": e2,
                "shift_form_rmse": float(np.sqrt(np.mean((shifted_all[test] - observed_z) ** 2))),
                "rank2_gain": (e1 - e2) / e1,
                "max_prediction_difference": float(np.max(np.abs(shifted_all[test] - rank2_z))),
            })

        _, rank2_all, _ = _predict_fold(va, fractions, np.ones(va.size, dtype=bool), np.ones(va.size, dtype=bool), period=PERIOD, rank=2, trend_degree=3)
        shifted_all, amplitude, phase = _shift_form(va, rank2_all)
        unwrapped = np.unwrap(phase)
        relative_shift = -(unwrapped - unwrapped[0]) * PERIOD / (2.0 * np.pi)
        phase_table = pd.DataFrame({
            "band": BANDS,
            "amplitude_standardized": amplitude,
            "phase_rad_unwrapped": unwrapped,
            "relative_shift_V_mod_period": relative_shift,
            "adjacent_shift_V_mod_period": np.diff(relative_shift, prepend=np.nan),
        })
        folds = pd.DataFrame(rows)
        max_error = float(max(folds["max_prediction_difference"].max(), np.max(np.abs(shifted_all - rank2_all))))
        summary = {
            "experiment": "A15_shift_equivalence",
            "dataset_sha256": audit["sha256"],
            "numerical_identity_tolerance": 1e-10,
            "max_prediction_difference": max_error,
            "identity_verified": bool(max_error < 1e-10),
            "steady_rank2_gain_median": float(folds.loc[folds["block"] > 1, "rank2_gain"].median()),
            "adjacent_shift_V_mod_period": phase_table["adjacent_shift_V_mod_period"].iloc[1:].tolist(),
            "interpretation": "The same fitted predictions admit a single-period, band-shifted form. Neither fit nor blocked CV can distinguish response physics from an unrecorded acquisition shift.",
            "timings": timings,
        }
    out_dir = write_outputs("a15_shift_equivalence", {"folds": folds, "phase_shifts": phase_table}, summary)
    return {"summary": summary, "out_dir": out_dir}
