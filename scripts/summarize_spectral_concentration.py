"""Locate the effective density's main lobe and compare with the known Ar I 4s range."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


AR_4S_RANGE = (11.54835442, 11.82807116)


def interval_mass(energy: np.ndarray, density: np.ndarray, low: float, high: float) -> float:
    """Integrate the piecewise-linear sampled density, including exact endpoints."""
    nodes = np.r_[low, energy[(energy > low) & (energy < high)], high]
    return float(np.trapezoid(np.interp(nodes, energy, density), nodes))


def peak_interval(energy: np.ndarray, density: np.ndarray) -> tuple[float, float]:
    """Half-height interval connected to the global density maximum."""
    mode = int(np.argmax(density))
    level = density[mode] / 2
    left = right = mode
    while left > 0 and density[left - 1] >= level:
        left -= 1
    while right < len(energy) - 1 and density[right + 1] >= level:
        right += 1
    low = float(energy[left]) if left == 0 else float(np.interp(
        level, density[left - 1:left + 1], energy[left - 1:left + 1]))
    high = float(energy[right]) if right == len(energy) - 1 else float(np.interp(
        level, density[right:right + 2][::-1], energy[right:right + 2][::-1]))
    return low, high


def summarize(evidence: Path) -> pd.DataFrame:
    reference = pd.read_csv(evidence / "distribution_reference.csv")
    keep = reference.family.isin(["C", "G1"]) & (
        reference.scope.eq("final") | reference.scope.str.startswith("final/width_"))
    rows = []
    for (scope, family, seed, unit), group in reference[keep].groupby(["scope", "family", "seed", "unit"]):
        group = group.sort_values("energy_eV")
        energy = group.energy_eV.to_numpy()
        density = group.density_per_eV.to_numpy()
        density = density / np.trapezoid(density, energy)
        low, high = peak_interval(energy, density)
        mode = float(energy[np.argmax(density)])
        rows.append({"scope": scope, "family": family, "seed": seed, "unit": unit,
            "mode_eV": mode, "peak_halfheight_low_eV": low, "peak_halfheight_high_eV": high,
            "peak_fwhm_eV": high - low, "peak_halfheight_mass": interval_mass(energy, density, low, high),
            "ar4s_low_eV": AR_4S_RANGE[0], "ar4s_high_eV": AR_4S_RANGE[1],
            "ar4s_mass": interval_mass(energy, density, *AR_4S_RANGE),
            "mode_above_ar4s_high_eV": mode - AR_4S_RANGE[1],
            "halfheight_overlaps_ar4s": bool(low <= AR_4S_RANGE[1] and high >= AR_4S_RANGE[0])})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    args = parser.parse_args()
    print(summarize(args.evidence).to_csv(index=False), end="")
