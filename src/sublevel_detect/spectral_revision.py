"""Revised effective spectrum; historical spectrum sources remain byte-frozen."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.interpolate import BSpline
from torch import nn
from torch.nn import functional as F

from .model import inv_bounded
from .spectral_models import EffectiveSpectrum, NIST_4S, simplex


@lru_cache(maxsize=12)
def revision_grid(domain=(9.0, 16.0), knot_step=0.1, step=0.01):
    x = np.linspace(*domain, round((domain[1] - domain[0]) / step) + 1)
    q = np.full(len(x), x[1] - x[0])
    q[[0, -1]] *= 0.5
    interior = np.linspace(*domain, round((domain[1] - domain[0]) / knot_step) + 1)[1:-1]
    knots = np.r_[np.repeat(domain[0], 4), interior, np.repeat(domain[1], 4)]
    basis = BSpline.design_matrix(x, knots, 3).toarray()
    integrals = q @ basis
    second = BSpline(knots, np.eye(basis.shape[1]), 3).derivative(2)(x)
    centers = np.array([np.mean(knots[i + 1:i + 4]) for i in range(len(integrals))])
    return x, q, basis / integrals, second / integrals, integrals, centers


class RevisedSpectrum(EffectiveSpectrum):
    def __init__(self, n_curves, family, *, seed=0, excitation=11.5, step=0.01,
                 knot_step=0.1, domain=(9.0, 16.0), phase_response=False,
                 normalization_voltage=80.0, fixed_width=None):
        super().__init__(n_curves, family, seed=seed, excitation=excitation,
                         step=step, fixed_width=fixed_width)
        self.domain, self.knot_step = tuple(domain), float(knot_step)
        self.phase_response = bool(phase_response)
        self.register_buffer("normalization_voltage", torch.tensor(float(normalization_voltage)))
        x, q, basis, second, integrals, centers = revision_grid(self.domain, knot_step, step)
        for name, values in (("energy_grid", x), ("quadrature", q),
                             ("spline_basis", basis), ("spline_second", second)):
            self._buffers[name] = torch.tensor(values, dtype=torch.float64)
        mu = float(np.clip(excitation + (0, -0.4, 0.4)[seed % 3],
                           self.domain[0] + 0.01, self.domain[1] - 0.01))
        if family in ("H1", "G1"):
            self.spectral_mu = nn.Parameter(torch.tensor(inv_bounded(mu, *self.domain)))
        if family == "G1":
            self.spectral_sigma.data.fill_(inv_bounded((0.5, 0.15, 1.0)[seed % 3], 0.05, 2.0))
        if family == "C":
            profile = np.ones_like(centers) if seed % 3 == 0 else np.exp(
                -0.5 * ((centers - mu) / (0.25 if seed % 3 == 1 else 1.0)) ** 2) + 1e-8
            weights = profile * integrals
            self.spectral_logits = nn.Parameter(torch.tensor(np.log(weights[:-1] / weights[-1])))
        self.raw_phase_vr = nn.Parameter(torch.tensor(inv_bounded(
            (0.0, 0.4, -0.4)[seed % 3], -1.0, 1.0)), requires_grad=self.phase_response)
        if fixed_width is None:
            self.raw_width.data.fill_(inv_bounded((1.25, 0.7, 3.0)[seed % 3], 0.25, 5.0))
        self.double()

    def density(self, spectrum_weights=None):
        if self.family == "C":
            return self.spline_basis @ (simplex(self.spectral_logits) if spectrum_weights is None else spectrum_weights)
        if self.family == "G1":
            mu = self.bounded(self.spectral_mu, *self.domain)
            sigma = self.bounded(self.spectral_sigma, 0.05, 2.0)
            values = torch.exp(-0.5 * ((self.energy_grid - mu) / sigma) ** 2)
            return values / (values @ self.quadrature)
        raise ValueError("A delta energy has masses, not an ordinary density")

    def level_params(self, spectrum_weights=None):
        if self.family == "H1":
            energies = self.bounded(self.spectral_mu, *self.domain).view(1)
            weights = energies.new_ones(1)
        else:
            energies = self.energy_grid
            weights = self.density(spectrum_weights) * self.quadrature
            weights = weights / weights.sum()
        return {"energies": energies, "weights": weights, "gaps": energies.new_empty(0)}

    def phys_params(self):
        p = super().phys_params()
        p["phase_beta"] = self.bounded(self.raw_phase_vr, -1.0, 1.0) if self.phase_response else self.raw_phase_vr.new_zeros(())
        return p

    def kernel_components(self, Va, Vr, curve_idx=None, nuisance_mode="neutral", spectrum_weights=None):
        # Versioned copy of the periodic kernel: fixed scale and optional Vr shift.
        p = self.phys_params()
        va, vr = Va.to(self.raw_amp), Vr.to(self.raw_amp)
        e_collision = va.clamp_min(0)
        scale = self.normalization_voltage
        drive = F.softplus(e_collision - p["v_emit"], beta=2.0)
        envelope = p["amp"] * (drive + 1e-4).pow(p["power"]) / (scale + 1).pow(p["power"])
        envelope = envelope + p["offset"] + p["slope"] * e_collision
        e_collect = va - p["vr_scale"] * vr
        collector_gate = torch.sigmoid((e_collect - p["collector_threshold"]) / (p["collector_width"] + 1e-6))
        transmission = p["collector_floor"] + (1 - p["collector_floor"]) * collector_gate
        late_gate = torch.sigmoid((e_collision - p["late_onset"]) / (p["late_width"] + 1e-6))
        high_vr_gate = torch.sigmoid((vr - 6) / 2)
        vr_norm = ((vr - 5) / 5).clamp(-1.5, 1.5)
        contrast = torch.exp((p["vr_contrast"] * vr_norm + p["late_contrast"] * late_gate
                              + p["vr_late_contrast"] * vr_norm * late_gate).clamp(-0.65, 0.65))
        envelope = envelope + p["vr_late_baseline"] * high_vr_gate * late_gate * (e_collision / scale).clamp(0, 1)
        loss_gate = torch.sigmoid((e_collision - p["high_energy_loss_onset"]) / (p["high_energy_loss_width"] + 1e-6))
        high_loss = 1 - p["high_energy_loss_strength"] * loss_gate * (0.35 + 0.65 * high_vr_gate)
        oscillation_voltage = e_collision - p["phase_beta"] * (vr - 6)
        dip = self.weighted_periodic_response(oscillation_voltage, p, self.level_params(spectrum_weights))
        modulation = (1 - p["osc_amp"] * contrast * dip * torch.exp(-p["damping"] * e_collision)).clamp(0.03, 1.15)
        return {"prediction": (envelope * transmission * high_loss * modulation).clamp_min(0),
                "weighted_dip": dip, "envelope": envelope, "modulation": modulation,
                "collector_transmission": transmission, "e_collision": e_collision,
                "e_collect": e_collect}

    def forward(self, Va, Vr, curve_idx=None, nuisance_mode="curve", spectrum_weights=None):
        pred = self.kernel_components(Va, Vr, curve_idx, nuisance_mode, spectrum_weights)["prediction"]
        if nuisance_mode == "curve":
            gain, bias, _ = self.nuisance_values(curve_idx)
            while gain.ndim < pred.ndim:
                gain, bias = gain.unsqueeze(-1), bias.unsqueeze(-1)
            pred = gain * pred + bias
        return pred.clamp_min(0)

    def curvature(self, spectrum_weights=None):
        if self.family != "C":
            return self.energy_grid.new_zeros(())
        weights = simplex(self.spectral_logits) if spectrum_weights is None else spectrum_weights
        return (self.spline_second @ weights).square() @ self.quadrature

    def common_penalty(self):
        p = self.phys_params()
        scales = {"offset": 0.8, "slope": 0.04, "vr_contrast": 0.35,
                  "late_contrast": 0.35, "vr_late_contrast": 0.45,
                  "vr_late_baseline": 0.05, "high_energy_loss_strength": 0.45,
                  "phase_beta": 1.0}
        loss = sum((p[name] / scale).square() for name, scale in scales.items())
        return loss + 0.25 * (self.raw_curve_gain.tanh().square().mean()
                              + self.raw_curve_bias.tanh().square().mean())


def quantiles(energy, weights, probabilities, density=None):
    """Invert an atomic CDF or the exact CDF of a piecewise-linear density."""
    e = np.asarray(energy, dtype=float)
    probabilities = np.asarray(probabilities, dtype=float)
    if density is None:
        order = np.argsort(e)
        cumulative = np.cumsum(np.asarray(weights, dtype=float)[order])
        return e[order][np.minimum(np.searchsorted(cumulative / cumulative[-1], probabilities), len(e) - 1)]
    d = np.asarray(density, dtype=float)
    dx = np.diff(e)
    cdf = np.r_[0, np.cumsum((d[:-1] + d[1:]) * dx / 2)]
    target = probabilities * cdf[-1]
    indices = np.clip(np.searchsorted(cdf, target, side="right") - 1, 0, len(e) - 2)
    area = target - cdf[indices]
    slope = (d[indices + 1] - d[indices]) / dx[indices]
    root = np.sqrt(np.maximum(d[indices] ** 2 + 2 * slope * area, 0))
    distance = np.divide(2 * area, d[indices] + root, out=np.zeros_like(area), where=d[indices] + root > 0)
    return e[indices] + np.clip(distance, 0, dx[indices])


def interval_mass(e, d, lo, hi):
    x = np.r_[lo, e[(e > lo) & (e < hi)], hi]
    return float(np.trapezoid(np.interp(x, e, d), x))


def spectrum_summary(net):
    with torch.no_grad():
        levels = net.level_params()
        e, w = (levels[key].cpu().numpy() for key in ("energies", "weights"))
        d = None if net.family == "H1" else net.density().cpu().numpy()
    mean = float(e @ w)
    qs = quantiles(e, w, [0.05, 0.5, 0.95], d)
    result = {"mean_eV": mean, "sd_eV": float(np.sqrt(w @ (e - mean) ** 2)),
              "q05_eV": float(qs[0]), "median_eV": float(qs[1]), "q95_eV": float(qs[2]),
              "kernel_width_V": float(net.phys_params()["width"].detach()),
              "phase_beta": float(net.phys_params()["phase_beta"].detach())}
    if d is None:
        return {**result, "mode_eV": float(e[0]), "peak_status": "delta"}
    d = d / np.trapezoid(d, e)
    index = int(np.argmax(d))
    left = right = index
    half = d[index] / 2
    while left > 0 and d[left - 1] >= half:
        left -= 1
    while right < len(e) - 1 and d[right + 1] >= half:
        right += 1
    lo = float(e[left]) if left == 0 else float(np.interp(half, d[left - 1:left + 1], e[left - 1:left + 1]))
    hi = float(e[right]) if right == len(e) - 1 else float(np.interp(half, d[right:right + 2][::-1], e[right:right + 2][::-1]))
    ratio = float(d[index] * (e[-1] - e[0]))
    status = "boundary_peak" if index in (0, len(e) - 1) else (
        "diffuse_spectrum" if ratio < 1.5 or (left == 0 and right == len(e) - 1) else "concentrated")
    return {**result, "mode_eV": float(e[index]), "peak_status": status,
            "peak_to_uniform": ratio, "peak_halfheight_low_eV": lo,
            "peak_halfheight_high_eV": hi, "peak_fwhm_eV": hi - lo,
            "peak_halfheight_mass": interval_mass(e, d, lo, hi),
            "ar4s_mass": interval_mass(e, d, *NIST_4S[[0, -1]]),
            "below_ar4s_mass": interval_mass(e, d, e[0], NIST_4S[0])}


def concentration_status(summaries):
    modes = [s["mode_eV"] for s in summaries]
    stable = max(modes) - min(modes) <= 0.5
    valid = all(s["peak_status"] == "concentrated" for s in summaries)
    return {"status": "stable_concentration" if stable and valid else (
        "unstable_concentration" if valid else "no_robust_concentration"),
        "mode_median_eV": float(np.median(modes)), "mode_range_eV": max(modes) - min(modes)}
