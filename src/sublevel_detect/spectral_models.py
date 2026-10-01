"""Effective energy measures on the existing phenomenological response kernel."""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from scipy.interpolate import BSpline
from torch import nn

from .model import PoissonRateFHCoreMultiLevel, inv_bounded

DOMAIN = (9.0, 16.0)
NIST_4S = np.array([11.54835442, 11.62359272, 11.72316039, 11.82807116])
LOCAL_FAMILIES = ("d1", "g1", "d2", "d3", "d4", "h4s", "equal4")
LOCAL_DIMENSIONS = {"d1": 1, "g1": 2, "d2": 3, "d3": 5, "d4": 7, "h4s": 4, "equal4": 4}


@lru_cache(maxsize=4)
def spline_grid(step: float = 0.02) -> tuple[np.ndarray, ...]:
    """Cache quadrature and normalized cubic basis; the grid is numerical only."""
    x = np.linspace(*DOMAIN, round((DOMAIN[1] - DOMAIN[0]) / step) + 1)
    q = np.full(len(x), x[1] - x[0])
    q[[0, -1]] *= 0.5
    knots = np.r_[np.repeat(DOMAIN[0], 4), np.arange(9.25, 16, 0.25), np.repeat(DOMAIN[1], 4)]
    basis = BSpline.design_matrix(x, knots, 3).toarray()
    integrals = q @ basis
    second = BSpline(knots, np.eye(basis.shape[1]), 3).derivative(2)(x)
    return x, q, basis / integrals, second / integrals, integrals


def simplex(raw: torch.Tensor) -> torch.Tensor:
    return torch.softmax(torch.cat([raw, raw.new_zeros(1)]), dim=0)


class EffectiveSpectrum(PoissonRateFHCoreMultiLevel):
    def __init__(self, n_curves: int, family: str, *, seed: int = 0,
                 excitation: float = 11.5, step: float = 0.02,
                 fixed_width: float | None = None) -> None:
        if family not in ("H1", "G1", "C"):
            raise ValueError(f"Unknown spectrum family: {family}")
        super().__init__(n_curves, n_levels=1, V_exc_init=min(excitation, 14.49),
                         init_jitter_scale=0.03 if seed else 0, init_seed=seed)
        self.family, self.step = family, float(step)
        for name in ("raw_E1", "raw_dE", "raw_level_logits", "raw_curve_dva"):
            getattr(self, name).requires_grad_(False)
        x, quad, basis, second, integrals = spline_grid(self.step)
        for name, value in (("energy_grid", x), ("quadrature", quad),
                            ("spline_basis", basis), ("spline_second", second)):
            self.register_buffer(name, torch.tensor(value, dtype=torch.float32))
        generator = torch.Generator().manual_seed(seed)
        mu = float(np.clip(excitation, 9.01, 15.99))
        if family in ("H1", "G1"):
            self.spectral_mu = nn.Parameter(torch.tensor(inv_bounded(mu, *DOMAIN)))
        if family == "G1":
            self.spectral_sigma = nn.Parameter(torch.tensor(inv_bounded(0.5, 0.05, 2.0)))
        if family == "C":
            logits = np.log(integrals[:-1] / integrals[-1])
            self.spectral_logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32)
                + (0.1 * torch.randn(len(logits), generator=generator) if seed else 0))
        if fixed_width is not None:
            self.raw_width.data.fill_(inv_bounded(fixed_width, 0.25, 5.0))
            self.raw_width.requires_grad_(False)

    def density(self) -> torch.Tensor:
        if self.family == "C":
            return self.spline_basis @ simplex(self.spectral_logits)
        if self.family == "G1":
            mu = self.bounded(self.spectral_mu, *DOMAIN)
            sigma = self.bounded(self.spectral_sigma, 0.05, 2.0)
            q = torch.exp(-0.5 * ((self.energy_grid - mu) / sigma) ** 2)
            return q / (q @ self.quadrature)
        raise ValueError("A delta energy has masses, not an ordinary density")

    def level_params(self) -> dict[str, torch.Tensor]:
        if self.family == "H1":
            energies = self.bounded(self.spectral_mu, *DOMAIN).view(1)
            weights = energies.new_ones(1)
        else:
            energies = self.energy_grid
            weights = self.density() * self.quadrature
            weights = weights / weights.sum()
        return {"energies": energies, "weights": weights, "gaps": energies.new_empty(0)}

    def curvature(self) -> torch.Tensor:
        if self.family != "C":
            return self.energy_grid.new_zeros(())
        second = self.spline_second @ simplex(self.spectral_logits)
        return second.square() @ self.quadrature

    def weighted_periodic_response(self, e_collision: torch.Tensor, p: dict,
                                   levels: dict) -> torch.Tensor:
        # Zero voltage shifts make the shared Va response identical across curves.
        if e_collision.ndim == 2 and not e_collision.requires_grad and torch.equal(
                e_collision, e_collision[0].expand_as(e_collision)):
            values = super().weighted_periodic_response(e_collision[0], p, levels)
            return values.unsqueeze(0).expand_as(e_collision)
        return super().weighted_periodic_response(e_collision, p, levels)

    def common_penalty(self) -> torch.Tensor:
        terms = [p.square().mean() * (0.25 if name.startswith("raw_curve") else 1.0)
                 for name, p in self.named_parameters()
                 if name.startswith("raw_") and p.requires_grad and p.numel()]
        return torch.stack(terms).sum()


def main_peak_window(models: list[EffectiveSpectrum]) -> dict:
    modes = [float(m.energy_grid[m.density().argmax()].detach().cpu()) for m in models]
    center = float(np.median(modes))
    eligible = DOMAIN[0] < center < DOMAIN[1]
    return {"status": "eligible" if eligible else "no_interior_main_peak",
            "center": center, "lo": max(DOMAIN[0], center - 0.5),
            "hi": min(DOMAIN[1], center + 0.5), "modes": modes,
            "mode_range": max(modes) - min(modes)}


class LocalSpectrum(EffectiveSpectrum):
    """Freeze outside quadrature masses and inside mass, replacing only the window."""
    def __init__(self, coarse: EffectiveSpectrum, family: str, window: dict) -> None:
        if window["status"] != "eligible":
            raise ValueError("no_interior_main_peak")
        if family not in LOCAL_FAMILIES:
            raise ValueError(f"Unknown local family: {family}")
        super().__init__(coarse.n_curves, "C", step=coarse.step)
        self.load_state_dict(coarse.state_dict())
        self.spectral_logits.requires_grad_(False)
        self.raw_width.requires_grad_(False)
        self.local_family, self.window = family, dict(window)
        self.lo, self.hi = float(window["lo"]), float(window["hi"])
        with torch.no_grad():
            levels = coarse.level_params()
            inside = (levels["energies"] >= self.lo) & (levels["energies"] <= self.hi)
            self.register_buffer("outside_energy", levels["energies"][~inside].clone())
            self.register_buffer("outside_mass", levels["weights"][~inside].clone())
            self.register_buffer("inside_grid", levels["energies"][inside].clone())
            self.register_buffer("inside_quad", coarse.quadrature[inside].clone())
            self.register_buffer("inside_mass", levels["weights"][inside].sum().clone())
        k = int(family[1:]) if family.startswith("d") else (4 if family in ("h4s", "equal4") else 1)
        self.k = k
        self.local_weights = nn.Parameter(torch.zeros(k - 1))
        if family.startswith("d"):
            self.local_positions = nn.Parameter(torch.zeros(k))
        elif family == "g1":
            self.local_mu = nn.Parameter(torch.tensor(inv_bounded(window["center"], self.lo, self.hi)))
            self.local_sigma = nn.Parameter(torch.tensor(inv_bounded(0.15, 0.05, 2.0)))
        else:
            base = NIST_4S if family == "h4s" else np.linspace(NIST_4S[0], NIST_4S[-1], 4)
            self.shift_lo = max(-0.25, self.lo - base[0])
            self.shift_hi = min(0.25, self.hi - base[-1])
            if self.shift_hi <= self.shift_lo:
                raise ValueError("outside_current_main_peak_candidates")
            self.register_buffer("template", torch.tensor(base, dtype=torch.float32))
            delta = np.clip(window["center"] - np.mean(base), self.shift_lo + 1e-6, self.shift_hi - 1e-6)
            self.local_shift = nn.Parameter(torch.tensor(inv_bounded(delta, self.shift_lo, self.shift_hi)))

    def local_measure(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.local_family.startswith("d"):
            slack = self.hi - self.lo - 0.02 * (self.k - 1)
            gaps = simplex(self.local_positions) * slack
            energies = self.lo + torch.cumsum(gaps, 0)[:self.k] + 0.02 * torch.arange(self.k, device=gaps.device)
            weights = simplex(self.local_weights)
        elif self.local_family == "g1":
            energies = self.inside_grid
            mu = self.bounded(self.local_mu, self.lo, self.hi)
            sigma = self.bounded(self.local_sigma, 0.05, 2.0)
            weights = torch.exp(-0.5 * ((energies - mu) / sigma) ** 2) * self.inside_quad
            weights = weights / weights.sum()
        else:
            energies = self.template + self.bounded(self.local_shift, self.shift_lo, self.shift_hi)
            weights = simplex(self.local_weights)
        return energies, weights * self.inside_mass

    def level_params(self) -> dict[str, torch.Tensor]:
        energies, weights = self.local_measure()
        return {"energies": torch.cat([self.outside_energy, energies]),
                "weights": torch.cat([self.outside_mass, weights]), "gaps": energies.new_empty(0)}

    def curvature(self) -> torch.Tensor:
        return self.inside_mass.new_zeros(())


def distribution_summary(model: EffectiveSpectrum) -> dict[str, float]:
    with torch.no_grad():
        levels = model.level_params()
        e, w = (levels[key].detach().cpu().numpy().astype(float) for key in ("energies", "weights"))
    order = np.argsort(e)
    e, w = e[order], w[order] / w.sum()
    cumulative = np.cumsum(w)
    quantile = lambda p: float(np.interp(p, cumulative, e))
    mean = float(w @ e)
    mode = float(model.energy_grid[model.density().argmax()].detach().cpu()) if model.family != "H1" and not isinstance(model, LocalSpectrum) else float(e[w.argmax()])
    return {"mode_eV": mode, "mean_eV": mean, "median_eV": quantile(0.5),
            "sd_eV": float(np.sqrt(w @ (e - mean) ** 2)),
            "q05_eV": quantile(0.05), "q95_eV": quantile(0.95),
            "kernel_width_V": float(model.phys_params()["width"].detach().cpu())}
