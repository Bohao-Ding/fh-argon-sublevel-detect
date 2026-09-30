from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _inverse_bounded(value: float, lower: float, upper: float) -> float:
    return _logit((float(value) - lower) / (upper - lower))


class ThinLayerFranckHertz(nn.Module):
    """Differentiable, low-order electron transport model.

    Each excitation channel represents a sub-ensemble whose repeated inelastic
    collisions are propagated through uniform thin acceleration layers.  The
    channel responses share the cathode supply and collector transmission.  This
    is a reduced master-equation approximation, not a Boltzmann or trajectory
    Monte Carlo solver.
    """

    def __init__(
        self,
        n_levels: int = 1,
        n_layers: int = 32,
        max_collisions: int = 8,
        quadrature_order: int = 10,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if not 1 <= int(n_levels) <= 8:
            raise ValueError("n_levels must be in [1, 8]")
        if int(n_layers) < 2 or int(max_collisions) < 1:
            raise ValueError("n_layers >= 2 and max_collisions >= 1 are required")
        self.n_levels = int(n_levels)
        self.n_layers = int(n_layers)
        self.max_collisions = int(max_collisions)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        self.raw_e1 = nn.Parameter(torch.tensor(_inverse_bounded(11.55, 9.0, 14.5)))
        if self.n_levels > 1:
            initial_gap = 0.16
            raw_gap = _inverse_bounded(initial_gap, 0.03, 1.25)
            gaps = torch.full((self.n_levels - 1,), raw_gap)
            gaps += 0.03 * torch.randn(self.n_levels - 1, generator=generator)
            self.raw_gaps = nn.Parameter(gaps)
        else:
            self.raw_gaps = nn.Parameter(torch.empty(0))
        logits = 0.02 * torch.randn(self.n_levels, generator=generator)
        self.raw_channel_logits = nn.Parameter(logits)

        self.raw_current_limit = nn.Parameter(torch.tensor(_inverse_bounded(5.0, 0.5, 12.0)))
        self.raw_supply_scale = nn.Parameter(torch.tensor(_inverse_bounded(42.0, 5.0, 120.0)))
        self.raw_supply_offset = nn.Parameter(torch.tensor(_inverse_bounded(3.8, 0.0, 8.0)))
        self.raw_acceleration_offset = nn.Parameter(torch.tensor(_inverse_bounded(3.8, 0.0, 8.0)))
        self.raw_collector_offset = nn.Parameter(torch.tensor(_inverse_bounded(0.45, 0.0, 3.0)))
        self.raw_retarding_scale = nn.Parameter(torch.tensor(_inverse_bounded(0.95, 0.50, 1.50)))
        self.raw_gate_sigma = nn.Parameter(torch.tensor(_inverse_bounded(0.55, 0.08, 3.0)))
        self.raw_thermal_energy = nn.Parameter(torch.tensor(_inverse_bounded(0.22, 0.03, 1.5)))
        self.raw_angle_sigma = nn.Parameter(torch.tensor(_inverse_bounded(0.70, 0.15, 1.5)))
        self.raw_optical_depth = nn.Parameter(torch.tensor(_inverse_bounded(9.0, 0.2, 30.0)))
        self.raw_cross_rise = nn.Parameter(torch.tensor(_inverse_bounded(0.65, 0.05, 5.0)))
        self.raw_cross_decay = nn.Parameter(torch.tensor(_inverse_bounded(35.0, 5.0, 120.0)))
        self.raw_cross_power = nn.Parameter(torch.tensor(_inverse_bounded(0.75, 0.25, 2.0)))
        self.raw_accel_scale = nn.Parameter(torch.tensor(_inverse_bounded(1.0, 0.94, 1.06)))

        laguerre_x, laguerre_w = np.polynomial.laguerre.laggauss(int(quadrature_order))
        legendre_x, legendre_w = np.polynomial.legendre.leggauss(int(quadrature_order))
        mu = 0.5 * (legendre_x + 1.0)
        mu_w = 0.5 * legendre_w * 2.0 * mu
        self.register_buffer("thermal_nodes", torch.tensor(laguerre_x, dtype=torch.float32))
        self.register_buffer("thermal_weights", torch.tensor(laguerre_w, dtype=torch.float32))
        self.register_buffer("angle_mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("angle_flux_weights", torch.tensor(mu_w, dtype=torch.float32))
        self.register_buffer(
            "collision_numbers",
            torch.arange(self.max_collisions + 1, dtype=torch.float32),
        )

    @staticmethod
    def bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * torch.sigmoid(raw)

    def excitation_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        e1 = self.bounded(self.raw_e1, 9.0, 14.5)
        if self.n_levels > 1:
            gaps = self.bounded(self.raw_gaps, 0.03, 1.25)
            energies = torch.cat((e1.view(1), e1 + torch.cumsum(gaps, dim=0)))
        else:
            energies = e1.view(1)
        return energies, torch.softmax(self.raw_channel_logits, dim=0)

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        energies, weights = self.excitation_parameters()
        return {
            "energies_eV": energies,
            "channel_weights": weights,
            "current_limit_uA": self.bounded(self.raw_current_limit, 0.5, 12.0),
            "supply_scale_V": self.bounded(self.raw_supply_scale, 5.0, 120.0),
            "supply_offset_eV": self.bounded(self.raw_supply_offset, 0.0, 8.0),
            "acceleration_offset_eV": self.bounded(self.raw_acceleration_offset, 0.0, 8.0),
            "collector_offset_eV": self.bounded(self.raw_collector_offset, 0.0, 3.0),
            "retarding_scale": self.bounded(self.raw_retarding_scale, 0.50, 1.50),
            "gate_sigma_eV": self.bounded(self.raw_gate_sigma, 0.08, 3.0),
            "thermal_energy_eV": self.bounded(self.raw_thermal_energy, 0.03, 1.5),
            "angle_sigma_rad": self.bounded(self.raw_angle_sigma, 0.15, 1.5),
            "optical_depth": self.bounded(self.raw_optical_depth, 0.2, 30.0),
            "cross_section_rise_eV": self.bounded(self.raw_cross_rise, 0.05, 5.0),
            "cross_section_decay_eV": self.bounded(self.raw_cross_decay, 5.0, 120.0),
            "cross_section_power": self.bounded(self.raw_cross_power, 0.25, 2.0),
            "acceleration_scale": self.bounded(self.raw_accel_scale, 0.94, 1.06),
        }

    def _collision_probabilities_multi(
        self,
        va: torch.Tensor,
        excitation_eV: torch.Tensor,
    ) -> torch.Tensor:
        params = self.physical_parameters()
        accelerated = torch.clamp(
            params["acceleration_scale"] * va.float() - params["acceleration_offset_eV"],
            min=0.0,
        )
        energies = excitation_eV.reshape(-1).to(device=accelerated.device, dtype=accelerated.dtype)
        count = self.collision_numbers.to(device=accelerated.device, dtype=accelerated.dtype)
        probability = torch.zeros(
            (accelerated.numel(), energies.numel(), self.max_collisions + 1),
            device=accelerated.device,
            dtype=accelerated.dtype,
        )
        probability[:, :, 0] = 1.0
        injection_mean = params["thermal_energy_eV"]
        for layer in range(self.n_layers):
            position = (float(layer) + 0.5) / float(self.n_layers)
            kinetic = (
                accelerated[:, None, None] * position
                + injection_mean
                - count[None, None, :] * energies[None, :, None]
            )
            excess = torch.relu(kinetic - energies[None, :, None])
            reduced = excess / (params["cross_section_rise_eV"] + excess + 1.0e-8)
            epsilon = torch.as_tensor(1.0e-8, device=reduced.device, dtype=reduced.dtype)
            cross_section = torch.pow(reduced + epsilon, params["cross_section_power"])
            cross_section = torch.clamp(
                cross_section - torch.pow(epsilon, params["cross_section_power"]),
                min=0.0,
            )
            cross_section = cross_section * torch.exp(-excess / params["cross_section_decay_eV"])
            collision = -torch.expm1(-params["optical_depth"] * cross_section / float(self.n_layers))
            collision = torch.cat(
                (collision[:, :, :-1], torch.zeros_like(collision[:, :, -1:])),
                dim=2,
            )
            stay = probability * (1.0 - collision)
            advance = F.pad(probability[:, :, :-1] * collision[:, :, :-1], (1, 0))
            probability = stay + advance
        return probability

    def collision_probabilities(self, va: torch.Tensor, excitation_eV: torch.Tensor) -> torch.Tensor:
        scalar = excitation_eV.ndim == 0
        probability = self._collision_probabilities_multi(va, excitation_eV)
        return probability[:, 0, :] if scalar else probability

    def _collector_transmission_multi(
        self,
        va: torch.Tensor,
        vr: torch.Tensor,
        excitation_eV: torch.Tensor,
    ) -> torch.Tensor:
        params = self.physical_parameters()
        accelerated = torch.clamp(
            params["acceleration_scale"] * va.float() - params["acceleration_offset_eV"],
            min=0.0,
        )
        energies = excitation_eV.reshape(-1).to(device=accelerated.device, dtype=accelerated.dtype)
        count = self.collision_numbers.to(device=accelerated.device, dtype=accelerated.dtype)
        thermal = params["thermal_energy_eV"] * self.thermal_nodes
        residual = (
            accelerated[:, None, None, None, None]
            + thermal[None, None, None, :, None]
            - count[None, None, :, None, None] * energies[None, :, None, None, None]
        )
        mu = self.angle_mu[None, None, None, None, :]
        longitudinal_energy = torch.clamp(residual, min=0.0) * torch.square(mu)
        barrier = params["collector_offset_eV"] + params["retarding_scale"] * vr.float()
        z = (longitudinal_energy - barrier[:, None, None, None, None]) / params["gate_sigma_eV"]
        energy_gate = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))

        theta = torch.acos(torch.clamp(self.angle_mu, 1.0e-6, 1.0))
        angular_acceptance = torch.exp(-0.5 * torch.square(theta / params["angle_sigma_rad"]))
        angular_weights = self.angle_flux_weights * angular_acceptance
        angular_weights = angular_weights / torch.clamp(torch.sum(angular_weights), min=1.0e-8)
        gate = torch.sum(energy_gate * angular_weights[None, None, None, None, :], dim=-1)
        gate = torch.sum(gate * self.thermal_weights[None, None, None, :], dim=-1)
        return gate

    def forward(self, va: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")
        params = self.physical_parameters()
        energies = params["energies_eV"]
        weights = params["channel_weights"]
        collision_mass = self._collision_probabilities_multi(va, energies)
        collector_gate = self._collector_transmission_multi(va, vr, energies)
        per_channel = torch.sum(collision_mass * collector_gate, dim=2)
        transmission = torch.sum(per_channel * weights[None, :], dim=1)

        supply_voltage = torch.clamp(
            params["acceleration_scale"] * va - params["supply_offset_eV"],
            min=0.0,
        )
        reduced_voltage = supply_voltage / params["supply_scale_V"]
        child_langmuir = torch.pow(torch.clamp(reduced_voltage, min=0.0), 1.5)
        supply = params["current_limit_uA"] * child_langmuir / (1.0 + child_langmuir)
        return torch.clamp(supply * transmission, min=0.0)

    def parameter_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "model_class": type(self).__name__,
            "n_levels": self.n_levels,
            "n_layers": self.n_layers,
            "max_collisions": self.max_collisions,
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }
        for name, value in self.physical_parameters().items():
            array = value.detach().cpu().numpy()
            report[name] = array.tolist() if array.ndim else float(array)
        return report
