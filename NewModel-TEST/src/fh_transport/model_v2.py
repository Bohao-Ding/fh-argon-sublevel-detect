from __future__ import annotations

import math
from itertools import product
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn


NIST_ARGON_4S_EV = (11.548355, 11.623593, 11.723161, 11.828072)


def _logit(value: float) -> float:
    value = min(max(float(value), 1.0e-6), 1.0 - 1.0e-6)
    return math.log(value / (1.0 - value))


def _inverse_bounded(value: float, lower: float, upper: float) -> float:
    return _logit((float(value) - lower) / (upper - lower))


def _enumerate_collision_histories(n_levels: int, max_collisions: int) -> list[tuple[int, ...]]:
    histories = []
    for counts in product(range(max_collisions + 1), repeat=n_levels):
        if sum(counts) <= max_collisions:
            histories.append(tuple(int(value) for value in counts))
    histories.sort(key=lambda item: (sum(item), item))
    return histories


class CompetingChannelFranckHertz(nn.Module):
    """Reduced thin-layer transport with competing excitation histories.

    The state is a vector of collision counts.  All accessible excitation
    channels compete within each layer, so a single electron can follow mixed
    histories such as E1 + E2.  The collector gate acts on signed axial energy
    and includes an exact reachability factor.  This remains a low-order master
    equation, not a full Boltzmann or trajectory Monte Carlo solver.
    """

    def __init__(
        self,
        hypothesis: str = "h1",
        n_layers: int = 32,
        max_collisions: int = 8,
        quadrature_order: int = 6,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if hypothesis not in {"h1", "h4s"}:
            raise ValueError("hypothesis must be 'h1' or 'h4s'")
        if int(n_layers) < 2 or int(max_collisions) < 1:
            raise ValueError("n_layers >= 2 and max_collisions >= 1 are required")
        self.hypothesis = hypothesis
        self.n_levels = 1 if hypothesis == "h1" else 4
        self.n_layers = int(n_layers)
        self.max_collisions = int(max_collisions)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        if hypothesis == "h1":
            self.raw_effective_energy = nn.Parameter(
                torch.tensor(_inverse_bounded(11.55, 10.5, 13.0))
            )
        else:
            self.raw_energy_shift = nn.Parameter(torch.tensor(_inverse_bounded(0.0, -0.35, 0.35)))
        self.raw_rate_logits = nn.Parameter(0.01 * torch.randn(self.n_levels, generator=generator))

        self.raw_current_limit = nn.Parameter(torch.tensor(_inverse_bounded(5.0, 0.5, 12.0)))
        self.raw_supply_scale = nn.Parameter(torch.tensor(_inverse_bounded(42.0, 5.0, 120.0)))
        self.raw_collision_offset = nn.Parameter(torch.tensor(_inverse_bounded(4.5, 0.0, 8.0)))
        self.raw_arrival_offset = nn.Parameter(torch.tensor(_inverse_bounded(4.2, 0.0, 8.0)))
        self.raw_collector_offset = nn.Parameter(torch.tensor(_inverse_bounded(0.40, 0.0, 3.0)))
        self.raw_retarding_scale = nn.Parameter(torch.tensor(_inverse_bounded(0.90, 0.50, 1.50)))
        self.raw_gate_sigma = nn.Parameter(torch.tensor(_inverse_bounded(1.50, 0.05, 3.00)))
        self.raw_thermal_energy = nn.Parameter(torch.tensor(_inverse_bounded(0.18, 0.03, 0.80)))
        self.raw_angle_sigma = nn.Parameter(torch.tensor(_inverse_bounded(0.70, 0.15, 1.50)))
        self.raw_arrival_scale = nn.Parameter(torch.tensor(_inverse_bounded(0.30, 0.05, 2.0)))
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
        self.register_buffer("nist_4s_eV", torch.tensor(NIST_ARGON_4S_EV, dtype=torch.float32))

        histories = _enumerate_collision_histories(self.n_levels, self.max_collisions)
        history_lookup = {history: index for index, history in enumerate(histories)}
        next_state = torch.full((self.n_levels, len(histories)), -1, dtype=torch.long)
        for state_index, history in enumerate(histories):
            if sum(history) >= self.max_collisions:
                continue
            for level in range(self.n_levels):
                advanced = list(history)
                advanced[level] += 1
                next_state[level, state_index] = history_lookup[tuple(advanced)]
        self.register_buffer("collision_counts", torch.tensor(histories, dtype=torch.float32))
        self.register_buffer("next_state", next_state)
        self._history_lookup = history_lookup

    @staticmethod
    def bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * torch.sigmoid(raw)

    def excitation_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self.hypothesis == "h1":
            energies = self.bounded(self.raw_effective_energy, 10.5, 13.0).view(1)
        else:
            shift = self.bounded(self.raw_energy_shift, -0.35, 0.35)
            energies = self.nist_4s_eV + shift
        return energies, torch.softmax(self.raw_rate_logits, dim=0)

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        energies, rate_fractions = self.excitation_parameters()
        return {
            "energies_eV": energies,
            "channel_rate_fractions": rate_fractions,
            "current_limit_uA": self.bounded(self.raw_current_limit, 0.5, 12.0),
            "supply_scale_V": self.bounded(self.raw_supply_scale, 5.0, 120.0),
            "collision_region_offset_eV": self.bounded(self.raw_collision_offset, 0.0, 8.0),
            "arrival_offset_eV": self.bounded(self.raw_arrival_offset, 0.0, 8.0),
            "collector_offset_eV": self.bounded(self.raw_collector_offset, 0.0, 3.0),
            "retarding_scale": self.bounded(self.raw_retarding_scale, 0.50, 1.50),
            "gate_sigma_eV": self.bounded(self.raw_gate_sigma, 0.05, 3.00),
            "thermal_energy_eV": self.bounded(self.raw_thermal_energy, 0.03, 0.80),
            "angle_sigma_rad": self.bounded(self.raw_angle_sigma, 0.15, 1.50),
            "arrival_scale_eV": self.bounded(self.raw_arrival_scale, 0.05, 2.0),
            "optical_depth": self.bounded(self.raw_optical_depth, 0.2, 30.0),
            "cross_section_rise_eV": self.bounded(self.raw_cross_rise, 0.05, 5.0),
            "cross_section_decay_eV": self.bounded(self.raw_cross_decay, 5.0, 120.0),
            "cross_section_power": self.bounded(self.raw_cross_power, 0.25, 2.0),
            "acceleration_scale": self.bounded(self.raw_accel_scale, 0.94, 1.06),
        }

    def history_index(self, counts: Iterable[int]) -> int:
        key = tuple(int(value) for value in counts)
        if key not in self._history_lookup:
            raise ValueError(f"Collision history is outside the state space: {key}")
        return self._history_lookup[key]

    def _cross_sections(
        self,
        kinetic: torch.Tensor,
        energies: torch.Tensor,
        rise: torch.Tensor,
        decay: torch.Tensor,
        power: torch.Tensor,
    ) -> torch.Tensor:
        excess = torch.relu(kinetic[..., None] - energies)
        reduced = excess / (rise + excess + 1.0e-8)
        epsilon = torch.as_tensor(1.0e-8, device=reduced.device, dtype=reduced.dtype)
        cross_section = torch.pow(reduced + epsilon, power)
        cross_section = torch.clamp(
            cross_section - torch.pow(epsilon, power), min=0.0
        )
        return cross_section * torch.exp(-excess / decay)

    def collision_state_probabilities(self, va: torch.Tensor) -> torch.Tensor:
        """Return P(history | Va, thermal node) with shape [N, Q, S]."""
        va = va.reshape(-1).float()
        params = self.physical_parameters()
        energies = params["energies_eV"]
        loss = self.collision_counts @ energies
        field_energy = params["acceleration_scale"] * va - params["collision_region_offset_eV"]
        thermal = params["thermal_energy_eV"] * self.thermal_nodes
        probability = torch.zeros(
            (va.numel(), thermal.numel(), self.collision_counts.shape[0]),
            device=va.device,
            dtype=va.dtype,
        )
        probability[:, :, 0] = 1.0
        active = (torch.sum(self.collision_counts, dim=1) < self.max_collisions).to(va.dtype)

        for layer in range(self.n_layers):
            position = (float(layer) + 0.5) / float(self.n_layers)
            kinetic = (
                field_energy[:, None, None] * position
                + thermal[None, :, None]
                - loss[None, None, :]
            )
            rates = self._cross_sections(
                kinetic,
                energies,
                params["cross_section_rise_eV"],
                params["cross_section_decay_eV"],
                params["cross_section_power"],
            )
            rates = (
                params["optical_depth"]
                * rates
                * params["channel_rate_fractions"][None, None, None, :]
                * active[None, None, :, None]
                / float(self.n_layers)
            )
            total_rate = torch.sum(rates, dim=-1)
            event_probability = -torch.expm1(-total_rate)
            channel_probability = event_probability[..., None] * rates / (
                total_rate[..., None] + 1.0e-12
            )
            updated = probability * (1.0 - event_probability)
            for level in range(self.n_levels):
                destinations = self.next_state[level]
                valid = destinations >= 0
                contribution = probability[:, :, valid] * channel_probability[:, :, valid, level]
                updated = torch.index_add(updated, 2, destinations[valid], contribution)
            probability = updated
        return probability

    def collector_state_transmission(self, va: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        """Return the angle-integrated collector gate with shape [N, Q, S]."""
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")
        params = self.physical_parameters()
        energies = params["energies_eV"]
        loss = self.collision_counts @ energies
        field_energy = params["acceleration_scale"] * va - params["arrival_offset_eV"]
        barrier = params["collector_offset_eV"] + params["retarding_scale"] * vr
        thermal = params["thermal_energy_eV"] * self.thermal_nodes
        reachability = -torch.expm1(
            -torch.relu(field_energy) / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
        )

        theta = torch.acos(torch.clamp(self.angle_mu, 1.0e-6, 1.0))
        angular_acceptance = torch.exp(-0.5 * torch.square(theta / params["angle_sigma_rad"]))
        angular_weights = self.angle_flux_weights * angular_acceptance
        angular_weights = angular_weights / torch.clamp(torch.sum(angular_weights), min=1.0e-8)

        gate = torch.zeros(
            (va.numel(), thermal.numel(), self.collision_counts.shape[0]),
            device=va.device,
            dtype=va.dtype,
        )
        for mu, weight in zip(self.angle_mu.unbind(), angular_weights.unbind()):
            axial_energy = (
                field_energy[:, None, None]
                + thermal[None, :, None] * torch.square(mu)
                - loss[None, None, :]
            )
            z = (axial_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
            gate = gate + weight * 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return gate * reachability[:, None, None]

    def forward(self, va: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")
        unique_va, inverse = torch.unique(va, sorted=True, return_inverse=True)
        collision_mass = self.collision_state_probabilities(unique_va)[inverse]
        collector_gate = self.collector_state_transmission(va, vr)
        transmission = torch.sum(collision_mass * collector_gate, dim=2)
        transmission = torch.sum(transmission * self.thermal_weights[None, :], dim=1)

        params = self.physical_parameters()
        supply_voltage = torch.relu(params["acceleration_scale"] * va)
        reduced_voltage = supply_voltage / params["supply_scale_V"]
        child_langmuir = torch.pow(reduced_voltage, 1.5)
        supply = params["current_limit_uA"] * child_langmuir / (1.0 + child_langmuir)
        return torch.clamp(supply * transmission, min=0.0)

    def parameter_report(self) -> dict[str, Any]:
        report: dict[str, Any] = {
            "model_class": type(self).__name__,
            "model_version": "v2a-competing-history",
            "hypothesis": self.hypothesis,
            "n_levels": self.n_levels,
            "n_layers": self.n_layers,
            "max_collisions": self.max_collisions,
            "state_space_size": int(self.collision_counts.shape[0]),
            "trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters()),
        }
        for name, value in self.physical_parameters().items():
            array = value.detach().cpu().numpy()
            report[name] = array.tolist() if array.ndim else float(array)
        if self.hypothesis == "h4s":
            report["fixed_relative_levels_source"] = "NIST ASD Ar I levels"
            report["nist_4s_reference_eV"] = list(NIST_ARGON_4S_EV)
        return report
