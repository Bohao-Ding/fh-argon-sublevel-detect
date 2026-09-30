from __future__ import annotations

import math
from typing import Any

import torch

from .model_v2d import EnergyDependentScatteringFranckHertz


class ExactLastCollisionFranckHertz(EnergyDependentScatteringFranckHertz):
    """Competing-history kernel with the full last-collision layer distribution."""

    def collision_state_last_distribution(self, va: torch.Tensor) -> torch.Tensor:
        """Return P(history, last-layer | Va, thermal node).

        Last-layer index 0 denotes no inelastic collision.  Index l+1 denotes
        that the most recent inelastic collision occurred in thin layer l.
        """
        va = va.reshape(-1).float()
        params = self.physical_parameters()
        energies = params["energies_eV"]
        loss = self.collision_counts @ energies
        field_energy = params["acceleration_scale"] * va - params["collision_region_offset_eV"]
        thermal = params["thermal_energy_eV"] * self.thermal_nodes
        joint = torch.zeros(
            (
                va.numel(),
                thermal.numel(),
                self.collision_counts.shape[0],
                self.n_layers + 1,
            ),
            device=va.device,
            dtype=va.dtype,
        )
        joint[:, :, 0, 0] = 1.0
        active = (torch.sum(self.collision_counts, dim=1) < self.max_collisions).to(va.dtype)

        for layer in range(self.n_layers):
            probability = torch.sum(joint, dim=3)
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
            updated = joint * (1.0 - event_probability)[..., None]
            arrivals = torch.zeros_like(probability)
            for level in range(self.n_levels):
                destinations = self.next_state[level]
                valid = destinations >= 0
                contribution = probability[:, :, valid] * channel_probability[:, :, valid, level]
                arrivals = torch.index_add(arrivals, 2, destinations[valid], contribution)
            selector = torch.zeros(self.n_layers + 1, device=va.device, dtype=va.dtype)
            selector[layer + 1] = 1.0
            joint = updated + arrivals[..., None] * selector
        return joint

    def collision_state_moments(self, va: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        joint = self.collision_state_last_distribution(va)
        positions = torch.cat(
            (
                torch.zeros(1, device=va.device, dtype=va.dtype),
                (torch.arange(self.n_layers, device=va.device, dtype=va.dtype) + 0.5)
                / float(self.n_layers),
            )
        )
        return torch.sum(joint, dim=3), torch.sum(joint * positions, dim=3)

    def collision_state_probabilities(self, va: torch.Tensor) -> torch.Tensor:
        return torch.sum(self.collision_state_last_distribution(va), dim=3)

    def _gate_at_position(
        self,
        va: torch.Tensor,
        vr: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        params = self.physical_parameters()
        energies = params["energies_eV"]
        loss = self.collision_counts @ energies
        field_energy = params["acceleration_scale"] * va - params["arrival_offset_eV"]
        barrier = params["collector_offset_eV"] + params["retarding_scale"] * vr
        thermal = params["thermal_energy_eV"] * self.thermal_nodes
        post_collision_energy = torch.relu(
            field_energy[:, None, None] * position
            + thermal[None, :, None]
            - loss[None, None, :]
        )
        axial_recovery = field_energy[:, None, None] * (1.0 - position)
        angular_weights = self.angle_flux_weights / torch.clamp(
            torch.sum(self.angle_flux_weights), min=1.0e-8
        )

        isotropic_gate = torch.zeros(
            (va.numel(), thermal.numel(), self.collision_counts.shape[0]),
            device=va.device,
            dtype=va.dtype,
        )
        for mu, weight in zip(self.angle_mu.unbind(), angular_weights.unbind()):
            axial_energy = axial_recovery + post_collision_energy * torch.square(mu)
            reachability = -torch.expm1(
                -torch.relu(axial_energy)
                / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
            )
            z = (axial_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
            isotropic_gate = isotropic_gate + weight * reachability * (
                0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            )

        forward_energy = axial_recovery + post_collision_energy
        forward_reachability = -torch.expm1(
            -torch.relu(forward_energy)
            / torch.clamp(params["arrival_scale_eV"], min=1.0e-8)
        )
        forward_z = (forward_energy - barrier[:, None, None]) / params["gate_sigma_eV"]
        forward_gate = forward_reachability * 0.5 * (
            1.0 + torch.erf(forward_z / math.sqrt(2.0))
        )
        collided = (torch.sum(self.collision_counts, dim=1) > 0).to(va.dtype)
        anisotropy = post_collision_energy / (
            post_collision_energy + params["scattering_anisotropy_scale_eV"]
        )
        anisotropy = anisotropy * collided[None, None, :]
        return isotropic_gate * (1.0 - anisotropy) + forward_gate * anisotropy

    def forward(self, va: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")
        unique_va, inverse = torch.unique(va, sorted=True, return_inverse=True)
        joint = self.collision_state_last_distribution(unique_va)[inverse]
        positions = torch.cat(
            (
                torch.zeros(1, device=va.device, dtype=va.dtype),
                (torch.arange(self.n_layers, device=va.device, dtype=va.dtype) + 0.5)
                / float(self.n_layers),
            )
        )
        transmission = torch.zeros_like(va)
        for index, position in enumerate(positions.unbind()):
            gate = self._gate_at_position(va, vr, position)
            state_transmission = torch.sum(joint[:, :, :, index] * gate, dim=2)
            transmission = transmission + torch.sum(
                state_transmission * self.thermal_weights[None, :], dim=1
            )

        params = self.physical_parameters()
        supply_voltage = torch.relu(params["acceleration_scale"] * va)
        reduced_voltage = supply_voltage / params["supply_scale_V"]
        child_langmuir = torch.pow(reduced_voltage, 1.5)
        supply = params["current_limit_uA"] * child_langmuir / (1.0 + child_langmuir)
        return torch.clamp(supply * transmission, min=0.0)

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2e-exact-last-collision",
                "collector_closure": "full discrete last-collision layer distribution",
                "last_collision_bins": self.n_layers + 1,
            }
        )
        return report
