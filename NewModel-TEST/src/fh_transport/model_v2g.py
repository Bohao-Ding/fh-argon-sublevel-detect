from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .model_v2 import _inverse_bounded
from .model_v2f import ThresholdGaussianFranckHertz


class GeneralizedSupplyFranckHertz(ThresholdGaussianFranckHertz):
    """v2f kernel with a generalized space-charge supply exponent."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_supply_exponent_v2g = nn.Parameter(
            torch.tensor(_inverse_bounded(1.30, 0.80, 1.80))
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = super().physical_parameters()
        params["supply_exponent"] = self.bounded(self.raw_supply_exponent_v2g, 0.80, 1.80)
        return params

    def forward(self, va: torch.Tensor, vr: torch.Tensor) -> torch.Tensor:
        va = va.reshape(-1).float()
        vr = vr.reshape(-1).float()
        if va.shape != vr.shape:
            raise ValueError("Va and Vr must have the same shape")
        unique_va, inverse = torch.unique(va, sorted=True, return_inverse=True)
        collision_mass, last_mass = self.collision_state_moments(unique_va)
        collision_mass = collision_mass[inverse]
        last_mass = last_mass[inverse]
        last_position = last_mass / torch.clamp(collision_mass, min=1.0e-12)
        collector_gate = self.collector_state_transmission(va, vr, last_position)
        transmission = torch.sum(collision_mass * collector_gate, dim=2)
        transmission = torch.sum(transmission * self.thermal_weights[None, :], dim=1)

        params = self.physical_parameters()
        supply_voltage = torch.relu(params["acceleration_scale"] * va)
        reduced_voltage = supply_voltage / params["supply_scale_V"]
        space_charge = torch.pow(reduced_voltage, params["supply_exponent"])
        supply = params["current_limit_uA"] * space_charge / (1.0 + space_charge)
        return torch.clamp(supply * transmission, min=0.0)

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2g-generalized-supply",
                "supply_law": "I_lim*u^p/(1+u^p), u=a*Va/V_scale",
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        )
        return report
