from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .model_v2 import _inverse_bounded
from .model_v2d import EnergyDependentScatteringFranckHertz
from .model_v2g import GeneralizedSupplyFranckHertz


class ReachabilityGatedFranckHertz(GeneralizedSupplyFranckHertz):
    """v2g kernel with a separate arrival-reachability condition.

    The reachability factor requires non-negative axial arrival energy before
    the Gaussian retarding-potential gate is applied.  Its energy scale controls
    the smooth transition from inaccessible to fully incident electrons.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.raw_reachability_scale_v2h = nn.Parameter(
            torch.tensor(_inverse_bounded(1.20, 0.05, 3.00))
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = super().physical_parameters()
        params["arrival_scale_eV"] = self.bounded(
            self.raw_reachability_scale_v2h, 0.05, 3.00
        )
        return params

    def collector_state_transmission(
        self,
        va: torch.Tensor,
        vr: torch.Tensor,
        last_position: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return EnergyDependentScatteringFranckHertz.collector_state_transmission(
            self, va, vr, last_position
        )

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2h-reachability-gated",
                "collector_reachability": (
                    "non-negative axial arrival energy followed by Gaussian barrier gate"
                ),
                "additional_trainable_parameters_vs_v2a": 1,
                "trainable_parameter_count": sum(
                    parameter.numel() for parameter in self.parameters() if parameter.requires_grad
                ),
            }
        )
        return report
