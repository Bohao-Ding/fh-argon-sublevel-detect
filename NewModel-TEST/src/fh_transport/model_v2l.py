from __future__ import annotations

from typing import Any

import torch

from .model_v2i import FiniteApertureFranckHertz
from .model_v2j import CollisionFieldEnsembleFranckHertz


class CalibratedCollisionFieldFranckHertz(CollisionFieldEnsembleFranckHertz):
    """v2j kernel with apparatus-calibrated collision-field dispersion.

    The two dispersion constants are fixed by the H1 morphology scan and are
    treated as shared apparatus-closure quantities rather than level-specific
    trainable parameters.
    """

    calibrated_spread_eV = 10.5
    calibrated_decay_V = 32.0

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        del self.raw_collision_field_spread_v2j
        del self.raw_collision_spread_decay_v2j
        self.register_buffer(
            "calibrated_collision_field_spread",
            torch.tensor(self.calibrated_spread_eV, dtype=torch.float32),
        )
        self.register_buffer(
            "calibrated_collision_spread_decay",
            torch.tensor(self.calibrated_decay_V, dtype=torch.float32),
        )

    def physical_parameters(self) -> dict[str, torch.Tensor]:
        params = FiniteApertureFranckHertz.physical_parameters(self)
        params.update(
            {
                "collision_field_spread_eV": self.calibrated_collision_field_spread,
                "collision_spread_decay_V": self.calibrated_collision_spread_decay,
            }
        )
        return params

    def parameter_report(self) -> dict[str, Any]:
        report = super().parameter_report()
        report.update(
            {
                "model_class": type(self).__name__,
                "model_version": "v2l-calibrated-collision-field",
                "collision_field_calibration": (
                    "fixed H1 morphology-scan apparatus closure"
                ),
                "collision_field_calibration_grid": (
                    "sigma0=10.5..11.97 eV, decay=24..36 V; selected lowest-NRMSE "
                    "15/15-feasible point"
                ),
                "additional_trainable_parameters_vs_v2a": 2,
                "trainable_parameter_count": sum(
                    parameter.numel()
                    for parameter in self.parameters()
                    if parameter.requires_grad
                ),
            }
        )
        return report
