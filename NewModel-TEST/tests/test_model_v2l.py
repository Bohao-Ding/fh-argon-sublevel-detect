import torch

from fh_transport.model_v2j import CollisionFieldEnsembleFranckHertz
from fh_transport.model_v2l import CalibratedCollisionFieldFranckHertz


def test_v2l_dispersion_is_fixed_and_not_trainable() -> None:
    model = CalibratedCollisionFieldFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    params = model.physical_parameters()
    assert float(params["collision_field_spread_eV"]) == 10.5
    assert float(params["collision_spread_decay_V"]) == 32.0
    trainable_names = {name for name, value in model.named_parameters() if value.requires_grad}
    assert not any("collision_field_spread" in name for name in trainable_names)
    assert not any("collision_spread_decay" in name for name in trainable_names)


def test_v2l_ignores_v2j_trainable_dispersion_when_loading_compatible_state() -> None:
    source = CollisionFieldEnsembleFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    target = CalibratedCollisionFieldFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=4
    )
    target_state = target.state_dict()
    compatible = {
        name: value
        for name, value in source.state_dict().items()
        if name in target_state and target_state[name].shape == value.shape
    }
    target.load_state_dict(compatible, strict=False)
    params = target.physical_parameters()
    assert float(params["collision_field_spread_eV"]) == 10.5
    assert float(params["collision_spread_decay_V"]) == 32.0


def test_v2l_forward_is_finite() -> None:
    model = CalibratedCollisionFieldFranckHertz(
        hypothesis="h1", n_layers=8, max_collisions=5, quadrature_order=4
    )
    va = torch.linspace(0.0, 60.0, 25)
    prediction = model(va, torch.full_like(va, 8.0))
    assert float(prediction[0].detach()) == 0.0
    assert torch.all(torch.isfinite(prediction))
