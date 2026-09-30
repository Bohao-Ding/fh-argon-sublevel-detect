import torch

from fh_transport.model_v2e import ExactLastCollisionFranckHertz


def test_v2e_joint_last_collision_distribution_conserves_probability() -> None:
    model = ExactLastCollisionFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=3
    )
    joint = model.collision_state_last_distribution(torch.tensor([0.0, 30.0, 70.0]))
    total = joint.sum(dim=(2, 3))
    assert torch.all(joint >= 0.0)
    assert torch.allclose(total, torch.ones_like(total), atol=2.0e-5)
    assert torch.count_nonzero(joint[:, :, 0, 1:]) == 0
    assert torch.count_nonzero(joint[:, :, 1:, 0]) == 0


def test_v2e_forward_has_finite_gradients() -> None:
    model = ExactLastCollisionFranckHertz(
        hypothesis="h1", n_layers=6, max_collisions=4, quadrature_order=3
    )
    va = torch.linspace(0.0, 60.0, 21)
    loss = model(va, torch.full_like(va, 8.0)).mean()
    loss.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.requires_grad
    ]
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
