from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from sublevel_detect import model


def _curves() -> list[dict]:
    va = np.linspace(0.0, 80.0, 21, dtype=np.float32)
    return [
        {
            "curve_id": 1,
            "curve_idx": 0,
            "Va": va,
            "Vr": 0.0,
            "Vr_is_vector": False,
            "Ip": (0.2 + 0.01 * va).astype(np.float32),
        }
    ]


def _config() -> model.Config:
    cfg = model.Config()
    cfg.optimizer = "adamw"
    cfg.epochs = 2
    cfg.checkpoint_min_interval = 1
    cfg.early_stop_min_epochs = 10
    cfg.early_stop_warmup = 10
    cfg.forward_prior_mode = "off"
    cfg.forward_evidence_path = ""
    return cfg


def _model(n_curves: int = 1) -> model.PoissonRateFHCoreMultiLevel:
    return model.PoissonRateFHCoreMultiLevel(
        n_curves=n_curves,
        n_max=4,
        n_levels=1,
        min_level_gap=0.04,
        V_exc_init=11.5,
        init_jitter_scale=0.0,
        init_seed=0,
        device=torch.device("cpu"),
    )


def test_prepare_training_curves_caches_static_masks() -> None:
    cfg = _config()
    cfg.vr_late_va_min = 60.0
    prepared = model.prepare_training_curves(_curves(), torch.device("cpu"), cfg)

    assert len(prepared) == 1
    assert prepared[0]["late_point_count"] == 6
    assert prepared[0]["late_mask"].dtype == torch.bool
    assert prepared[0]["high_vr_curve"] is False


def test_batched_curve_loss_is_numerically_equivalent() -> None:
    cfg = _config()
    curves = _curves()
    second = {**curves[0], "curve_id": 2, "curve_idx": 1, "Vr": 4.0}
    curves.append(second)
    prepared = model.prepare_training_curves(curves, torch.device("cpu"), cfg)
    batched = model.prepare_batched_training_inputs(prepared)
    trained = _model(n_curves=2)

    scalar = model.compute_losses_multilevel(cfg, trained, prepared)
    vectorized = model.compute_losses_multilevel(
        cfg,
        trained,
        prepared,
        batched_inputs=batched,
    )

    assert batched is not None
    for name in scalar:
        assert float(vectorized[name].detach()) == pytest.approx(
            float(scalar[name].detach()), abs=1e-7
        )


def test_atomic_torch_save_preserves_previous_checkpoint_on_failure(monkeypatch, tmp_path) -> None:
    target = tmp_path / "checkpoint.pt"
    model.atomic_torch_save({"value": 1}, target)
    before = target.read_bytes()

    def fail_save(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(model.torch, "save", fail_save)
    with pytest.raises(OSError, match="simulated"):
        model.atomic_torch_save({"value": 2}, target)

    assert target.read_bytes() == before
    assert list(tmp_path.glob(".*.tmp")) == []


def test_terminal_resume_does_not_train_an_extra_epoch_and_accumulates_time(tmp_path) -> None:
    cfg = _config()
    first = model.train_multilevel(
        cfg,
        _model(),
        _curves(),
        torch.device("cpu"),
        tmp_path,
        epochs=2,
    )
    checkpoint = tmp_path / "checkpoint_last.pt"
    checkpoint_before = checkpoint.read_bytes()

    resumed = model.train_multilevel(
        cfg,
        _model(),
        _curves(),
        torch.device("cpu"),
        tmp_path,
        epochs=2,
        resume_ckpt=checkpoint,
    )

    assert resumed["epochs_completed"] == 2
    assert resumed["best_epoch"] == first["best_epoch"]
    assert resumed["training_terminal"] is True
    assert resumed["terminal_reason"] == "completed"
    assert resumed["elapsed_seconds"] >= first["elapsed_seconds"]
    assert checkpoint.read_bytes() == checkpoint_before


def test_resume_rejects_config_mismatch_and_nonresumable_checkpoint(tmp_path) -> None:
    cfg = _config()
    model.train_multilevel(
        cfg,
        _model(),
        _curves(),
        torch.device("cpu"),
        tmp_path,
        epochs=2,
    )
    checkpoint = tmp_path / "checkpoint_last.pt"

    changed = _config()
    changed.lr = cfg.lr * 2.0
    with pytest.raises(ValueError, match="config mismatch"):
        model.train_multilevel(
            changed,
            _model(),
            _curves(),
            torch.device("cpu"),
            tmp_path / "changed",
            epochs=2,
            resume_ckpt=checkpoint,
        )

    weights_only = tmp_path / "weights_only.pt"
    model.atomic_torch_save(_model().state_dict(), weights_only)
    with pytest.raises(ValueError, match="missing resumable training state"):
        model.train_multilevel(
            cfg,
            _model(),
            _curves(),
            torch.device("cpu"),
            tmp_path / "weights",
            epochs=2,
            resume_ckpt=weights_only,
        )


def test_worker_runtime_sets_single_thread_environment(monkeypatch) -> None:
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    monkeypatch.delenv("MKL_NUM_THREADS", raising=False)

    assert model.configure_worker_runtime(1) == 1
    assert os.environ["OMP_NUM_THREADS"] == "1"
    assert os.environ["MKL_NUM_THREADS"] == "1"
    assert torch.get_num_threads() == 1
