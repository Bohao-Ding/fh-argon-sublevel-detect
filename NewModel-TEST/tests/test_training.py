from fh_transport.train import PlateauStopper


def test_plateau_stopper_terminates_after_minimum_epochs() -> None:
    stopper = PlateauStopper(min_epochs=10, patience=4, min_delta_rel=1.0e-4, smoothing=2)
    stopped_at = None
    for epoch in range(1, 30):
        should_stop, _ = stopper.update(epoch, 1.0)
        if should_stop:
            stopped_at = epoch
            break
    assert stopped_at is not None
    assert stopped_at >= 10

