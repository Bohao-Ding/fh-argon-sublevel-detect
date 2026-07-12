from __future__ import annotations

from sublevel_detect import cli


def test_validation_defaults_off_and_validation_only_enables_it() -> None:
    default_args = cli.build_parser().parse_args([])
    staged_args = cli.build_parser().parse_args(["--validation-only", "selector"])

    assert default_args.validation is False
    assert default_args.validation_only is None
    assert cli.validation_enabled(default_args) is False
    assert staged_args.validation is False
    assert staged_args.validation_only == "selector"
    assert cli.validation_enabled(staged_args) is True


def test_validation_only_routes_one_stage_without_running_main(monkeypatch) -> None:
    validation_calls: list[dict] = []

    def fail_main_run(**kwargs):
        raise AssertionError("validation-only must not run the production main pipeline")

    def fake_validation_run(**kwargs):
        validation_calls.append(kwargs)
        return {"ok": True, "requested_stages": ["selector"]}

    monkeypatch.setattr(cli.main_pipeline, "run", fail_main_run)
    monkeypatch.setattr(cli.validation_pipeline, "run", fake_validation_run)

    exit_code = cli.main(["--mode", "smoke", "--validation-only", "selector"])

    assert exit_code == 0
    assert len(validation_calls) == 1
    assert validation_calls[0]["requested_stage"] == "selector"
    assert validation_calls[0]["mode"] == "smoke"


def test_full_validation_routes_all_stages_without_running_main(monkeypatch) -> None:
    validation_calls: list[dict] = []

    def fail_main_run(**kwargs):
        raise AssertionError("validation must resolve a frozen baseline instead of retraining main")

    def fake_validation_run(**kwargs):
        validation_calls.append(kwargs)
        return {"ok": True, "requested_stages": ["selector", "bootstrap", "holdout", "synthetic", "benchmark"]}

    monkeypatch.setattr(cli.main_pipeline, "run", fail_main_run)
    monkeypatch.setattr(cli.validation_pipeline, "run", fake_validation_run)

    assert cli.main(["--mode", "smoke", "--validation"]) == 0
    assert validation_calls[0]["requested_stage"] is None


def test_validation_failure_returns_nonzero(monkeypatch) -> None:
    monkeypatch.setattr(cli.validation_pipeline, "run", lambda **kwargs: {"ok": False})

    assert cli.main(["--mode", "smoke", "--validation-only", "selector"]) == 1


def test_legacy_run_without_validation_still_runs_main(monkeypatch) -> None:
    calls = {"main": 0, "validation": 0}

    def fake_main_run(**kwargs):
        calls["main"] += 1
        return {"sweep_dir": "unused", "ok": True}

    def fake_validation_run(**kwargs):
        calls["validation"] += 1
        return {"ok": True}

    monkeypatch.setattr(cli.main_pipeline, "run", fake_main_run)
    monkeypatch.setattr(cli.validation_pipeline, "run", fake_validation_run)

    assert cli.main(["--mode", "smoke"]) == 0
    assert calls == {"main": 1, "validation": 0}
