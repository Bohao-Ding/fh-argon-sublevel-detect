from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from sublevel_detect import paths


def test_smoke_with_ablation_writes_main_and_ablation_outputs() -> None:
    output_root = Path(tempfile.mkdtemp(prefix="sublevel_detect_smoke_", dir=paths.PROJECT_ROOT))
    try:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                str(paths.PROJECT_ROOT / "run.py"),
                "--mode",
                "smoke",
                "--output",
                str(output_root),
                "--exclude",
                "hpopt",
                "--ablation",
                "--robustness",
            ],
            cwd=str(paths.PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (output_root / "main" / "fullscan" / "decision.json").exists()
        assert (output_root / "main" / "fullscan" / "scan_summary.csv").exists()
        assert (output_root / "main" / "k_selected_full" / "prediction_points.csv").exists()
        assert (output_root / "main" / "paper_summary.json").exists()
        assert (output_root / "main" / "paper_summary.md").exists()
        assert (output_root / "ablation" / "ablation_summary.csv").exists()
        assert (output_root / "ablation" / "ablation_report.md").exists()
        assert (output_root / "robustness" / "robustness_summary.json").exists()
        assert (output_root / "robustness" / "selector_weight_perturbation.csv").exists()
        assert (output_root / "robustness" / "leave_one_vr_out_summary.csv").exists()
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_smoke_with_sensitivity_writes_prior_and_uncertainty_outputs() -> None:
    output_root = Path(tempfile.mkdtemp(prefix="sublevel_detect_sensitivity_smoke_", dir=paths.PROJECT_ROOT))
    try:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                str(paths.PROJECT_ROOT / "run.py"),
                "--mode",
                "smoke",
                "--output",
                str(output_root),
                "--exclude",
                "hpopt",
                "--sensitivity",
            ],
            cwd=str(paths.PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert (output_root / "sensitivity" / "sensitivity_summary.json").exists()
        assert (output_root / "sensitivity" / "prior_strength" / "prior_strength_selection.csv").exists()
        assert (output_root / "sensitivity" / "prior_strength" / "prior_strength_channel_drift.csv").exists()
        assert (output_root / "sensitivity" / "uncertainty" / "channel_uncertainty_samples.csv").exists()
        assert (output_root / "sensitivity" / "uncertainty" / "channel_uncertainty_conditional_k4.csv").exists()
        assert (output_root / "sensitivity" / "uncertainty" / "channel_uncertainty_anchor_matched.csv").exists()
        assert (output_root / "sensitivity" / "uncertainty" / "channel_uncertainty_summary.csv").exists()
        assert (output_root / "sensitivity" / "uncertainty" / "uncertainty_selection_summary.csv").exists()
    finally:
        shutil.rmtree(output_root, ignore_errors=True)


def test_h4s_validation_smoke_writes_targeted_outputs_and_summaries() -> None:
    output_root = Path(tempfile.mkdtemp(prefix="sublevel_detect_validation_smoke_", dir=paths.PROJECT_ROOT))
    try:
        env = dict(os.environ)
        env["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            [
                sys.executable,
                str(paths.PROJECT_ROOT / "run.py"),
                "--mode",
                "smoke",
                "--output",
                str(output_root),
                "--exclude",
                "hpopt",
                "--validation-only",
                "h4s",
                "--device",
                "cpu",
            ],
            cwd=str(paths.PROJECT_ROOT),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        validation_root = output_root / "validation"
        assert (validation_root / "experiment_manifest.json").exists()
        assert (validation_root / "progress.json").exists()
        assert (validation_root / "validation_summary.json").exists()
        assert (validation_root / "validation_summary.md").exists()
        h4s_root = validation_root / "h4s_comparison"
        assert (h4s_root / "h4s_comparison_summary.json").exists()
        assert (h4s_root / "hypothesis_manifest.json").exists()
        assert (h4s_root / "unit_status.csv").exists()
        assert (h4s_root / "zero_shot_metrics.csv").exists()
        assert (h4s_root / "zero_shot_predictions.csv").exists()
        assert (h4s_root / "paired_contrasts.csv").exists()
        assert len(list(validation_root.glob("*/stage_manifest.json"))) == 1
        assert len(list(validation_root.glob("*/stage_result.json"))) == 1
    finally:
        shutil.rmtree(output_root, ignore_errors=True)
