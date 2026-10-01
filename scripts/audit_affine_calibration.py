"""Reanalyse frozen spectra for current readout and voltage-scale ambiguity.

No physical-kernel retraining, NIST anchoring, or heldout recalibration occurs.
The old smoothing choice is frozen inside each outer training set. Only the
identity/common affine readout is selected by its corresponding inner folds.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import find_peaks, savgol_filter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sublevel_detect.affine_readout import apply_readout, fit_readout, readout_values, select_readout
from sublevel_detect.spectral_models import EffectiveSpectrum, NIST_4S
from sublevel_detect.spectral_pipeline import (curves_hash, load_curves, morphology,
    predict, source_hashes, training_curves)
from sublevel_detect.validation_common import atomic_json_dump, sha256_file
from sublevel_detect.validation_holdout import prediction_metrics

EVIDENCE = ROOT / "source_data_package/spectral_evidence_v1"
REFERENCE = ROOT / "source_data_package/calibration_reference_v1"
MODES = ("none", "global", "linear_vr_diagnostic")


def feature_audit(curves: list[dict]) -> tuple[list[dict], list[dict]]:
    features, regressions = [], []
    for window in (7, 11, 15):
        for curve in curves:
            smooth = savgol_filter(curve["Ip"], window, 3)
            for label, sign in (("peak", 1), ("trough", -1)):
                indices = find_peaks(sign * smooth,
                    prominence=max(float(np.ptp(curve["Ip"])) * 0.025, 0.001), distance=12)[0]
                for order, index in enumerate(indices):
                    features.append(dict(window=window, Vr=curve["Vr"], feature=label,
                        visible_order=order, Va_V=float(curve["Va"][index])))
        frame = pd.DataFrame(features)
        subset = frame[(frame.window == window) & (frame.feature == "trough")
            & frame.Vr.isin((4.0, 6.0, 8.0)) & frame.Va_V.between(30, 75)].copy()
        if subset.groupby("Vr").size().tolist() != [4, 4, 4]:
            regressions.append(dict(window=window, status="unmatched_feature_counts"))
            continue
        subset["order"] = subset.groupby("Vr").cumcount()
        n, z, voltage = subset.order.to_numpy(), subset.Vr.to_numpy() - 6, subset.Va_V.to_numpy()
        for name, columns in (("common", [np.ones(len(n)), n]),
                ("Vr_shift", [np.ones(len(n)), n, z]),
                ("Vr_shift_scale", [np.ones(len(n)), n, z, n * z])):
            design = np.column_stack(columns)
            coefficients = np.linalg.lstsq(design, voltage, rcond=None)[0]
            residual = design @ coefficients - voltage
            regressions.append(dict(window=window, status="descriptive", model=name,
                intercept_V=float(coefficients[0]), spacing_V=float(coefficients[1]),
                shift_V_per_V=float(coefficients[2]) if len(coefficients) > 2 else 0,
                spacing_trend_V_per_V=float(coefficients[3]) if len(coefficients) > 3 else 0,
                rms_V=float(np.sqrt(np.mean(residual ** 2))), points=len(voltage)))
    return features, regressions


def run(output: Path) -> dict:
    torch.set_num_threads(1)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be empty: {output}")
    for package in (EVIDENCE, REFERENCE):
        for line in (package / "SHA256SUMS.txt").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            if sha256_file(package / relative) != expected:
                raise ValueError(f"Frozen evidence byte mismatch: {relative}")
    identity = json.loads((EVIDENCE / "run_identity.json").read_text())
    if identity["sources"] != source_hashes():
        raise ValueError("Frozen kernel source mismatch")
    input_path = ROOT / "data/argon/FHdata.xlsx"
    if identity["input_sha256"] != sha256_file(input_path):
        raise ValueError("Frozen input byte mismatch")
    curves = load_curves(input_path)
    by_vr = {c["Vr"]: c for c in curves}
    nets, receipts, calibrations, scores, points = {}, {}, [], [], []

    def load(unit):
        unit = unit[:20]
        if unit not in nets:
            receipt_path = EVIDENCE / "units" / f"{unit}.json"
            if not receipt_path.exists():
                receipt_path = REFERENCE / "units" / f"{unit}.json"
            receipt = json.loads(receipt_path.read_text())
            spec = receipt["spec"]
            net = EffectiveSpectrum(spec["n_curves"], spec["family"], seed=spec["seed"],
                excitation=spec["excitation"], step=spec["step"], fixed_width=spec["fixed_width"])
            with torch.no_grad():
                for name, parameter in net.named_parameters():
                    parameter.copy_(torch.tensor(receipt["raw_parameter_values"][name]))
            nets[unit], receipts[unit] = net, receipt
        return nets[unit], receipts[unit]

    def evaluate(unit, heldout, modes, **context):
        net, receipt = load(unit)
        train = training_curves([by_vr[v] for v in receipt["spec"]["training_vr"]])
        train_predictions = predict(net, train)
        base = predict(net, [by_vr[heldout]])[0]
        result = []
        for mode in modes:
            calibration = fit_readout(train, train_predictions, mode)
            values = apply_readout(base, heldout, calibration)
            gain, bias = readout_values(calibration, heldout)
            metrics = prediction_metrics(by_vr[heldout]["Ip"], values,
                denominator_observed=by_vr[heldout]["Ip"])
            row = dict(**context, unit=unit[:20], family=receipt["spec"]["family"],
                seed=receipt["spec"]["seed"], heldout_vr=heldout, mode=mode,
                gain=gain, offset_uA=bias, **metrics,
                bias_uA=float(np.mean(values - by_vr[heldout]["Ip"])))
            row.update(morphology(by_vr[heldout]["Va"], by_vr[heldout]["Ip"], values))
            calibrations.append(dict(scope=context["scope"], unit=unit[:20],
                heldout_vr=heldout, **calibration))
            result.append(row)
            if context["scope"] != "inner":
                for va, observed, prediction in zip(by_vr[heldout]["Va"], by_vr[heldout]["Ip"], values):
                    points.append(dict(scope=context["scope"], unit=unit[:20], mode=mode,
                        family=row["family"], seed=row["seed"], Vr=heldout, Va=float(va),
                        observed_uA=float(observed), predicted_uA=float(prediction)))
        scores.extend(result)
        return result

    inner = pd.read_csv(EVIDENCE / "inner_scores.csv")
    smoothing = pd.read_csv(EVIDENCE / "selection.csv")
    smoothing = smoothing[(smoothing.stage == "lambda") & smoothing.selected]
    choices, selection_rows = {}, []
    for selection in smoothing.to_dict("records"):
        scope, penalty = selection["scope"], float(selection["candidate"])
        subset = inner[(inner.scope == scope) & (inner.stage == "continuous")
                       & np.isclose(inner["lambda"], penalty, rtol=1e-10, atol=0)]
        rows = []
        for row in subset.to_dict("records"):
            rows.extend(evaluate(row["unit"], row["fold"], MODES[:2], scope="inner", parent=scope))
        choices[scope], summary = select_readout(rows)
        selection_rows.extend(dict(scope=scope, frozen_lambda=penalty, **r) for r in summary)

    outer = pd.read_csv(EVIDENCE / "outer_scores.csv")
    for row in outer[outer.family.isin(("H1", "G1", "C"))].to_dict("records"):
        result = evaluate(row["unit"], row["heldout_vr"], MODES, scope="outer")
        for value in result:
            value["selected"] = row["family"] == "C" and value["mode"] == choices[row["scope"]]

    # Final readout choice is frozen above before computing the 10 V stress scores.
    final = pd.read_csv(EVIDENCE / "final_training_scores.csv")
    nuisance = []
    for unit in final[final.family.isin(("H1", "G1", "C"))].unit.unique():
        net, receipt = load(unit)
        for i, vr in enumerate(receipt["spec"]["training_vr"]):
            gain, bias, _ = net.nuisance_values(i)
            nuisance.append(dict(family=net.family, seed=receipt["spec"]["seed"], Vr=vr,
                gain=float(gain.detach()), offset_uA=float(bias.detach()),
                phase_V=float(net.phys_params()["phase"].detach())))
        for vr in (*receipt["spec"]["training_vr"], 10.0):
            evaluate(unit, vr, MODES, scope="stress" if vr == 10 else "training_diagnostic")

    features, regressions = feature_audit(curves)
    frame = pd.DataFrame(scores)
    outer_frame = frame[frame.scope == "outer"]
    medians = outer_frame.groupby(["family", "mode", "heldout_vr"], as_index=False).nrmse.median()
    means = medians.groupby(["family", "mode"], as_index=False).nrmse.mean()
    selected = outer_frame[(outer_frame.family == "C") & outer_frame.selected.eq(True)]
    selected_mean = float(selected.groupby("heldout_vr").nrmse.median().mean())
    mode = float(np.median([r["distribution"]["mode_eV"] for r in receipts.values()
        if r["spec"]["family"] == "C" and r["spec"]["n_curves"] == 4]))
    summary = dict(schema="affine-calibration-audit-v1", physical_kernel_retrained=False,
        train_points=644, training_hash=curves_hash(training_curves(curves)),
        choices=choices, outer_mean_seed_median_nrmse=means.to_dict("records"),
        C_nested_selected_mean_nrmse=selected_mean,
        absolute_voltage_scale_identifiable=False, instrument_error_proven=False,
        C_mode_eV_unchanged=mode,
        hypothetical_gain_to_map_C_mode_into_4s=[float(NIST_4S[0] / mode), float(NIST_4S[-1] / mode)],
        linear_vr_is_diagnostic_only=True, stress_has_influenced_selection=False,
        baseline_evidence_sha256=sha256_file(EVIDENCE / "SHA256SUMS.txt"),
        additional_reference_sha256=sha256_file(REFERENCE / "SHA256SUMS.txt"),
        input_sha256=identity["input_sha256"], frozen_kernel_sources=source_hashes(),
        audit_sources={"audit_affine_calibration.py": sha256_file(Path(__file__)),
            "affine_readout.py": sha256_file(ROOT / "src/sublevel_detect/affine_readout.py")})
    output.mkdir(parents=True, exist_ok=True)
    for filename, data in (("scores.csv", frame), ("outer_fold_medians.csv", medians),
            ("selection.csv", pd.DataFrame(selection_rows)), ("prediction_points.csv", pd.DataFrame(points)),
            ("existing_nuisance.csv", pd.DataFrame(nuisance)), ("features.csv", pd.DataFrame(features)),
            ("feature_regressions.csv", pd.DataFrame(regressions))):
        data.to_csv(output / filename, index=False)
    atomic_json_dump(calibrations, output / "readout_parameters.json")
    atomic_json_dump(summary, output / "summary.json")
    (output / "SHA256SUMS.txt").write_text("".join(
        f"{sha256_file(path)}  {path.name}\n" for path in sorted(output.iterdir()) if path.is_file()), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output.resolve()), indent=2))
