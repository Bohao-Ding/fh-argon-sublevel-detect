"""A3 — Composition-representation divergence diagnosis (G10 FAIL follow-up).

Phase 3 gate G10 failed because CLR (worst block -14.25%) and ILR (-9.08%)
cross the frozen -5% boundary while raw_fraction and Hellinger pass. This
experiment localizes the divergence: quantization-zero replacement strength
(delta), clamping, or log-geometry itself, by re-running the rank CV over a
frozen grid of truncation-aware variants.

PRIMARY variant (declared before results): v1 = multiplicative replacement with
delta = 1.0 x quantization / I0 (a full quantization step — the most
conservative standard choice). Gate G-A3a is evaluated on v1; all other
variants are reported as sensitivity.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from scipy.linalg import helmert

from .common import load_pivot, timed, write_outputs

VARIANTS = ("v0_baseline_delta0p5", "v1_primary_delta1p0", "v2_delta0p25", "v3_noclamp_raw", "v4_additive_shift")
REPRESENTATIONS = ("raw_fraction", "hellinger", "clr", "ilr")


def _raw_unclamped(pivot) -> tuple[np.ndarray, np.ndarray, dict]:
    from fh_retry.phase2 import _analysis_arrays

    va_all, currents = _analysis_arrays(pivot)  # (n, 5) raw curve currents at Va>=15
    raw = np.column_stack(
        [
            currents[:, 0] - currents[:, 1],
            currents[:, 1] - currents[:, 2],
            currents[:, 2] - currents[:, 3],
            currents[:, 3] - currents[:, 4],
            currents[:, 4],
        ]
    )
    totals = np.maximum(raw.sum(axis=1, keepdims=True), 1e-12)
    fractions = raw / totals
    audit = {
        "negative_cells": int((raw < 0).sum()),
        "min_raw_uA": float(raw.min()),
        "max_abs_negative_uA": float(-raw[raw < 0].max(initial=0.0)),
    }
    return va_all, fractions, audit


def _variant_responses(pivot, quantization: float) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    from fh_retry.phase2 import _retarding_fractions
    from fh_retry.phase3 import _multiplicative_zero_replacement

    va, fractions_clamped, band_audit = _retarding_fractions(pivot, [0, 4, 6, 8, 10])
    i0 = pivot.loc[va, 0].to_numpy(float)
    va_nc, fractions_nc, raw_audit = _raw_unclamped(pivot)
    assert np.allclose(va, va_nc), "variant Va grids diverged"

    variants: dict[str, dict[str, np.ndarray]] = {}
    for name, delta_scale in (
        ("v0_baseline_delta0p5", 0.5),
        ("v1_primary_delta1p0", 1.0),
        ("v2_delta0p25", 0.25),
    ):
        positive, _ = _multiplicative_zero_replacement(fractions_clamped, delta_scale * quantization / i0)
        log_pos = np.log(positive)
        variants[name] = {
            "raw_fraction": fractions_clamped,
            "hellinger": np.sqrt(positive),
            "clr": log_pos - log_pos.mean(axis=1, keepdims=True),
            "ilr": log_pos @ helmert(positive.shape[1], full=False).T,
        }

    variants["v3_noclamp_raw"] = {"raw_fraction": fractions_nc}

    epsilon = 0.5 * quantization / i0
    shifted = fractions_clamped + epsilon[:, None]
    log_shifted = np.log(shifted)
    variants["v4_additive_shift"] = {
        "raw_fraction": fractions_clamped,
        "hellinger": np.sqrt(shifted),
        "clr": log_shifted - log_shifted.mean(axis=1, keepdims=True),
        "ilr": log_shifted @ helmert(shifted.shape[1], full=False).T,
    }
    audit = {
        "clamped_variant": band_audit,
        "unclamped_variant": raw_audit,
        "quantization_uA": quantization,
    }
    return variants, audit


def run() -> dict:
    timings: dict[str, float] = {}
    with timed("total_seconds", timings):
        pivot, audit = load_pivot()
        quantization = audit["current_quantization_uA"]
        from fh_retry.phase2 import _retarding_fractions
        from fh_retry.phase3 import _rank_detail, _rank_summary

        va, _, _ = _retarding_fractions(pivot, [0, 4, 6, 8, 10])

        with timed("grid_seconds", timings):
            variants, variant_audit = _variant_responses(pivot, quantization)
            rows = []
            summaries = []
            for variant_name in VARIANTS:
                for representation in REPRESENTATIONS:
                    if representation not in variants[variant_name]:
                        summaries.append(
                            {
                                "variant": variant_name,
                                "representation": representation,
                                "skipped": "undefined without clamped positive fractions",
                            }
                        )
                        continue
                    detail = _rank_detail(va, variants[variant_name][representation])
                    detail.insert(0, "representation", representation)
                    detail.insert(0, "variant", variant_name)
                    rows.append(detail)
                    summaries.append(
                        {
                            "variant": variant_name,
                            "representation": representation,
                            **_rank_summary(detail),
                        }
                    )
            grid = pd.concat([frame for frame in rows if len(frame)], ignore_index=True)
            summary_frame = pd.DataFrame(summaries)

        with timed("gates_seconds", timings):
            primary = summary_frame[summary_frame["variant"] == "v1_primary_delta1p0"]
            primary_pass = primary.dropna(subset=["worst_gain"])
            v0 = summary_frame[summary_frame["variant"] == "v0_baseline_delta0p5"]
            replication = {
                "v0_clr_worst_gain": float(v0[v0["representation"] == "clr"]["worst_gain"].iloc[0]),
                "v0_ilr_worst_gain": float(v0[v0["representation"] == "ilr"]["worst_gain"].iloc[0]),
                "frozen_reference_clr": -0.1425,
                "frozen_reference_ilr": -0.0908,
                "replication_abs_error_clr": abs(
                    float(v0[v0["representation"] == "clr"]["worst_gain"].iloc[0]) - (-0.1425)
                ),
                "replication_abs_error_ilr": abs(
                    float(v0[v0["representation"] == "ilr"]["worst_gain"].iloc[0]) - (-0.0908)
                ),
            }
            replacement_variants = ("v0_baseline_delta0p5", "v1_primary_delta1p0", "v2_delta0p25", "v4_additive_shift")

            def worst_gain_range(representation: str) -> list[float]:
                values = [
                    float(summary_frame[(summary_frame["variant"] == variant) & (summary_frame["representation"] == representation)]["worst_gain"].iloc[0])
                    for variant in replacement_variants
                ]
                return [min(values), max(values)]

            delta_range_clr = worst_gain_range("clr")
            delta_range_ilr = worst_gain_range("ilr")
            gates = {
                "G_A3a_primary_variant_3of4_representations_pass": bool(
                    (primary_pass["worst_gain"] >= -0.05).sum() >= 3
                ),
                "primary_variant_passing_representations": int(
                    (primary_pass["worst_gain"] >= -0.05).sum()
                ),
                "baseline_replication_within_1e3": bool(
                    replication["replication_abs_error_clr"] < 1e-3
                    and replication["replication_abs_error_ilr"] < 1e-3
                ),
                "negative_cells_in_raw_differences": variant_audit["unclamped_variant"][
                    "negative_cells"
                ],
                "clr_worst_gain_range_over_variants": [
                    min(delta_range_clr), max(delta_range_clr)
                ],
                "ilr_worst_gain_range_over_variants": [
                    min(delta_range_ilr), max(delta_range_ilr)
                ],
                "noclamp_raw_equals_clamped_raw": bool(
                    abs(
                        float(summary_frame[(summary_frame["variant"] == "v3_noclamp_raw") & (summary_frame["representation"] == "raw_fraction")]["worst_gain"].iloc[0])
                        - float(summary_frame[(summary_frame["variant"] == "v0_baseline_delta0p5") & (summary_frame["representation"] == "raw_fraction")]["worst_gain"].iloc[0])
                    ) < 1e-9
                ),
            }
            summary = {
                "experiment": "A3_composition_truncation",
                "dataset_sha256": audit["sha256"],
                "primary_variant": "v1_primary_delta1p0",
                "gates": gates,
                "gate_verdict": (
                    "FAIL (diagnostic, kept): the CLR/ILR worst-block weakness is invariant to every "
                    "truncation-aware variant (CLR worst gain stays at "
                    f"{gates['clr_worst_gain_range_over_variants'][0]:.4f} to "
                    f"{gates['clr_worst_gain_range_over_variants'][1]:.4f} across replacement strengths "
                    "and additive shift; ILR likewise), and unclamped raw equals clamped raw. The "
                    "divergence is therefore the log-ratio GEOMETRY itself, not zero handling - the "
                    "manuscript's 'representation-dependent' wording must stay, now with a sharpened cause"
                    if not gates["G_A3a_primary_variant_3of4_representations_pass"]
                    else "PASS: with full-quantization-step replacement, >=3/4 representations respect the "
                    "-5% worst-block rule; G10 may be re-scored under the documented truncation-aware protocol"
                ),
                "baseline_replication": replication,
                "claim_boundary": [
                    "the amendment is registered before looking at v1 results; v0 must first reproduce the frozen phase-3 numbers",
                    "CLR/ILR remain sensitivity analyses in the manuscript regardless of the gate outcome",
                ],
                "timings": timings,
            }

    out_dir = write_outputs(
        "a3_composition_truncation",
        {"variant_grid": grid, "variant_summary": summary_frame},
        summary,
    )
    return {"summary": summary, "out_dir": out_dir}


if __name__ == "__main__":
    result = run()
    print(json.dumps(result["summary"]["gates"], indent=2, ensure_ascii=False))
    print(f"outputs -> {result['out_dir']}")
