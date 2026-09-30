"""EXPERIMENT_V2 common utilities.

Scope: Track-A diagnostic experiments on the archived Franck-Hertz dataset.
Model definitions are reused unchanged from the frozen FULL-REtry package
(`FULL-REtry/src/fh_retry`); this package only adds diagnostic logic,
variant grids, debiased scoring, and reporting.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]  # .../EXPERIMENT_V2
WORKSPACE_ROOT = HERE.parents[2]  # .../FrankHertz_Experiment
FULL_RETRY_SRC = WORKSPACE_ROOT / "FULL-REtry" / "src"
DATA_PATH = WORKSPACE_ROOT / "FULL-REtry" / "data" / "FHdata.xlsx"
RESULTS_ROOT = PACKAGE_ROOT / "results"

if str(FULL_RETRY_SRC) not in sys.path:
    sys.path.insert(0, str(FULL_RETRY_SRC))

PERIOD_PRIOR = 11.55  # mirror of fh_retry.analysis.PERIOD_PRIOR
STEADY_START_V = 34.0  # plan A1: steady-periodic region starts at the second valley
BLOCK_WIDTH_V = 11.5


def load_pivot():
    """Load and audit the archived dataset, returning the (161, 5) pivot."""
    from fh_retry.analysis import load_and_audit

    _, pivot, audit = load_and_audit(DATA_PATH)
    return pivot, audit


def sha256_frame(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(frame.to_csv(index=True).encode("utf-8"))
    return digest.hexdigest().upper()


@contextmanager
def timed(key: str, sink: dict):
    started = time.perf_counter()
    try:
        yield
    finally:
        sink[key] = time.perf_counter() - started


def write_outputs(result_dir_name: str, tables: dict[str, pd.DataFrame], summary: dict) -> Path:
    """Write CSV tables, a summary JSON (with table hashes, environment, wall time)."""
    import scipy

    out_dir = RESULTS_ROOT / result_dir_name
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts = {}
    for name, frame in tables.items():
        path = out_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        receipts[name] = sha256_frame(frame)
    payload = {
        "summary": summary,
        "table_sha256": receipts,
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pandas": pd.__version__,
        },
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    return out_dir


def block_masks(va: np.ndarray, blocks: list[tuple[float, float]]) -> list[np.ndarray]:
    return [(va >= low) & (va < high) for low, high in blocks]


def original_blocks() -> list[tuple[float, float]]:
    from fh_retry.analysis import BLOCKS

    return list(BLOCKS)


def steady_blocks() -> list[tuple[float, float]]:
    """Plan A1: four full-period blocks starting at the second valley (~34 V)."""
    return [(STEADY_START_V + BLOCK_WIDTH_V * i, STEADY_START_V + BLOCK_WIDTH_V * (i + 1)) for i in range(4)]


def split_first_blocks() -> list[tuple[float, float]]:
    """Plan A1: first half-period [22.5, 28.25) excluded, remainder as its own block, then 4 steady blocks."""
    mid = 22.5 + BLOCK_WIDTH_V / 2
    return [(mid, STEADY_START_V)] + steady_blocks()


BLOCK_DEFINITIONS = {
    "original_5": original_blocks,
    "steady_4": steady_blocks,
    "split_first_5": split_first_blocks,
}
