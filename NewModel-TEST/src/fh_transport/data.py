from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


REQUIRED_COLUMNS = ("curve_id", "Vr", "Va", "IuA")


@dataclass(frozen=True)
class CurveData:
    frame: pd.DataFrame
    va: torch.Tensor
    vr: torch.Tensor
    current: torch.Tensor
    curve_index: torch.Tensor
    curve_ids: tuple[int, ...]
    curve_ranges: torch.Tensor
    zero_mask: torch.Tensor


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Data file not found: {source}")
    if source.suffix.lower() in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    elif source.suffix.lower() == ".csv":
        frame = pd.read_csv(source)
    else:
        raise ValueError(f"Unsupported data format: {source.suffix}")
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    clean = frame.loc[:, REQUIRED_COLUMNS].copy()
    for name in REQUIRED_COLUMNS:
        clean[name] = pd.to_numeric(clean[name], errors="raise")
    clean = clean.sort_values(["curve_id", "Va"]).reset_index(drop=True)
    if clean.duplicated(["curve_id", "Va"]).any():
        raise ValueError("Duplicate (curve_id, Va) rows are not allowed")
    return clean


def load_curve_data(path: str | Path, device: torch.device) -> CurveData:
    frame = read_frame(path)
    curve_ids = tuple(int(value) for value in sorted(frame["curve_id"].unique()))
    id_to_index = {curve_id: index for index, curve_id in enumerate(curve_ids)}
    indices = frame["curve_id"].map(id_to_index).to_numpy(dtype=np.int64)
    ranges = []
    for curve_id in curve_ids:
        values = frame.loc[frame["curve_id"] == curve_id, "IuA"].to_numpy(dtype=np.float32)
        ranges.append(max(float(np.ptp(values)), 1.0e-4))
    current = torch.as_tensor(frame["IuA"].to_numpy(dtype=np.float32), device=device)
    return CurveData(
        frame=frame,
        va=torch.as_tensor(frame["Va"].to_numpy(dtype=np.float32), device=device),
        vr=torch.as_tensor(frame["Vr"].to_numpy(dtype=np.float32), device=device),
        current=current,
        curve_index=torch.as_tensor(indices, dtype=torch.long, device=device),
        curve_ids=curve_ids,
        curve_ranges=torch.as_tensor(ranges, dtype=torch.float32, device=device),
        zero_mask=current <= 1.0e-3,
    )

