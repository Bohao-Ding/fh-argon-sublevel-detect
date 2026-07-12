from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping


ALLOWED_STAGE_STATUSES = {
    "not_started",
    "smoke_passed",
    "formal_running",
    "complete",
    "incomplete",
    "blocked",
}

LOCAL_BASELINE_FILES = {
    "config": Path("main/fullscan/config_used.json"),
    "decision": Path("main/fullscan/decision.json"),
    "model_selection": Path("main/fullscan/model_selection_table.csv"),
    "forward_evidence": Path("main/fullscan/forward_evidence.json"),
    "selected_scorecard": Path("main/k_selected_full/scorecard.json"),
    "selected_checkpoint": Path("main/k_selected_full/checkpoint_best.pt"),
    "prediction_points": Path("main/k_selected_full/prediction_points.csv"),
}

PACKAGE_BASELINE_FILES = {
    "config": Path("output_results/main/fullscan/config_used.json"),
    "decision": Path("output_results/main/fullscan/decision.json"),
    "model_selection": Path("output_results/main/fullscan/model_selection_table.csv"),
    "forward_evidence": Path("output_results/main/fullscan/forward_evidence.json"),
    "selected_scorecard": Path("run_records/k_selected_full/scorecard.json"),
    "selected_checkpoint": Path("run_records/k_selected_full/checkpoint_best.pt"),
    "prediction_points": Path("run_records/k_selected_full/prediction_points.csv"),
    "channel_parameters": Path("manuscript_source_tables/channel_parameters.csv"),
}


@dataclass(frozen=True)
class Baseline:
    kind: str
    root: Path
    files: dict[str, Path]
    hashes: dict[str, str]

    @property
    def identity_hash(self) -> str:
        return canonical_hash({"kind": self.kind, "hashes": self.hashes})


@dataclass(frozen=True)
class RunIdentity:
    code_commit: str
    code_dirty: bool
    baseline_hash: str
    matrix_hash: str
    seed_hash: str


def _json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def canonical_hash(payload: Any) -> str:
    encoded = json.dumps(
        _json_ready(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def _resolved_files(root: Path, layout: Mapping[str, Path]) -> dict[str, Path]:
    return {key: (root / relative).resolve() for key, relative in layout.items()}


def _missing(files: Mapping[str, Path]) -> list[str]:
    return [key for key, path in files.items() if not path.is_file()]


def resolve_baseline(output_root: str | Path, *, package_root: str | Path) -> Baseline:
    local_root = Path(output_root).expanduser().resolve()
    package = Path(package_root).expanduser().resolve()
    local_files = _resolved_files(local_root, LOCAL_BASELINE_FILES)
    if not _missing(local_files):
        return Baseline(
            kind="local",
            root=local_root,
            files=local_files,
            hashes={key: sha256_file(path) for key, path in local_files.items()},
        )

    package_files = _resolved_files(package, PACKAGE_BASELINE_FILES)
    missing = _missing(package_files)
    if missing:
        details = ", ".join(f"{key}={package_files[key]}" for key in missing)
        raise FileNotFoundError(f"Packaged baseline is incomplete: {details}")
    return Baseline(
        kind="package",
        root=package,
        files=package_files,
        hashes={key: sha256_file(path) for key, path in package_files.items()},
    )


def atomic_json_dump(payload: Any, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(_json_ready(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


def assert_reusable(manifest_path: str | Path, identity: RunIdentity) -> None:
    path = Path(manifest_path)
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    existing = payload.get("identity")
    expected = asdict(identity)
    if existing != expected:
        raise ValueError(
            f"Validation identity mismatch; refusing to reuse or overwrite {path}. "
            f"existing={existing}, expected={expected}"
        )


def write_stage_status(path: str | Path, status: str, **details: Any) -> None:
    if status not in ALLOWED_STAGE_STATUSES:
        raise ValueError(f"Unsupported stage status: {status}")
    atomic_json_dump({"status": status, **details}, path)


def write_progress(
    path: str | Path,
    *,
    completed: int,
    failed: int,
    total: int,
    **details: Any,
) -> None:
    completed = int(completed)
    failed = int(failed)
    total = int(total)
    if min(completed, failed, total) < 0:
        raise ValueError("Progress counts must be non-negative")
    if completed + failed > total:
        raise ValueError("completed + failed exceeds preregistered total")
    atomic_json_dump(
        {
            "completed": completed,
            "failed": failed,
            "pending": total - completed - failed,
            "total": total,
            **details,
        },
        path,
    )


def wilson_interval(successes: int, total: int, *, z: float = 1.959963984540054) -> tuple[float, float]:
    successes = int(successes)
    total = int(total)
    if total <= 0:
        raise ValueError("Wilson interval requires total > 0")
    if successes < 0 or successes > total:
        raise ValueError("Wilson successes must satisfy 0 <= successes <= total")
    proportion = successes / total
    z2 = float(z) ** 2
    denominator = 1.0 + z2 / total
    center = (proportion + z2 / (2.0 * total)) / denominator
    half_width = (
        float(z)
        * math.sqrt(proportion * (1.0 - proportion) / total + z2 / (4.0 * total**2))
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)
