import importlib.util
from pathlib import Path
from shutil import copyfile, copytree

import pytest


def _verifier():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_research_evidence.py"
    spec = importlib.util.spec_from_file_location("verify_research_evidence", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_published_evidence_is_complete_and_matches_primary_claims():
    report = _verifier().verify()
    assert report["h4s_complete_units"] == 60
    assert report["A4"] == "NOT_EXECUTED"


def test_modified_receipt_fails_before_analysis(tmp_path):
    evidence = tmp_path / "source_data_package/research_evidence"
    evidence.mkdir(parents=True)
    (evidence / "receipt.json").write_text("changed", encoding="utf-8")
    (evidence / "SHA256SUMS.txt").write_text("0" * 64 + "  receipt.json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="evidence SHA-256 mismatch: receipt.json"):
        _verifier().verify(tmp_path)


def test_legacy_checkout_newlines_do_not_change_frozen_input_identity(tmp_path):
    root = Path(__file__).resolve().parents[1]
    copytree(root / "source_data_package/research_evidence", tmp_path / "source_data_package/research_evidence")
    files = (
        "data/argon/FHdata.xlsx", "FULL-REtry/data/FHdata.xlsx", "NewModel-TEST/data/FHdata.xlsx",
        "source_data_package/output_results/sensitivity/uncertainty/uncertainty_selection_summary.csv",
        "source_data_package/output_results/main/fullscan/model_selection_table.csv",
    )
    for relative in files:
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(root / relative, target)
    legacy = tmp_path / files[-1]
    legacy.write_bytes(legacy.read_bytes().replace(b"\r\n", b"\n"))
    assert _verifier().verify(tmp_path)["h4s_complete_units"] == 60
