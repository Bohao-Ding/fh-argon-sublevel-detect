"""Regression checks for A7's portable, receipt-backed evidence inputs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from exp_v2 import a7_evidence_summary as a7


def test_a7_reads_all_folds_and_transport_gates_from_repository():
    folds, metadata = a7._c2_artifacts()
    assert folds.heldout_vr.tolist() == [0, 4, 6, 8, 10]
    assert folds.delta_nrmse.notna().all()
    assert folds.iloc[-1].delta_nrmse > 0
    assert metadata["fold_source"] == "fresh_read"
    assert metadata["v2l"]["h1_gates"] == "15/15"
    assert metadata["v2l"]["h4s_gates"] == "12/15"
    assert metadata["v2l"]["top_channel_fraction"] == pytest.approx(0.913614273)


@pytest.mark.parametrize("source", ["H4S_CONTRASTS_PATH", "V2L_SUMMARY_PATH"])
def test_missing_receipt_cannot_be_replaced_by_documented_values(monkeypatch, tmp_path, source):
    monkeypatch.setattr(a7, source, tmp_path / "missing_receipt")
    with pytest.raises(FileNotFoundError):
        a7._c2_artifacts()
