from sublevel_detect.self_check import check_project


def test_research_versions_are_allowed_but_neural_version_labels_are_still_blocked(tmp_path):
    research = tmp_path / "NewModel-TEST" / "README.md"
    neural = tmp_path / "src" / "sublevel_detect" / "model.py"
    research.parent.mkdir()
    neural.parent.mkdir(parents=True)
    research.write_text("v" + "2 transport experiment", encoding="utf-8")
    neural.write_text("# v" + "2 obsolete neural label", encoding="utf-8")
    hits = check_project(tmp_path)["blocked_term_hits"]
    assert hits == [{"path": str(neural.relative_to(tmp_path)), "term": "version_numeric_label"}]


def test_standalone_root_inside_output_is_checked_and_test_receipts_are_ignored(tmp_path):
    root = tmp_path / "output" / "checkout"
    source = root / "src" / "model.py"
    receipt = root / ".test-tmp" / "fit.json"
    source.parent.mkdir(parents=True)
    receipt.parent.mkdir(parents=True)
    source.write_text("# v" + "2 obsolete label", encoding="utf-8")
    receipt.write_text('"v' + '2 temporary receipt"', encoding="utf-8")
    assert check_project(root)["blocked_term_hits"] == [
        {"path": str(source.relative_to(root)), "term": "version_numeric_label"}
    ]
