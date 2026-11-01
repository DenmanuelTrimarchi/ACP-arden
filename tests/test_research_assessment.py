"""Evidence classification, provenance and research-summary regressions."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ACP_arden as acp


def save(root: Path, name: str, data: dict) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def comparison(lower: float = 0.01, upper: float = 0.07) -> dict:
    interval = {"estimate": 0.04, "lower_95": lower, "upper_95": upper,
                "valid_replicates": 100, "requested_replicates": 100}
    return {"public_manifest_sha256": "shared", "replicates": 100, "analyses": {
        "full_protocol": {"pipelines": {}, "paired_differences": [{
            "left": "YuNet + SFace", "right": "SCRFD + ArcFace", "direction": "right minus left",
            "metrics": {"end_to_end_duplicate_detection_rate": interval,
                        "fpir": {"estimate": -0.001, "lower_95": -0.003,
                                 "upper_95": 0.002, "valid_replicates": 100}}}]}}}


def test_paired_direction_requires_interval_support_and_unresolved_is_not_equivalence(tmp_path: Path) -> None:
    save(tmp_path, "comparative_statistics.json", comparison())
    assessment = acp.build_research_assessment(tmp_path)
    detection, fpir = assessment["paired_claims"]
    assert detection["evidence_class"] == "exploratory_paired_support"
    assert detection["estimate"] == 0.04
    assert fpir["evidence_class"] == "difference_unresolved"
    text = acp.render_research_assessment(tmp_path)
    assert "+4.00 [+1.00, +7.00]" in text
    assert "Difference unresolved; interval includes zero" in text
    assert "not equivalence" in fpir["limitation"]
    assert "prespecified winner" in text


@pytest.mark.parametrize("lower,upper", [(0.08, 0.01), (float("nan"), 0.07), (0.01, float("inf"))])
def test_invalid_intervals_cannot_support_a_direction(tmp_path: Path, lower: float, upper: float) -> None:
    save(tmp_path, "comparative_statistics.json", comparison(lower, upper))
    claim = acp.build_research_assessment(tmp_path)["paired_claims"][0]
    assert claim["evidence_class"] == "descriptive_only"
    assert claim["lower_95"] is None and claim["upper_95"] is None


def test_conflicting_manifest_aliases_withhold_paired_support(tmp_path: Path) -> None:
    save(tmp_path, "comparative_statistics.json", comparison())
    save(tmp_path, "pipeline_comparison_metrics.json", {
        "public_manifest_digest": "shared", "protocol_digest": "different"})
    assessment = acp.build_research_assessment(tmp_path)
    assert assessment["provenance_check"]["status"] == "insufficient_or_conflicting_saved_manifests"
    assert all(row["evidence_class"] == "descriptive_only" for row in assessment["paired_claims"])
    assert "Cross-pipeline paired support is withheld" in acp.render_research_assessment(tmp_path)


def test_observed_verification_and_calibration_are_read_from_their_own_denominators(tmp_path: Path) -> None:
    save(tmp_path, "lfw_final_metrics.json", {"mean_fold_accuracy": 0.95, "failure_rate": 0.2,
                                           "end_to_end_correct_decision_rate": 0.76})
    save(tmp_path, "bfw_open_set_test_metrics.json", {"methods": {acp.METHOD_B: {
        "at_lfw_control_threshold": {"fpir": 0.12, "tpir_rank1": 0.96},
        "primary_operating_point": {"fpir": 0.01, "tpir_rank1": 0.9}}}})
    text = acp.render_research_assessment(tmp_path)
    assert "95.00%" in text and "coverage 80.00%" in text
    assert "across all intended pairs 76.00%" in text
    assert "12.00%" in text and "1.00% and 90.00%" in text
    assert "same three-image BFW templates" in text


def test_research_argument_precedes_expandable_measurements_and_uses_primary_sources(tmp_path: Path) -> None:
    report = acp.render_research_report(tmp_path)
    assert report.index("## Research assessment") < report.index("<details>") < report.index("## 1. LFW")
    assert report.count("<details>") == report.count("</details>") == 1
    assert "central target" in report and "additional application policy" in report
    assert "no isolated architecture or training-loss effect" in report
    assert "deployment fairness" in report and "not a fresh independent benchmark replication" in report
    for name, url in acp.RESEARCH_PRIMARY_SOURCES.items():
        assert f"[{name}]({url})" in report
    assert "Paired comparative evidence is not available" in report


def test_read_only_research_summary_does_not_write_or_analyse_private_inputs(tmp_path: Path, monkeypatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("A saved research summary must not run private-data analysis")
    monkeypatch.setattr(acp, "generate_threshold_sensitivity", forbidden)
    assert acp.main(["--mode", "research-summary", "--results-root", str(tmp_path)]) == 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status,expected", [("evaluated", 0), ("partial", 1), ("not_available", 1)])
def test_research_evaluation_reports_incomplete_evidence_without_claiming_completion(tmp_path: Path, monkeypatch, status, expected) -> None:
    calls = []
    monkeypatch.setattr(acp, "generate_threshold_sensitivity", lambda root: calls.append(root) or {"status": status})
    assert acp.main(["--mode", "research-evaluation", "--results-root", str(tmp_path)]) == expected
    assert calls == [tmp_path]
    assert (tmp_path / "research_assessment.json").is_file()
    assert (tmp_path / "RESEARCH_REPORT.md").is_file()
    assert not any("feedback" in path.name.lower() or "guide" in path.name.lower() for path in tmp_path.iterdir())
