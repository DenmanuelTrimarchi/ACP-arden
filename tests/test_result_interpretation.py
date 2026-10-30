"""Saved-evidence interpretation and development/test separation regressions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

import ACP_arden as acp


def write(root: Path, name: str, payload: dict[str, Any]) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def policy() -> dict[str, Any]:
    return {
        "status": acp.OPEN_SET_STATUS_FROZEN,
        "primary_fpir_target": 0.003,
        "operating_points": {
            str(target): {"target_fpir": target, "threshold": threshold,
                          "development_fpir": target / 2, "development_tpir_rank1": 0.99}
            for target, threshold in ((0.001, 0.8), (0.003, 0.6), (0.01, 0.4))
        },
    }


def observed(threshold: float, tpir: float, fpir: float) -> dict[str, Any]:
    return {"threshold": threshold, "tpir_rank1": tpir, "fpir": fpir,
            "scored_mated_probes": 8, "scored_non_mated_probes": 20}


def test_sensitivity_keeps_development_and_test_rates_and_intended_denominator(tmp_path: Path) -> None:
    write(tmp_path, "bfw_open_set_threshold.json", policy())
    method = {"coverage": {"intended_mated_probes": 10}, "operating_points": {
        "0.001": observed(0.8, 0.25, 0),
        "0.003": observed(0.6, 0.75, 0.05),
        "0.01": observed(0.4, 1.0, 0.1),
    }}
    write(tmp_path, "bfw_open_set_test_metrics.json", {"methods": {acp.METHOD_B: method}})
    rows = acp.threshold_sensitivity_rows(tmp_path)
    primary = rows[1]
    assert len(rows) == 12
    assert primary["development_target_fpir"] == 0.003
    assert primary["development_fpir"] == 0.0015
    assert primary["test_fpir"] == 0.05
    assert primary["false_reviews_per_1000_scored_non_mated"] == 50
    assert primary["correctly_detected_mated_probes"] == 6
    assert primary["end_to_end_duplicate_detection_rate"] == 0.6  # Not conditional 0.75.
    rendered = acp.render_threshold_sensitivity(tmp_path)
    assert "6 / 10 (60.00%)" in rendered
    assert "not an empirically validated moderation requirement" in rendered
    assert "not a new paired significance analysis" in rendered
    assert "not reselected using test outcomes" in rendered
    table = [line for line in rendered.splitlines() if line.startswith("|")]
    assert len(table) == 14 and "\n\n|" not in "\n".join(table)
    assert "\n".join(table) in rendered


def test_rounded_conditional_rate_cannot_invent_an_observed_detection_count(tmp_path: Path) -> None:
    write(tmp_path, "bfw_open_set_threshold.json", policy())
    write(tmp_path, "bfw_open_set_test_metrics.json", {"methods": {acp.METHOD_B: {
        "coverage": {"intended_mated_probes": 10},
        "primary_operating_point": observed(0.6, 0.67, 0.05)}}})
    primary = acp.threshold_sensitivity_rows(tmp_path)[1]
    assert primary["test_tpir_rank1"] == 0.67
    assert primary["correctly_detected_mated_probes"] is None
    assert primary["end_to_end_duplicate_detection_rate"] is None


def test_comparator_does_not_turn_development_points_or_histograms_into_test_evidence(tmp_path: Path) -> None:
    comparator = {"threshold_status": acp.OPEN_SET_STATUS_FROZEN,
                  "operating_points": policy()["operating_points"],
                  "rates": observed(0.6, 0.75, 0.05),
                  "coverage": {"intended_mated_probes": 10},
                  "similarity_histograms": {"mated_correct_identity": {"counts": [8]}}}
    write(tmp_path, "pipeline_comparison_metrics.json", {"held_out_metrics": {
        "insightface-scrfd-arcface-buffalo_l": comparator}})
    before = (tmp_path / "pipeline_comparison_metrics.json").read_bytes()
    rows = acp.threshold_sensitivity_rows(tmp_path)[-3:]
    assert [r["test_status"] for r in rows] == ["not available", "available", "not available"]
    assert rows[0]["development_fpir"] == 0.0005
    assert rows[0]["test_tpir_rank1"] is None
    assert rows[2]["test_fpir"] is None
    assert rows[1]["end_to_end_duplicate_detection_rate"] == 0.6
    assert (tmp_path / "pipeline_comparison_metrics.json").read_bytes() == before


@pytest.mark.parametrize("unfrozen", [False, True])
def test_unmatched_or_unfrozen_policy_does_not_claim_held_out_evidence(tmp_path: Path, unfrozen: bool) -> None:
    selected = policy()
    if unfrozen:
        selected["status"] = "candidate"
    write(tmp_path, "bfw_open_set_threshold.json", selected)
    write(tmp_path, "bfw_open_set_test_metrics.json", {"methods": {acp.METHOD_B: {
        "coverage": {"intended_mated_probes": 10},
        "primary_operating_point": observed(0.5, 1.0, 0.0)}}})
    primary = acp.threshold_sensitivity_rows(tmp_path)[1]
    assert primary["test_status"] == "not available"
    assert primary["test_fpir"] is None
    assert primary["end_to_end_duplicate_detection_rate"] is None


def synthetic_run(partition: str) -> acp.OpenSetRunResult:
    group = acp.BFW_SUBGROUPS[0]
    rows = [
        acp.OpenSetSearchResult("mate1", "known1", group, "mated_probe",
                               top_similarity=0.9, correct_rank=1, correct_similarity=0.9),
        acp.OpenSetSearchResult("mate2", "known2", group, "mated_probe",
                               failure_code="zero_faces"),
        acp.OpenSetSearchResult("new1", "unknown1", group, "non_mated_probe", top_similarity=0.2),
        acp.OpenSetSearchResult("new2", "unknown2", group, "non_mated_probe", top_similarity=0.4),
    ]
    return acp.OpenSetRunResult(acp.METHOD_B, partition, [], rows, 2, 2)


def test_future_comparison_saves_all_test_points_and_preserves_primary_without_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    thresholds = {0.001: 0.95, 0.003: 0.7, 0.01: 0.3}
    calls: list[float] = []

    def select(rows: Any, *, target_fpir: float) -> dict[str, Any]:
        calls.append(target_fpir)
        return {"threshold": thresholds[target_fpir], "target_fpir": target_fpir,
                "development_fpir": 0.0}

    def refuse_inference(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("Canonical runs must not trigger fresh model inference")

    monkeypatch.setattr(acp, "select_open_set_threshold", select)
    monkeypatch.setattr(acp, "run_open_set_method", refuse_inference)
    test = synthetic_run("test")
    result = acp.evaluate_one_pipeline_for_comparison(
        acp.OpenSetProtocol([], 42, {}), name="synthetic",
        detector=cast(acp.FaceDetector, object()), embedder=cast(acp.FaceEmbedder, object()),
        canonical_development_run=synthetic_run("development"), canonical_test_run=test,
        bootstrap_replicates=10,
    )
    assert calls == list(acp.FPIR_TARGETS)
    assert set(result["test_operating_points"]) == {"0.001", "0.003", "0.01"}
    primary = result["test_operating_points"]["0.003"]
    assert {key: primary[key] for key in result["rates"]} == result["rates"]
    assert primary["end_to_end_duplicate_detection_rate"] == result["end_to_end_duplicate_detection_rate"] == 0.5
    assert result["test_operating_points"]["0.001"]["end_to_end_duplicate_detection_rate"] == 0
    assert result["test_operating_points"]["0.01"]["fpir"] == 0.5


def test_protocol_summary_records_safe_recipe_and_actual_subgroup_identity_counts() -> None:
    images = [acp.BfwImage(Path(group) / f"id{i}" / f"{j:02d}.jpg", f"{group}/id{i}", group,
                           f"{group}-{i}-{j}", f"{group}-{i}")
              for group in acp.BFW_SUBGROUPS for i in range(4) for j in range(10)]
    protocol = acp.build_open_set_protocol(acp.BfwDataset(images, "digest", "metadata.csv"),
                                          seed=123, mated_probes_per_identity=2,
                                          non_mated_probes_per_identity=3)
    summary = acp.open_set_protocol_summary(protocol)
    recipe = summary["construction_recipe"]
    assert recipe["mated_probes_per_identity"] == 2
    assert recipe["non_mated_probes_per_identity_maximum"] == 3
    assert "never HMAC" in recipe["identifier_key_dependency"]
    assert summary["seed"] == 123
    assert summary["test"]["roles"]["mated_probe"] == 16
    assert all(counts["non_mated_probe"] == 1
               for counts in summary["test"]["identity_roles_by_subgroup"].values())
    metadata_only = acp.OpenSetProtocol(protocol.entries, protocol.seed, {"new_public_metadata": True})
    assert acp.public_manifest_digest(protocol) == acp.public_manifest_digest(metadata_only)
    assert acp.private_protocol_context_digest(protocol) == acp.private_protocol_context_digest(metadata_only)


def test_methodology_uses_saved_counts_and_distinguishes_configurations_and_fairness(tmp_path: Path) -> None:
    counts = {"identities": 18, "mated_identities": 7, "non_mated_identities": 11,
              "roles": {"gallery_enrolment": 21, "mated_probe": 35, "non_mated_probe": 165},
              "images": 221, "identity_roles_by_subgroup": {"synthetic_group": {"non_mated_probe": 11}}}
    write(tmp_path, "bfw_open_set_protocol_summary.json", {
        "seed": 987, "protocol_version": "fixture-v2", "public_manifest_sha256": "public-digest",
        "development": counts, "test": counts,
    })
    text = acp.render_methodological_scope(tmp_path)
    assert "seed: 987" in text and "fixture-v2" in text
    assert "18 identities, 7 gallery/mated identities, 11 non-mated identities" in text
    assert "165 intended non-mated probes, 221 selected images" in text
    assert "synthetic_group: 11" in text
    assert "not an official BFW identification benchmark" in text
    assert "not an isolated network architecture effect" in text
    assert "population shift prevent deployment fairness claims" in text
    assert "200 non-mated" not in text


def test_report_includes_methodology_sensitivity_and_evidence_boundaries(tmp_path: Path) -> None:
    report = acp.render_research_report(tmp_path)
    assert "Dataset selection and protocol justification" in report
    assert "Frozen-threshold sensitivity and moderation requirements" in report
    assert "Evidence boundaries and statistical interpretation" in report
    assert "not evidence of equivalence or no effect" in report
    assert "Claims requiring further investigation" in report
    assert "not available" in report
    assert "Development FPIR budgets" in report


def test_detector_analysis_publishes_the_research_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(acp, "run_detector_ablation", lambda output_root: {})
    monkeypatch.setattr(acp, "write_detector_ablation_outputs", lambda *args: None)
    assert acp.action_run_detector_ablation(tmp_path) == 0
    assert {path.name for path in tmp_path.glob("*.md")} == {"RESEARCH_REPORT.md"}
    report = (tmp_path / "RESEARCH_REPORT.md").read_text(encoding="utf-8")
    assert "Dataset selection and protocol justification" in report
    assert "Frozen-threshold sensitivity and moderation requirements" in report
