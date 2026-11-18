"""Retrospective analysis of frozen policies and retained canonical outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import ACP_arden as a


def run(partition: str) -> a.OpenSetRunResult:
    group = a.BFW_SUBGROUPS[0]
    rows = [
        a.OpenSetSearchResult(f"{partition}-m1", f"{partition}-known1", group, "mated_probe",
                              top_similarity=.95, correct_rank=1, correct_similarity=.95),
        a.OpenSetSearchResult(f"{partition}-m2", f"{partition}-known1", group, "mated_probe",
                              top_similarity=.75, correct_rank=1, correct_similarity=.75),
        a.OpenSetSearchResult(f"{partition}-m3", f"{partition}-known2", group, "mated_probe",
                              top_similarity=.96, correct_rank=2, correct_similarity=.85),
        a.OpenSetSearchResult(f"{partition}-m4", f"{partition}-known2", group, "mated_probe",
                              failure_code="zero_faces"),
        a.OpenSetSearchResult(f"{partition}-n1", f"{partition}-new1", group, "non_mated_probe", top_similarity=.1 if partition == "development" else .5),
        a.OpenSetSearchResult(f"{partition}-n2", f"{partition}-new1", group, "non_mated_probe", top_similarity=.2),
        a.OpenSetSearchResult(f"{partition}-n3", f"{partition}-new2", group, "non_mated_probe", top_similarity=.25 if partition == "development" else .4),
        a.OpenSetSearchResult(f"{partition}-n4", f"{partition}-new2", group, "non_mated_probe", failure_code="image_error"),
    ]
    return a.OpenSetRunResult(a.METHOD_B, partition, [], rows, 2, 2)


@pytest.fixture()
def sources(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    root = tmp_path / "aggregate"
    bases: dict[str, Path] = {}
    for index, (name, policy_root, evidence_relative, _, pair) in enumerate(a._threshold_sensitivity_sources(root)):
        base = tmp_path / "private" / f"pipeline{index}.json"
        bases[name] = base
        provenance = {}
        for partition in ("development", "test"):
            outcome = run(partition)
            context = {"partition": partition, "public_manifest_digest": "same-public-manifest",
                       "model_sha256": {"yunet": pair[0], "sface": pair[1]},
                       "dataset_metadata_sha256": "metadata-fingerprint", "evaluated_image_set_sha256": "image-fingerprint",
                       "mated_probe_samples": sorted(r.sample_id for r in outcome.search_results if r.role == "mated_probe"),
                       "non_mated_probe_samples": sorted(r.sample_id for r in outcome.search_results if r.role == "non_mated_probe")}
            path = a.canonical_cache_path(partition, base)
            digest = a.save_canonical_run(outcome, path, context)
            provenance[f"canonical_{partition}_run_digest"] = digest
            provenance[f"canonical_{partition}_context_sha256"] = a.context_digest(context)
        points = {}
        for target, threshold in zip(a.FPIR_TARGETS, (.9, .6, .3)):
            rates = a.open_set_rates_at_threshold(run("development").search_results, threshold)
            points[str(target)] = {"target_fpir": target, "threshold": threshold,
                                   "development_fpir": rates["fpir"], "development_tpir_rank1": rates["tpir_rank1"],
                                   "development_tpir_rank5": rates["tpir_rank5"]}
        policy = {"status": a.OPEN_SET_STATUS_FROZEN, "primary_fpir_target": a.PRIMARY_FPIR_TARGET,
                  "public_manifest_sha256": "same-public-manifest", "operating_points": points,
                  "model_sha256": {"detector": pair[0], "recognition": pair[1]},
                  "dataset_provenance": {"metadata_sha256": "metadata-fingerprint", "evaluated_image_set_sha256": "image-fingerprint"}}
        a.write_json_artifact(policy_root / "bfw_open_set_threshold.json", policy)
        primary = a._threshold_test_counts(run("test"), .6)
        evidence: dict[str, Any] = {**provenance, "public_manifest_sha256": "same-public-manifest"}
        if name == "SCRFD + ArcFace":
            evidence["comparator_three_image_open_set_calibrated"] = {"rates": primary,
                "end_to_end_duplicate_detection_rate": primary["end_to_end_duplicate_detection_rate"]}
            evidence["classifier"] = {"intended_mated_probes": 4, "intended_non_mated_probes": 4}
        else:
            evidence["methods"] = {a.METHOD_B: {"primary_operating_point": primary,
                "coverage": {"intended_mated_probes": 4, "intended_non_mated_probes": 4},
                "end_to_end_duplicate_detection_rate": primary["end_to_end_duplicate_detection_rate"]}}
        a.write_json_artifact(root / evidence_relative, evidence)
    return root, bases


def test_backfill_uses_frozen_points_without_inference_selection_or_source_changes(sources, monkeypatch):
    root, bases = sources
    before = {p: p.read_bytes() for p in root.rglob("*.json")}

    def forbidden(*args, **kwargs):
        pytest.fail("A retrospective analysis cannot fit, select thresholds or run inference")

    for name in ("select_open_set_threshold", "run_open_set_method", "load_models", "load_arcface_pipeline", "load_image_bgr"):
        monkeypatch.setattr(a, name, forbidden)
    result = a.generate_threshold_sensitivity(root, cache_bases=bases, replicates=100)
    repeat = a.generate_threshold_sensitivity(root, cache_bases=bases, replicates=100)
    assert repeat["analysis_content_sha256"] == result["analysis_content_sha256"]
    assert result["status"] == "evaluated"
    assert all(p.read_bytes() == content for p, content in before.items())
    assert {p.name for p in root.glob("threshold*.json")} == {a.THRESHOLD_SENSITIVITY_FILENAME}
    comparator = result["pipelines"]["SCRFD + ArcFace"]
    assert set(comparator["test_operating_points"]) == {"0.001", "0.003", "0.01"}
    primary = comparator["test_operating_points"]["0.003"]
    assert primary["mated_correct_rank1_referred"] == 2
    assert primary["tpir_rank1"] == pytest.approx(2/3)
    assert primary["end_to_end_duplicate_detection_rate"] == .5
    assert primary["mated_wrong_identity_referred"] == 1
    assert primary["mated_correct_rank5_referred"] == 3
    assert primary["intended_mated_probes"] == 4 and primary["scored_mated_probes"] == 3
    loose = comparator["test_operating_points"]["0.01"]
    assert loose["development_fpir"] == 0 and loose["fpir"] == pytest.approx(2/3)
    assert loose["test_fpir_exceeds_development_target"] is True
    assert loose["non_mated_incorrectly_referred"] == 2
    assert loose["false_reviews_per_1000_intended_non_mated"] == 500
    assert loose["false_reviews_per_1000_non_mated"] == pytest.approx(2000/3)
    for partition in ("development", "test"):
        assert comparator["canonical_run_digests"][partition]
    text = (root / a.THRESHOLD_SENSITIVITY_FILENAME).read_text()
    assert "test-known1" not in text and "development-m1" not in text and str(bases["YuNet + SFace"]) not in text


def test_paired_budget_uncertainty_retains_identity_clusters_and_distinguishes_rank(sources):
    root, bases = sources
    result = a.generate_threshold_sensitivity(root, cache_bases=bases, replicates=2000)
    changes = result["paired_budget_differences_by_pipeline"]["YuNet + SFace"]
    change = next(row for row in changes if row["left"] == "0.001" and row["right"] == "0.003")
    detection = change["metrics"]["end_to_end_duplicate_detection_rate"]
    assert detection["estimate"] == .25
    assert (detection["lower_95"], detection["upper_95"]) == (0, .5)
    assert change["metrics"]["fpir"]["estimate"] == 0
    strict = result["pipelines"]["YuNet + SFace"]["test_operating_points"]["0.001"]
    assert strict["mated_correct_rank1_referred"] == strict["mated_correct_rank5_referred"] == 1
    assert strict["confidence_intervals"]["fpir"]["zero_event_identity_bound"]["independent_identities"] == 2
    # Identical pipelines share identical draws, so every paired change is zero.
    for budget in result["paired_pipeline_comparisons_by_budget"].values():
        for comparison in budget["analyses"]["full_protocol"]["paired_differences"]:
            assert comparison["metrics"]["end_to_end_duplicate_detection_rate"]["lower_95"] == 0
            assert comparison["metrics"]["end_to_end_duplicate_detection_rate"]["upper_95"] == 0


def test_missing_cache_produces_an_honest_partial_result(sources):
    root, bases = sources
    a.canonical_cache_path("test", bases["SCRFD + ArcFace"]).unlink()
    result = a.generate_threshold_sensitivity(root, cache_bases=bases, replicates=20)
    assert result["status"] == "partial"
    assert result["pipelines"]["SCRFD + ArcFace"]["status"] == "not_available"
    assert "test_operating_points" not in result["pipelines"]["SCRFD + ArcFace"]


@pytest.mark.parametrize("fault", ["cache_score", "published_digest", "policy_status", "development_rate", "intended_denominator"])
def test_invalid_or_unbound_evidence_cannot_supply_new_test_points(sources, fault):
    root, bases = sources
    policy_path = root / a.ARCFACE_REVIEW_DIRNAME / "bfw_open_set_threshold.json"
    evidence_path = root / a.ARCFACE_REVIEW_DIRNAME / "ml_review_test_metrics.json"
    if fault == "cache_score":
        path = a.canonical_cache_path("test", bases["SCRFD + ArcFace"])
        payload = json.loads(path.read_text()); payload["search_results"][0]["correct_similarity"] = .1
    elif fault in ("policy_status", "development_rate"):
        path = policy_path; payload = json.loads(path.read_text())
        if fault == "policy_status": payload["status"] = "candidate"
        else: payload["operating_points"]["0.001"]["development_tpir_rank1"] = 1.0
    else:
        path = evidence_path; payload = json.loads(path.read_text())
        if fault == "published_digest": payload["canonical_test_run_digest"] = "different"
        else: payload["classifier"]["intended_mated_probes"] = 3
    path.write_text(json.dumps(payload))
    result = a.generate_threshold_sensitivity(root, cache_bases=bases, replicates=20)
    assert result["pipelines"]["SCRFD + ArcFace"]["status"] == "invalid"
    assert "test_operating_points" not in result["pipelines"]["SCRFD + ArcFace"]


def test_report_consumes_exact_cache_counts_but_rejects_stale_source_fingerprints(sources):
    root, bases = sources
    a.generate_threshold_sensitivity(root, cache_bases=bases, replicates=20)
    rows = a.threshold_sensitivity_rows(root)
    assert len(rows) == 12 and all(r["test_status"] == "available" for r in rows)
    assert all(r["analysis_source"] == "validated_canonical_outcomes" for r in rows)
    assert rows[-3]["correctly_detected_mated_probes"] == 1
    rendered = a.render_threshold_sensitivity(root)
    assert "Within-pipeline changes" in rendered and "identity-bootstrap draws" in rendered
    assert "unavailable point" in rendered
    policy_path = root / a.ARCFACE_REVIEW_DIRNAME / "bfw_open_set_threshold.json"
    policy_path.write_text(policy_path.read_text() + " ")
    comparator = a.threshold_sensitivity_rows(root)[-3:]
    assert comparator[0]["test_status"] == comparator[2]["test_status"] == "not available"
    assert comparator[0]["confidence_intervals"] is None


def test_empty_workspace_does_not_invent_threshold_results(tmp_path):
    result = a.generate_threshold_sensitivity(tmp_path, replicates=20)
    assert result["status"] == "not_available"
    assert result["paired_pipeline_comparisons_by_budget"] == {}
    assert all(node["status"] == "not_available" for node in result["pipelines"].values())


def test_modified_analysis_values_are_not_accepted_as_validated_cache_evidence(sources):
    root, bases = sources
    a.generate_threshold_sensitivity(root, cache_bases=bases, replicates=20)
    path = root / a.THRESHOLD_SENSITIVITY_FILENAME
    payload = json.loads(path.read_text())
    payload["pipelines"]["SCRFD + ArcFace"]["test_operating_points"]["0.001"]["mated_correct_rank1_referred"] = 4
    path.write_text(json.dumps(payload))
    assert a._current_threshold_sensitivity(root) == {}
    assert a.threshold_sensitivity_rows(root)[-3]["test_status"] == "not available"
