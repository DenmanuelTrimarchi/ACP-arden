"""Acceptance tests for canonical reuse, profile-consistency rates, the
generated reports, the README and the pinned comparison installation.

These check what the published artefacts and documents actually say, so a
figure or claim cannot drift away from the data behind it.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import pytest

import ACP_arden as acp

ROOT = Path(acp.__file__).parent
AGG = ROOT / "results" / "aggregate"
FIG = ROOT / "results" / "figures"


def _load(name: str):
    path = AGG / name
    if not path.is_file():
        pytest.skip(f"{name} not present in this checkout")
    return json.loads(path.read_text(encoding="utf-8"))


def _text(name: str) -> str:
    path = ROOT / name if (ROOT / name).is_file() else AGG / name
    if not path.is_file():
        pytest.skip(f"{name} not present in this checkout")
    return path.read_text(encoding="utf-8")


# --- Canonical reuse across experiments ---------------------------------------


def test_all_primary_frozen_thresholds_agree() -> None:
    open_set = _load("bfw_open_set_test_metrics.json")
    review = _load("ml_review_test_metrics.json")
    pipeline = _load("pipeline_comparison_metrics.json")
    six = open_set["operating_threshold"]
    seven = review["comparator_three_image_open_set_calibrated"]["operating_threshold"]
    assert six == pytest.approx(seven, abs=0.0)
    held_out = pipeline.get("held_out_metrics") or {}
    primary = next((v for k, v in held_out.items() if "opencv" in k), None)
    if primary:
        assert six == pytest.approx(primary["development_threshold"], abs=0.0), (
            "Experiment 8 selected a different threshold from the same development data"
        )


def test_canonical_digests_agree_across_experiments() -> None:
    open_set = _load("bfw_open_set_test_metrics.json")
    review = _load("ml_review_test_metrics.json")
    pipeline = _load("pipeline_comparison_metrics.json")
    assert open_set["canonical_test_run_digest"] == review["canonical_test_run_digest"]
    assert (open_set["canonical_development_run_digest"]
            == review["canonical_development_run_digest"])
    if pipeline.get("evaluated") == "yes":
        assert pipeline["canonical_test_run_digest"] == open_set["canonical_test_run_digest"]
        assert (pipeline["canonical_development_run_digest"]
                == open_set["canonical_development_run_digest"])


def test_context_digests_are_published() -> None:
    for name in ("bfw_open_set_test_metrics.json", "ml_review_test_metrics.json",
                 "pipeline_comparison_metrics.json"):
        payload = _load(name)
        if name.startswith("pipeline") and payload.get("evaluated") != "yes":
            continue
        for key in ("canonical_development_context_sha256", "canonical_test_context_sha256",
                    "cache_schema_version"):
            assert key in payload, f"{name} missing {key}"


# --- Profile-consistency rates and policy -------------------------------------


def _consistency_entries():
    payload = _load("profile_photo_consistency_metrics.json")
    return payload["pipelines"]


def test_no_duplicate_screening_policy_appears_in_consistency_artefacts() -> None:
    """The two tasks refer in opposite directions; the screening note would
    describe the wrong one."""
    # The screening note may appear only under its own clearly labelled field,
    # never as the generic policy for a consistency result.
    for name in ("profile_photo_consistency.json", "profile_photo_consistency_metrics.json"):
        payload = _load(name)
        assert "policy_note" not in payload, f"{name} carries an unqualified policy_note"
        screening = payload.get("duplicate_screening_policy_note", "")
        if screening:
            assert "gallery screening only" in screening
    for entry in _consistency_entries().values():
        assert "policy_note" not in entry, "a consistency entry carries a generic policy_note"
        assert "below the frozen consistency threshold" in entry[
            "profile_consistency_policy_note"
        ]


def test_conditional_rates_use_scored_outcomes_only() -> None:
    for entry in _consistency_entries().values():
        scored = entry["scored_same_person_photographs"]
        assert scored == (entry["consistent_same_person_photographs"]
                          + entry["inconsistent_same_person_review_candidates"])
        if scored:
            assert entry["same_person_consistency_rate_conditional"] == pytest.approx(
                entry["consistent_same_person_photographs"] / scored
            )
        controls = entry["scored_mismatched_controls"]
        assert controls == (entry["mismatched_controls_correctly_identified"]
                            + entry["mismatched_controls_false_consistent"])
        if controls:
            assert entry["mismatch_detection_rate_conditional"] == pytest.approx(
                entry["mismatched_controls_correctly_identified"] / controls
            )


def test_end_to_end_rates_use_intended_outcomes() -> None:
    for entry in _consistency_entries().values():
        intended = entry["intended_same_person_photographs"]
        assert intended == (entry["scored_same_person_photographs"]
                            + entry["same_person_extraction_failures"]
                            + entry["gallery_reference_unavailable"])
        assert entry["same_person_consistency_rate_end_to_end"] == pytest.approx(
            entry["consistent_same_person_photographs"] / intended
        )
        # Extraction failures must reduce the end-to-end rate.
        if entry["same_person_extraction_failures"]:
            assert (entry["same_person_consistency_rate_end_to_end"]
                    < entry["same_person_consistency_rate_conditional"])


def test_extraction_failures_stay_unresolved() -> None:
    """A failure is neither a match nor a mismatch and must not enter either
    conditional denominator."""
    for entry in _consistency_entries().values():
        assert entry["same_person_extraction_coverage"] == pytest.approx(
            entry["scored_same_person_photographs"]
            / entry["intended_same_person_photographs"]
        )
        assert "resolves nothing" in entry["outcome_policy"]


def test_both_pipelines_appear_in_consistency_json_and_csv() -> None:
    pipeline = _load("pipeline_comparison_metrics.json")
    entries = _consistency_entries()
    if pipeline.get("evaluated") == "yes":
        assert len(entries) == 2
    rows = list(csv.DictReader(open(AGG / "profile_photo_consistency_metrics.csv",
                                    encoding="utf-8")))
    assert {r["pipeline"] for r in rows} == set(entries)
    published = {r["outcome"] for r in rows}
    for field in acp.CONSISTENCY_RATE_FIELDS:
        assert field in published, field


# --- Per-pipeline sex aggregates ----------------------------------------------


def test_dedicated_sex_aggregate_files_exist_with_four_groups() -> None:
    pipeline = _load("pipeline_comparison_metrics.json")
    if pipeline.get("evaluated") != "yes":
        pytest.skip("comparison not evaluated")
    payload = _load("pipeline_sex_aggregated_metrics.json")
    assert len(payload["groups"]) == 4
    assert "pooled over identity outcomes" in payload["aggregation"]
    for entry in payload["groups"].values():
        assert entry["identities"] > 0
        assert entry["bootstrap_replicates"] == acp.BOOTSTRAP_REPLICATES
        assert entry["seed"] == 20260727
    rows = list(csv.DictReader(open(AGG / "pipeline_sex_aggregated_metrics.csv",
                                    encoding="utf-8")))
    assert len(rows) == 4
    assert {r["sex"] for r in rows} == {"female", "male"}


# --- Generated reports --------------------------------------------------------


def test_the_experiment_eight_report_carries_numbers_not_only_prose() -> None:
    pipeline = _load("pipeline_comparison_metrics.json")
    if pipeline.get("evaluated") != "yes":
        pytest.skip("comparison not evaluated")
    text = _text("PRETRAINED_PIPELINE_COMPARISON_REPORT.md")
    for heading in ("## Held-out results", "## Cost", "## Female aggregate",
                    "## Male aggregate", "## Subgroup extremes",
                    "## Profile-photo consistency",
                    "## Canonical primary-pipeline provenance"):
        assert heading in text, heading
    for token in ("Frozen threshold", "CMC rank-1", "Complete pipeline mean",
                  "Detector model size", "identity-cluster bootstrap"):
        assert token in text, token
    assert pipeline["canonical_test_run_digest"] in text


def test_the_research_report_covers_both_pipelines_and_both_directions() -> None:
    pipeline = _load("pipeline_comparison_metrics.json")
    text = _text("RESEARCH_REPORT.md")
    if pipeline.get("evaluated") == "yes":
        female = text[text.index("## 7."):text.index("## 8.")]
        male = text[text.index("## 8."):text.index("## 9.")]
        for section, label in ((female, "female"), (male, "male")):
            assert "opencv" in section and "insightface" in section, (
                f"the {label} section does not compare both pipelines"
            )
    consistency = text[text.index("## 9."):text.index("## 10.")]
    for token in ("Consistency (cond.)", "Consistency (end-to-end)",
                  "Mismatch detection (cond.)", "Mismatch detection (end-to-end)"):
        assert token in consistency, token
    # The two threshold directions must not be collapsed into one statement.
    assert "a *high* similarity" in text and "a *low* similarity" in text


# --- README -------------------------------------------------------------------


def test_the_readme_carries_no_stale_claims() -> None:
    text = _text("README.md")
    for stale in ("not_run_model_files_not_configured",
                  "Experiment 8 has produced no real held-out metrics",
                  "optional pipeline comparison; deliberately unpinned",
                  "holds six figures"):
        assert stale not in text, f"README still contains: {stale}"


def test_the_readme_does_not_recommend_the_unsafe_install() -> None:
    text = _text("README.md")
    assert not re.search(r"pip install -r requirements-comparison\.txt(?!\s*`?:)", text) or (
        "Do **not** run a plain" in text
    )
    assert "scripts/install_comparison_environment.sh" in text


def test_the_readme_figure_inventory_matches_the_generated_files() -> None:
    if not FIG.is_dir():
        pytest.skip("figures not generated")
    text = _text("README.md")
    stems = sorted(p.stem for p in FIG.glob("*.png"))
    assert f"holds {len(stems)} figures" in text
    for stem in stems:
        assert stem in text, f"README omits figure {stem}"


def test_the_readme_names_the_authoritative_sources() -> None:
    text = _text("README.md")
    assert "pipeline_comparison_metrics.json" in text
    assert "pretrained_pipeline_comparison.csv" in text


# --- Installation -------------------------------------------------------------


def test_every_optional_dependency_is_exactly_pinned() -> None:
    text = _text("requirements-comparison-deps.txt")
    pins = [line.strip() for line in text.splitlines()
            if line.strip() and not line.startswith("#")]
    assert pins, "no pins recorded"
    for pin in pins:
        assert "==" in pin, f"{pin} is not exactly pinned"
    assert any(p.startswith("opencv-python-headless==4.13.0.92") for p in pins)
    assert not any(p.startswith("opencv-python==") for p in pins)


def test_the_install_script_enforces_the_dependency_contract() -> None:
    script = _text("scripts/install_comparison_environment.sh")
    assert "VIRTUAL_ENV" in script and "refusing to install globally" in script
    assert "--no-deps insightface==1.0.1" in script
    assert "requirements-comparison-deps.txt" in script
    assert "onnxruntime==1.28.0" in _text("requirements-comparison-deps.txt")
    for check in ("opencv-python is installed and will shadow", "cv2.__version__",
                  "cv2.__file__", "insightface.model_zoo",
                  "insightface.utils import face_align", "raise SystemExit(1)"):
        assert check in script, check


def test_the_running_environment_satisfies_the_contract() -> None:
    import cv2
    import importlib.metadata as meta

    assert str(cv2.__version__).startswith("4.13.0")
    with pytest.raises(meta.PackageNotFoundError):
        meta.version("opencv-python")
    assert meta.version("opencv-python-headless") == "4.13.0.92"


def test_experiment_eight_provenance_reports_optional_dependencies() -> None:
    for name in ("pipeline_comparison_metrics.json", "pipeline_comparison_protocol.json",
                 "pipeline_comparison_confidence_intervals.json"):
        payload = _load(name)
        versions = payload.get("dependency_versions") or {}
        for package in ("insightface", "onnxruntime", "onnx", "scipy", "scikit-image",
                        "tqdm", "requests"):
            assert package in versions, f"{name} omits {package}"
        distribution = payload.get("opencv_distribution") or {}
        assert distribution.get("opencv_distribution") == "opencv-python-headless"
        assert distribution.get("conflicting_opencv_python_present") is False
        # A local module path must never reach a published artefact.
        assert "/Users/" not in json.dumps(payload)
