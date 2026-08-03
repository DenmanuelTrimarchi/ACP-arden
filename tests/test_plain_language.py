"""Acceptance tests for the plain-language terminal layer.

The programme is read by people who do not already know what FPIR, TPIR,
enrolment or a cluster bootstrap are. These tests check what the terminal
actually says: that every headline figure carries its denominator, that no
wording claims a fact the artefact does not establish, and that a missing
optional experiment produces an instruction rather than a crash.

They deliberately do not check any scientific value, which the existing
methodology tests already cover.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict

import pytest

import ACP_arden as acp

ROOT = Path(acp.__file__).parent
AGG = ROOT / "results" / "aggregate"


def _summary(renderer, *args: Any) -> str:
    return renderer(*args) if args else renderer()


# --- Wording that must never appear -------------------------------------------

# The programme produces review signals. It establishes none of these facts,
# so the vocabulary is banned outright rather than discouraged.
FORBIDDEN_CLAIMS = (
    "duplicate images found",
    "fake profiles detected",
    "stolen photographs identified",
    "fraudulent users found",
    "matching profiles proved",
    "fraud detected",
    "confirmed fraud",
)

PLAIN_RENDERERS = (
    "render_baseline_plain_summary",
    "render_open_set_plain_summary",
    "render_ml_review_plain_summary",
    "render_pipeline_plain_summary",
    "render_overall_conclusion",
)


def _flat(text: str) -> str:
    """Collapse wrapping, so a phrase split across two terminal lines still
    matches. Presentation wraps text; the wording is what is under test."""
    return re.sub(r"\s+", " ", text)


def _all_plain_text() -> str:
    parts = [acp.MENU_TEXT, acp.PROGRAMME_INTRODUCTION, acp.render_glossary(),
             acp.render_model_overview(), acp.render_dataset_overview()]
    for name in PLAIN_RENDERERS:
        parts.append(getattr(acp, name)(AGG))
    for preview in acp.EXPERIMENT_PREVIEWS.values():
        parts.append(preview)
    return "\n".join(parts).lower()


@pytest.mark.parametrize("claim", FORBIDDEN_CLAIMS)
def test_no_output_claims_a_confirmed_finding(claim: str) -> None:
    assert claim not in _all_plain_text(), f"the terminal output claims: {claim}"


def test_the_referral_disclaimer_accompanies_every_result_summary() -> None:
    """A count of referrals without this sentence invites the wrong reading."""
    for name in ("render_baseline_plain_summary", "render_open_set_plain_summary",
                 "render_ml_review_plain_summary", "render_pipeline_plain_summary"):
        text = getattr(acp, name)(AGG)
        assert "model-generated review signal" in text, name
        assert "not proof" in text, name


def test_the_disclaimer_names_what_it_does_not_prove() -> None:
    disclaimer = acp.REFERRAL_DISCLAIMER.lower()
    for claim in ("belong to the same person", "stolen", "fraud occurred"):
        assert claim in disclaimer, claim


# --- Counts, denominators and percentages -------------------------------------


def test_format_count_and_percentage_always_shows_the_denominator() -> None:
    text = acp.format_count_and_percentage(942, 1000, noun="intended photographs")
    assert "942" in text and "1,000" in text and "94.20%" in text


def test_format_count_and_percentage_handles_absent_values() -> None:
    assert acp.format_count_and_percentage(None, 100) == "not available"
    assert acp.format_count_and_percentage(float("nan"), 100) == "not available"
    # A count with no denominator is shown as a count, never as a bare share.
    assert "%" not in acp.format_count_and_percentage(5, 0)
    assert "%" not in acp.format_count_and_percentage(5, None)


def test_format_count_and_percentage_can_carry_the_technical_name() -> None:
    text = acp.format_count_and_percentage(15, 2859, technical="FPIR")
    assert text.endswith("[FPIR]")
    assert "15 of 2,859" in text


@pytest.mark.parametrize(
    ("renderer", "artefact"),
    [("render_baseline_plain_summary", "lfw_final_metrics.json"),
     ("render_open_set_plain_summary", "bfw_open_set_test_metrics.json")],
)
def test_every_percentage_is_accompanied_by_a_count(renderer: str, artefact: str) -> None:
    """A percentage on its own cannot be checked by a reader.

    Percentages inside a comparison table are exempt: the table's own row
    labels and the counts printed beneath it supply the denominators."""
    if not (AGG / artefact).is_file():
        pytest.skip(f"{artefact} not present in this checkout")
    text = getattr(acp, renderer)(AGG)
    body, _, _table = text.partition("  ---")
    for line in body.splitlines():
        if "%" not in line or line.strip().startswith("|"):
            continue
        if "approximately" in line or "at least" in line.lower():
            continue
        assert re.search(r"\d[\d,]* of \d[\d,]*", line), (
            f"{renderer} shows a percentage with no denominator: {line!r}"
        )


def test_the_denominator_distinction_is_explained_where_both_appear() -> None:
    for name in ("render_baseline_plain_summary", "render_open_set_plain_summary",
                 "render_ml_review_plain_summary", "render_pipeline_plain_summary"):
        text = getattr(acp, name)(AGG)
        assert "Conditional results use only" in text, name
        assert "End-to-end results use every" in text, name


# --- Menu -----------------------------------------------------------------------


def test_the_menu_is_grouped_by_purpose() -> None:
    for heading in ("SETUP AND VALIDATION", "ORIGINAL FIVE EXPERIMENTS",
                    "BFW EXTENSION EXPERIMENTS"):
        assert heading in acp.MENU_TEXT, heading


def test_option_thirteen_is_described_as_experiments_seven_and_eight() -> None:
    """It runs Experiments 7 and 8, not every extension experiment."""
    line = next(l for l in acp.MENU_TEXT.splitlines() if l.strip().startswith("13."))
    assert "Experiments 7 and 8" in line
    assert "both extension experiments" not in acp.MENU_TEXT.lower()


def test_every_menu_option_remains_reachable() -> None:
    """Regrouping must not drop an option."""
    for option in ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9.",
                   "10.", "11.", "12.", "13."):
        assert option in acp.MENU_TEXT, option


def test_the_menu_names_the_pipelines_readably() -> None:
    assert "YuNet + SFace" in acp.MENU_TEXT
    assert "SCRFD + ArcFace" in acp.MENU_TEXT


# --- Introduction, previews and overviews ---------------------------------------


def test_the_introduction_states_the_limits_of_the_programme() -> None:
    text = _flat(acp.PROGRAMME_INTRODUCTION)
    assert "does not automatically identify fraud" in text
    assert "No face-recognition model is trained or fine-tuned" in text
    assert "public academic benchmark datasets" in text


@pytest.mark.parametrize(
    "key", ["full", "open-set", "ml-review", "pipeline-compare", "extensions", "review"]
)
def test_every_long_running_option_has_a_preview(key: str) -> None:
    preview = acp.render_experiment_preview(key)
    assert preview, key
    assert "Purpose:" in preview or "demonstration" in preview.lower()


def test_the_previews_are_wired_to_the_menu_options() -> None:
    for choice, key in acp.MENU_PREVIEW_KEYS.items():
        assert key in acp.EXPERIMENT_PREVIEWS, f"option {choice} has no preview"


def test_the_experiment_seven_preview_excludes_demographic_predictors() -> None:
    preview = acp.render_experiment_preview("ml-review")
    assert "does not use" in preview
    assert "sex, ethnicity" in preview


def test_the_experiment_eight_preview_states_the_comparison_is_whole_pipeline() -> None:
    preview = acp.render_experiment_preview("pipeline-compare")
    assert "cannot be attributed only" in preview
    assert "detection, alignment and preprocessing also" in preview


def test_the_model_overview_states_who_trained_each_model() -> None:
    text = acp.render_model_overview()
    for model in ("YuNet", "SFace", "SCRFD", "ArcFace", "Logistic regression"):
        assert model in text, model
    assert text.count("not trained by this project") == 4
    assert "Fitted by this project using BFW development identities only" in text


def test_the_dataset_overview_does_not_call_benchmark_people_users() -> None:
    text = acp.render_dataset_overview()
    for dataset in ("LFW", "CPLFW", "BFW"):
        assert dataset in text, dataset
    assert "not users of a dating application" in text


# --- Glossary ------------------------------------------------------------------


@pytest.mark.parametrize(
    "term",
    ["Enrolment", "Gallery", "Mated probe", "Non-mated probe", "False review",
     "TPIR@1", "FPIR", "Conditional rate", "End-to-end rate", "Confidence interval"],
)
def test_the_glossary_defines_each_technical_term(term: str) -> None:
    glossary = acp.render_glossary()
    assert f"{term}:" in glossary, term
    definition = glossary.split(f"{term}:", 1)[1].split("\n\n", 1)[0].strip()
    assert len(definition) > 20, f"{term} has no usable definition"


# --- Experiment-specific content -------------------------------------------------


def test_the_experiment_seven_summary_states_the_negative_finding() -> None:
    text = _flat(acp.render_ml_review_plain_summary(AGG))
    assert "main hypothesis was not achieved" in text
    assert "valid negative research finding" in text
    assert "does not indicate that the programme failed to run" in text


def test_the_experiment_seven_criteria_are_readable() -> None:
    text = acp.render_ml_review_plain_summary(AGG)
    assert "Fewer unnecessary reviews than the similarity method" in text
    assert "At least 90% gallery enrolment coverage" in text


def test_the_experiment_eight_summary_carries_real_comparison_values() -> None:
    payload = json.loads((AGG / "pipeline_comparison_metrics.json").read_text())
    if payload.get("evaluated") != "yes":
        pytest.skip("comparison not evaluated in this checkout")
    text = acp.render_pipeline_plain_summary(AGG)
    for row in ("Known duplicate detection", "End-to-end detection",
                "New profiles wrongly sent for review", "False reviews per 1,000",
                "Mean complete processing time", "Model storage"):
        assert row in text, row
    # Real values, not placeholders.
    assert "[value]" not in text and "not available" not in text
    assert "%" in text and "ms" in text and "MB" in text


def test_the_experiment_eight_summary_leads_with_readable_pipeline_names() -> None:
    payload = json.loads((AGG / "pipeline_comparison_metrics.json").read_text())
    if payload.get("evaluated") != "yes":
        pytest.skip("comparison not evaluated")
    text = acp.render_pipeline_plain_summary(AGG)
    heading = text.split("Technical pipeline identifiers:")[0]
    assert "YuNet + SFace" in heading and "SCRFD + ArcFace" in heading
    # The internal identifier belongs under the technical heading only.
    assert "opencv-sface-2021dec-yunet-2023mar" not in heading
    assert "opencv-sface-2021dec-yunet-2023mar" in text


def test_the_review_interface_warns_that_it_uses_the_baseline_method() -> None:
    preview = acp.render_experiment_preview("review")
    assert "not confirmed duplicate profiles" in preview
    assert "high false-review rate" in preview
    assert "not a production moderation decision" in preview


def test_the_overall_conclusion_covers_all_five_findings() -> None:
    text = _flat(acp.render_overall_conclusion(AGG))
    for number in ("1.", "2.", "3.", "4.", "5."):
        assert number in text, number
    assert "does not prove identity, photograph ownership, fraud or profile" in text


# --- Status wording --------------------------------------------------------------


def test_internal_status_values_are_translated_for_display_only() -> None:
    assert acp.plain_status("evaluated_non_commercial_academic_research") == (
        "Evaluation completed for non-commercial academic research."
    )
    assert acp.plain_status("ml_review_tested") == "Classifier evaluation completed."
    assert acp.plain_status("open_set_tested") == (
        "Held-out open-set evaluation completed."
    )
    # An unknown status is shown rather than hidden behind friendly wording.
    assert acp.plain_status("some_new_state") == "some_new_state"
    assert acp.plain_status(None) == "Status not recorded."


def test_the_stored_status_values_are_unchanged() -> None:
    """Display wording must never be written back into an artefact."""
    for name, expected in (
        ("pipeline_comparison_metrics.json", {
            "evaluated_non_commercial_academic_research",
            "not_run_licensing_unresolved", "not_run_models_unavailable"}),
        ("ml_review_test_metrics.json", {"ml_review_tested"}),
        ("bfw_open_set_test_metrics.json", {"open_set_tested"}),
    ):
        path = AGG / name
        if not path.is_file():
            continue
        status = json.loads(path.read_text()).get("status")
        assert status in expected, f"{name} stores an unexpected status: {status}"


# --- Missing artefacts ------------------------------------------------------------


@pytest.mark.parametrize(
    ("renderer", "expected"),
    [("render_baseline_plain_summary", "option 3"),
     ("render_open_set_plain_summary", "option 8"),
     ("render_ml_review_plain_summary", "option 10"),
     ("render_pipeline_plain_summary", "option 12")],
)
def test_a_missing_artefact_gives_an_instruction_not_a_crash(
    tmp_path: Path, renderer: str, expected: str
) -> None:
    text = getattr(acp, renderer)(tmp_path)
    assert "not available yet" in text
    assert expected in text


def test_the_overall_conclusion_survives_a_missing_comparison(tmp_path: Path) -> None:
    text = acp.render_overall_conclusion(tmp_path)
    assert "has not been run" in text
    assert "5." in text


# --- Privacy ----------------------------------------------------------------------


def test_no_plain_summary_prints_a_path_identity_or_record_level_score() -> None:
    text = _all_plain_text()
    for forbidden in ("/users/", "/home/", "\\users\\", ".jpg", ".png",
                      "identity_hash", "sample_id", "image_path"):
        assert forbidden not in text, f"the terminal output contains {forbidden}"
    # "embedding" may be named, but only to say it is not used as a predictor
    # or not published; never as a printed value.
    for sentence in re.split(r"(?<=[.;])\s", text):
        if "embedding" in sentence:
            assert any(
                phrase in sentence
                for phrase in ("does not use", "not published", "never", "no ",
                               "converts a detected face", "representation")
            ), f"embeddings mentioned without an exclusion: {sentence[:120]!r}"
    # Opaque identifiers are 32 or 64 hexadecimal characters; neither belongs
    # in a plain-language summary.
    assert not re.search(r"\b[0-9a-f]{32,}\b", text)


def test_the_review_status_wording_covers_every_stored_status() -> None:
    for status in acp.REVIEW_STATUSES:
        assert status in acp.REVIEW_STATUS_WORDING, status
    # The stored vocabulary itself is unchanged.
    assert acp.REVIEW_STATUSES == [
        "open", "confirmed_duplicate", "false_match", "dismissed"
    ]


# --- SCRFD warning handling --------------------------------------------------------


def test_the_scrfd_explanation_sets_expectations_without_hiding_failure() -> None:
    text = acp.SCRFD_WARNING_EXPLANATION
    assert "dynamic output dimensions" in text
    assert "do not indicate that the evaluation has failed" in text
    assert "stop if the returned detector outputs are invalid" in text


def test_scrfd_validation_rejects_malformed_detector_output() -> None:
    """The warning is only tolerable because the outputs are actually checked."""
    import numpy as np

    class _Detector:
        def __init__(self, outputs: Any) -> None:
            self._outputs = outputs

        def detect(self, *_args: Any, **_kwargs: Any) -> Any:
            if isinstance(self._outputs, Exception):
                raise self._outputs
            return self._outputs

    good = (np.array([[10.0, 10.0, 40.0, 40.0, 0.9]]),
            np.zeros((1, 5, 2), dtype=np.float64))
    acp.validate_scrfd_outputs(_Detector(good))          # must not raise
    acp.validate_scrfd_outputs(_Detector((None, None)))  # no detection is fine

    bad_cases = [
        ("wrong output count", (np.array([]),)),
        ("negative box", (np.array([[40.0, 40.0, 10.0, 10.0, 0.9]]), None)),
        ("non-finite box", (np.array([[np.nan, 0.0, 4.0, 4.0, 0.9]]), None)),
        ("narrow box", (np.array([[1.0, 2.0]]), None)),
        ("wrong landmark shape",
         (good[0], np.zeros((1, 3, 2), dtype=np.float64))),
        ("non-finite landmark",
         (good[0], np.full((1, 5, 2), np.inf, dtype=np.float64))),
        ("detector raises", RuntimeError("session failed")),
    ]
    for label, outputs in bad_cases:
        with pytest.raises(acp.PipelineUnavailableError):
            acp.validate_scrfd_outputs(_Detector(outputs))


# --- The formal reports are untouched -----------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["FINAL_EVALUATION_REPORT.md", "OPEN_SET_EVALUATION_REPORT.md",
     "ML_REVIEW_EVALUATION_REPORT.md", "PRETRAINED_PIPELINE_COMPARISON_REPORT.md",
     "RESEARCH_REPORT.md"],
)
def test_the_academic_reports_keep_their_technical_wording(name: str) -> None:
    """The plain layer is additional. The formal reports keep formal metrics."""
    path = AGG / name
    if not path.is_file():
        pytest.skip(f"{name} not generated in this checkout")
    text = path.read_text(encoding="utf-8")
    assert any(term in text for term in ("FPIR", "TPIR", "threshold", "coverage")), name
    # Each report keeps its methodological caveat, though not all use the same
    # heading: the comparison report states it as a scope limitation instead.
    assert any(
        phrase in text.lower()
        for phrase in ("limitation", "cannot be attributed", "not proof",
                       "does not prove")
    ), name
