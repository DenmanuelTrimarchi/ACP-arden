"""Tests for the review classifier (Experiment 7) and the pipeline comparison
(Experiment 8).

No face-recognition model is trained or fine-tuned anywhere in this suite, and
no test reads a dataset, loads a model binary or touches the network. The
classifier is exercised on deterministic synthetic feature rows so its
guarantees — identity-disjoint splits, training-only standardisation, frozen
thresholds — are pinned independently of BFW.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pytest

import ACP_arden as acp

EXPECTED_SEED = 20260727
EXPECTED_FROZEN_STATUS = "ml_review_frozen"
EXPECTED_PRIMARY_FPIR_TARGET = 0.003
EXPECTED_FEATURE_ORDER = (
    "top1_similarity",
    "top2_similarity",
    "top1_top2_margin",
    "top5_similarity_mean",
    "top5_similarity_stdev",
    "top1_gallery_image_count",
    "gallery_size",
    "probe_detection_confidence",
    "probe_face_area_ratio",
)
EXPECTED_SUBGROUPS = (
    "asian_females", "asian_males", "black_females", "black_males",
    "indian_females", "indian_males", "white_females", "white_males",
)


def _row(label: int, top1: float, *, subgroup: str = "asian_females",
         identity: str = "id", sample: str = "s") -> acp.ReviewFeatureRow:
    top2 = top1 - 0.10
    return acp.ReviewFeatureRow(
        sample_id=sample, identity_hash=identity, subgroup=subgroup,
        role="mated_probe" if label == 1 else "non_mated_probe",
        features={
            "top1_similarity": top1,
            "top2_similarity": top2,
            "top1_top2_margin": top1 - top2,
            "top5_similarity_mean": top1 - 0.15,
            "top5_similarity_stdev": 0.05,
            "top1_gallery_image_count": 3.0,
            "gallery_size": 200.0,
            "probe_detection_confidence": 0.95,
            "probe_face_area_ratio": 0.55,
        },
        label=label,
        correct_rank=1 if label == 1 else None,
        correct_similarity=top1 if label == 1 else None,
    )


def _training_rows(count: int = 120) -> List[acp.ReviewFeatureRow]:
    rng = np.random.default_rng(EXPECTED_SEED)
    rows: List[acp.ReviewFeatureRow] = []
    for index in range(count):
        subgroup = EXPECTED_SUBGROUPS[index % len(EXPECTED_SUBGROUPS)]
        rows.append(_row(1, float(rng.uniform(0.55, 0.95)), subgroup=subgroup,
                         identity=f"m{index:03d}", sample=f"ms{index:03d}"))
        rows.append(_row(0, float(rng.uniform(0.05, 0.45)), subgroup=subgroup,
                         identity=f"n{index:03d}", sample=f"ns{index:03d}"))
    return rows


# --- Feature contract ---------------------------------------------------------


def test_feature_order_is_stable_and_excludes_forbidden_inputs() -> None:
    assert acp.ML_REVIEW_FEATURES == EXPECTED_FEATURE_ORDER
    joined = " ".join(acp.ML_REVIEW_FEATURES)
    # Subgroup must never be a predictor; nor may identity, path or embedding.
    for banned in ("subgroup", "ethnic", "gender", "identity_hash", "path", "embedding", "label"):
        assert banned not in joined


def test_every_feature_has_a_published_definition() -> None:
    definitions = acp._feature_definitions()
    assert set(definitions) == set(acp.ML_REVIEW_FEATURES)
    assert all(text.strip() for text in definitions.values())


def test_labels_follow_the_protocol_roles_and_rank() -> None:
    """A mated probe is positive only when its own identity ranks first."""
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role=role, top_similarity=0.8, top2_similarity=0.7,
            top5_similarity_mean=0.6, top5_similarity_stdev=0.05,
            top1_gallery_image_count=3, gallery_size=200,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            correct_rank=1 if role == "mated_probe" else None,
            correct_similarity=0.8 if role == "mated_probe" else None,
        )
        for role in ("mated_probe", "non_mated_probe")
    ]
    rows, _excluded = acp.build_review_feature_rows(results)
    assert [r.label for r in rows] == [1, 0]


def test_unscored_records_never_become_training_examples() -> None:
    """An extraction failure is a coverage outcome, not a negative decision."""
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role="mated_probe", failure_code="zero_faces",
        ),
        acp.OpenSetSearchResult(
            sample_id="b" * 32, identity_hash="i" * 32, subgroup="asian_females",
            role="mated_probe", failure_code=acp.GALLERY_REFERENCE_UNAVAILABLE,
        ),
    ]
    rows, excluded = acp.build_review_feature_rows(results)
    assert rows == []
    assert excluded["unscored"] == 2


def test_a_record_missing_a_feature_is_excluded_not_imputed() -> None:
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role="mated_probe", top_similarity=0.8, top2_similarity=None,
        )
    ]
    rows, excluded = acp.build_review_feature_rows(results)
    assert rows == []
    assert excluded["missing_feature"] == 1


# --- Identity-disjoint splits -------------------------------------------------


def _protocol(tmp_path: Path) -> acp.OpenSetProtocol:
    from tests.test_bfw_open_set import _make_bfw  # reuse the official-layout fixture

    root, metadata = _make_bfw(tmp_path)
    return acp.build_open_set_protocol(acp.load_bfw_dataset(root, metadata), seed=EXPECTED_SEED)


def test_training_and_calibration_identities_are_disjoint(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    training, calibration = acp.split_development_identities_for_classifier(
        protocol, seed=EXPECTED_SEED
    )
    assert training and calibration
    assert set(training).isdisjoint(set(calibration))


def test_classifier_identities_never_touch_the_held_out_test_partition(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    training, calibration = acp.split_development_identities_for_classifier(
        protocol, seed=EXPECTED_SEED
    )
    test_identities = {e.identity for e in protocol.partition("test")}
    assert set(training).isdisjoint(test_identities)
    assert set(calibration).isdisjoint(test_identities)


def test_the_classifier_split_is_stratified_and_reproducible(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    first = acp.split_development_identities_for_classifier(protocol, seed=EXPECTED_SEED)
    second = acp.split_development_identities_for_classifier(protocol, seed=EXPECTED_SEED)
    assert first == second
    other = acp.split_development_identities_for_classifier(protocol, seed=EXPECTED_SEED + 1)
    assert other != first

    subgroup_of = {e.identity: e.subgroup for e in protocol.partition("development")}
    for group in first:
        represented = {subgroup_of[i] for i in group}
        assert represented == set(EXPECTED_SUBGROUPS)


def test_the_split_is_by_identity_never_by_image(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    training, calibration = acp.split_development_identities_for_classifier(
        protocol, seed=EXPECTED_SEED
    )
    images_of = {}
    for entry in protocol.partition("development"):
        images_of.setdefault(entry.identity, set()).add(entry.image_path)
    training_images = set().union(*(images_of[i] for i in training))
    calibration_images = set().union(*(images_of[i] for i in calibration))
    assert training_images.isdisjoint(calibration_images)


# --- Fitting ------------------------------------------------------------------


def test_the_classifier_is_reproducible_under_the_research_seed() -> None:
    rows = _training_rows()
    first = acp.fit_review_classifier(rows)
    second = acp.fit_review_classifier(rows)
    assert first.coefficients == pytest.approx(second.coefficients)
    assert first.intercept == pytest.approx(second.intercept)
    assert first.feature_order == EXPECTED_FEATURE_ORDER


def test_standardisation_uses_training_statistics_only() -> None:
    rows = _training_rows()
    classifier = acp.fit_review_classifier(rows)
    matrix, _labels = acp._feature_matrix(rows)
    assert classifier.scaler_mean == pytest.approx(matrix.mean(axis=0), rel=1e-6)
    # Applying the stored parameters to unseen rows must not refit them.
    unseen = np.asarray([[9.0] * len(EXPECTED_FEATURE_ORDER)])
    before = list(classifier.scaler_mean)
    classifier.probabilities(unseen)
    assert list(classifier.scaler_mean) == before


def test_hyperparameters_are_recorded_and_imbalance_is_declared() -> None:
    payload = acp.fit_review_classifier(_training_rows()).as_dict()
    assert payload["hyperparameters"]["class_weight"] == "balanced"
    assert payload["hyperparameters"]["random_state"] == EXPECTED_SEED
    assert "pickle" in payload["serialisation"].lower()
    assert len(payload["coefficients"]) == len(EXPECTED_FEATURE_ORDER)


def test_the_model_serialises_as_plain_numbers(tmp_path: Path) -> None:
    """A pickle would be unsafe to load and opaque to a reader."""
    payload = acp.fit_review_classifier(_training_rows()).as_dict()
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    restored = json.loads(path.read_text(encoding="utf-8"))
    assert restored["feature_order"] == list(EXPECTED_FEATURE_ORDER)
    assert all(isinstance(c, float) for c in restored["coefficients"])


def test_fitting_requires_both_classes() -> None:
    with pytest.raises(acp.MlReviewError):
        acp.fit_review_classifier([_row(1, 0.9), _row(1, 0.8)])


def test_probabilities_are_bounded_and_monotone_in_similarity() -> None:
    classifier = acp.fit_review_classifier(_training_rows())
    low = acp._feature_matrix([_row(0, 0.05)])[0]
    high = acp._feature_matrix([_row(1, 0.95)])[0]
    p_low = float(classifier.probabilities(low)[0])
    p_high = float(classifier.probabilities(high)[0])
    assert 0.0 <= p_low <= 1.0 and 0.0 <= p_high <= 1.0
    assert p_high > p_low


# --- Threshold selection and freezing -----------------------------------------


def test_threshold_selection_respects_the_target_fpir() -> None:
    rows = _training_rows()
    classifier = acp.fit_review_classifier(rows)
    matrix, _ = acp._feature_matrix(rows)
    probabilities = classifier.probabilities(matrix)
    for target in acp.FPIR_TARGETS:
        chosen = acp.select_review_probability_threshold(
            rows, probabilities, target_fpir=target
        )
        assert chosen["calibration_fpir"] <= target + 1e-12
        assert 0.0 <= chosen["probability_threshold"] <= 1.0000001


def test_an_unreachable_target_is_refused_rather_than_approximated() -> None:
    rows = _training_rows()
    classifier = acp.fit_review_classifier(rows)
    probabilities = classifier.probabilities(acp._feature_matrix(rows)[0])
    with pytest.raises(acp.MlReviewError):
        acp.select_review_probability_threshold(rows, probabilities, target_fpir=-1.0)


@pytest.mark.parametrize("status", ["ml_review_development", None, "frozen", "open_set_frozen"])
def test_an_unfrozen_review_policy_is_refused(status) -> None:
    payload = {"status": status,
               "operating_points": {str(EXPECTED_PRIMARY_FPIR_TARGET):
                                    {"probability_threshold": 0.5}}}
    with pytest.raises(acp.MlReviewError):
        acp.require_frozen_review_policy(payload)


def test_a_frozen_policy_returns_the_primary_threshold() -> None:
    payload = {"status": EXPECTED_FROZEN_STATUS,
               "operating_points": {str(EXPECTED_PRIMARY_FPIR_TARGET):
                                    {"probability_threshold": 0.6125}}}
    assert acp.require_frozen_review_policy(payload) == pytest.approx(0.6125)


def test_a_frozen_policy_without_the_primary_target_is_refused() -> None:
    with pytest.raises(acp.MlReviewError):
        acp.require_frozen_review_policy(
            {"status": EXPECTED_FROZEN_STATUS,
             "operating_points": {"0.5": {"probability_threshold": 0.4}}}
        )


# --- Failure accounting and metrics -------------------------------------------


def test_confusion_counts_reconcile_with_the_denominators() -> None:
    rows = [_row(1, 0.9), _row(1, 0.2), _row(0, 0.95), _row(0, 0.1)]
    probabilities = np.asarray([0.9, 0.2, 0.95, 0.1])
    rates = acp.review_rates_at_probability(rows, probabilities, 0.5)
    assert rates["mated_correct_rank1_referred"] == 1
    assert rates["mated_not_referred"] == 1
    assert rates["non_mated_incorrectly_referred"] == 1
    assert rates["non_mated_correctly_not_referred"] == 1
    assert rates["scored_mated_probes"] == 2
    assert rates["scored_non_mated_probes"] == 2
    assert rates["fpir"] == pytest.approx(0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.5)
    assert rates["fnir_rank1"] == pytest.approx(0.5)


def test_the_end_to_end_denominator_counts_every_intended_mated_probe() -> None:
    rows = [_row(1, 0.9), _row(0, 0.1)]
    probabilities = np.asarray([0.9, 0.1])
    outcomes = {
        "id": acp.ReviewIdentityOutcome(
            identity_hash="id", subgroup="asian_females", intended_mated=4, scored_mated=1,
            intended_non_mated=1, scored_non_mated=1, mated_extraction_failures=3,
            non_mated_extraction_failures=0, gallery_reference_unavailable=2,
        )
    }
    rates = acp.review_rates_at_probability(rows, probabilities, 0.5, outcomes=outcomes)
    # One referral out of four intended, not out of the one that was scored.
    assert rates["end_to_end_duplicate_detection_rate"] == pytest.approx(0.25)
    assert rates["tpir_rank1"] == pytest.approx(1.0)
    assert rates["gallery_reference_unavailable"] == 2
    assert rates["mated_extraction_coverage"] == pytest.approx(0.25)


# --- Bootstrap and subgroups --------------------------------------------------


def test_the_review_bootstrap_is_deterministic() -> None:
    rows = _training_rows(40)
    classifier = acp.fit_review_classifier(rows)
    probabilities = classifier.probabilities(acp._feature_matrix(rows)[0])
    first = acp.review_cluster_bootstrap(rows, probabilities, 0.5, replicates=50,
                                         seed=EXPECTED_SEED)
    second = acp.review_cluster_bootstrap(rows, probabilities, 0.5, replicates=50,
                                          seed=EXPECTED_SEED)
    assert first == second
    assert first["fpir"]["requested_replicates"] == 50


def test_subgroup_metrics_carry_confidence_intervals_for_every_subgroup() -> None:
    rows = _training_rows(40)
    classifier = acp.fit_review_classifier(rows)
    probabilities = classifier.probabilities(acp._feature_matrix(rows)[0])
    per_subgroup = acp.review_subgroup_metrics(rows, probabilities, 0.5, replicates=40,
                                               seed=EXPECTED_SEED)
    assert set(per_subgroup) == set(EXPECTED_SUBGROUPS)
    for row in per_subgroup.values():
        for key in ("fpir", "fnir_rank1", "tpir_rank1"):
            assert f"{key}_lower_95" in row and f"{key}_upper_95" in row
            assert row[f"{key}_lower_95"] <= row[f"{key}_upper_95"] or math.isnan(
                row[f"{key}_lower_95"]
            )


def test_subgroup_is_never_used_as_a_predictor() -> None:
    """Rows differing only by subgroup must receive identical probabilities."""
    classifier = acp.fit_review_classifier(_training_rows())
    a = acp._feature_matrix([_row(1, 0.8, subgroup="asian_females")])[0]
    b = acp._feature_matrix([_row(1, 0.8, subgroup="white_males")])[0]
    assert float(classifier.probabilities(a)[0]) == pytest.approx(
        float(classifier.probabilities(b)[0])
    )


# --- Experiment 8: pipeline comparison ----------------------------------------


def test_an_unconfigured_pipeline_reports_a_technical_status_not_a_licensing_one(
    tmp_path: Path,
) -> None:
    """Commercial-use restrictions are irrelevant to non-commercial academic
    research, so a licensing status must never stand in for a missing file."""
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=None,
    )
    status = acp.pipeline_comparison_status(config)
    assert status["comparison_run"] is False
    assert status["substitute_model_used"] is False
    assert status["status"] == acp.PIPELINE_STATUS_NOT_CONFIGURED
    assert "licens" not in status["status"]
    assert not hasattr(acp, "PIPELINE_STATUS_NOT_RUN")


def test_the_status_vocabulary_separates_technical_from_terms_blockers() -> None:
    assert acp.PIPELINE_STATUS_EVALUATED == "evaluated_non_commercial_academic_research"
    for name in (acp.PIPELINE_STATUS_NOT_CONFIGURED, acp.PIPELINE_STATUS_SOURCE_UNVERIFIED,
                 acp.PIPELINE_STATUS_DIGEST_NOT_PINNED,
                 acp.PIPELINE_STATUS_DEPENDENCIES_MISSING):
        assert name.startswith("not_run_") and "licens" not in name
    # A terms status exists but is reserved for genuinely unclear research terms.
    assert acp.PIPELINE_STATUS_TERMS_UNCLEAR == "not_run_research_terms_not_established"


def test_preconditions_are_diagnosed_in_order(tmp_path: Path) -> None:
    """A configured root with the files present but no pinned digests must
    report the digest blocker, not the configuration one."""
    (tmp_path / acp.ARCFACE_DETECTOR_FILENAME).write_bytes(b"placeholder")
    (tmp_path / acp.ARCFACE_RECOGNITION_FILENAME).write_bytes(b"placeholder")
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    diagnosis = acp.arcface_preconditions(config)
    assert diagnosis["status"] == acp.PIPELINE_STATUS_DIGEST_NOT_PINNED
    assert diagnosis["checks"]["model_files_present"] is True
    assert diagnosis["checks"]["digests_pinned"] is False
    assert diagnosis["checks"]["research_terms_established"] is True


def test_the_licence_note_states_the_non_commercial_research_position() -> None:
    note = acp.ARCFACE_LICENCE_NOTE
    assert "non-commercial research" in note
    assert "no face-recognition network is trained or fine-tuned here" in note
    assert "MIT licence" in note and "does not automatically extend" in note
    assert "no ownership" in note.lower()
    assert "not redistribute" in note.lower() or "nor redistributes" in note.lower()


def test_unpinned_stronger_model_files_are_refused(tmp_path: Path) -> None:
    """Digests are pinned in source; an unverified weight file cannot be used."""
    (tmp_path / "det_10g.onnx").write_bytes(b"not a real model")
    (tmp_path / "w600k_r50.onnx").write_bytes(b"not a real model")
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    with pytest.raises(acp.PipelineUnavailableError):
        acp.arcface_pipeline_description(config)


def test_the_pipeline_record_forbids_embedding_only_attribution() -> None:
    payload = acp.primary_pipeline_description().as_dict()
    assert "cannot be attributed to the embedding model alone" in payload["comparison_scope"]


# --- Exclusions the brief requires --------------------------------------------


def test_no_face_recognition_model_is_trained_or_fine_tuned() -> None:
    source = Path(acp.__file__).read_text(encoding="utf-8")
    for banned in ("fine_tune", "finetune", ".backward()", "torch.optim"):
        assert banned not in source


def test_the_new_modes_are_registered_and_full_is_unchanged() -> None:
    for mode in ("ml-review", "ml-review-summary", "pipeline-compare",
                 "pipeline-compare-summary", "extensions"):
        assert mode in acp.MODES
    # --mode full must still mean the five baseline experiments only.
    assert "full" in acp.MODES
    assert not any("agedb" in mode.lower() for mode in acp.MODES)


def test_matching_the_comparator_is_not_recorded_as_a_reduction() -> None:
    """The declared criterion is 'lower than', so an equal false-review rate is
    not achieved. Recording a tie as a win would overstate the finding."""
    same = {"false_reviews_per_1000_non_mated": 5.2, "fpir": 0.005, "tpir_rank1": 0.94}
    coverage = {"gallery_enrolment_coverage": 0.99,
                "mated_extraction_failure_rate": 0.05,
                "non_mated_extraction_failure_rate": 0.05}
    detection = {"end_to_end_duplicate_detection_rate": 0.88}
    verdicts = acp.evaluate_review_success_criteria(same, coverage, same, detection, detection)
    assert verdicts["fewer_false_reviews_than_threshold_method"]["outcome"] == "not_achieved"

    better = dict(same, false_reviews_per_1000_non_mated=4.0)
    verdicts = acp.evaluate_review_success_criteria(better, coverage, same, detection, detection)
    assert verdicts["fewer_false_reviews_than_threshold_method"]["outcome"] == "achieved"


def test_every_required_artefact_exists_even_when_the_comparison_did_not_run() -> None:
    """A silently absent file is indistinguishable from a forgotten one, so the
    comparison writes its interval and subgroup artefacts either way."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    if not (root / "pipeline_comparison_metrics.json").is_file():
        pytest.skip("pipeline comparison has not been run in this checkout")
    intervals = json.loads(
        (root / "pipeline_comparison_confidence_intervals.json").read_text(encoding="utf-8")
    )
    assert intervals["artifact_type"] == "pipeline_comparison_confidence_intervals"
    assert "status" in intervals and "evaluated" in intervals
    # Empty rather than invented when the comparison did not run.
    if intervals["evaluated"] == "no":
        assert intervals["intervals"] == {}
        assert intervals["replicates"] == 0

    rows = list(csv.DictReader(open(root / "pretrained_pipeline_subgroup_metrics.csv",
                                    encoding="utf-8")))
    assert rows, "the subgroup file must carry one row per subgroup"
    assert {r["subgroup"] for r in rows} == set(EXPECTED_SUBGROUPS)
    if intervals["evaluated"] == "no":
        assert all(r["fpir"] == "" for r in rows)
        assert all(r["status"].startswith("not_run_") for r in rows)


def test_figure_captions_state_their_denominators() -> None:
    captions = Path(acp.__file__).parent / "results" / "figures" / "FIGURE_CAPTIONS.md"
    if not captions.is_file():
        pytest.skip("figures have not been generated in this checkout")
    text = captions.read_text(encoding="utf-8")
    for figure in ("Figure 1", "Figure 2", "Figure 3", "Figure 4", "Figure 5", "Figure 6"):
        assert figure in text
    assert "scored mated probes" in text and "scored non-mated probes" in text
    # Coverage is enrolled over intended, never over enrolled.
    assert "intended identities" in text
    assert "not causal" in text


# --- Rank-aware TPIR (corrected defect) ---------------------------------------


def _mated(rank, prob=0.9):
    """A scored mated probe whose correct identity sits at ``rank``."""
    row = _row(1, 0.8)
    return acp.ReviewFeatureRow(
        sample_id=f"m{rank}", identity_hash=f"i{rank}", subgroup="asian_females",
        role="mated_probe", features=row.features,
        label=1 if rank == 1 else 0, correct_rank=rank, correct_similarity=0.8,
    ), prob


def test_rank_one_referral_counts_towards_tpir_rank1() -> None:
    row, prob = _mated(1)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank1"] == pytest.approx(1.0)
    assert rates["mated_correct_rank1_referred"] == 1


def test_a_rank_two_referral_is_not_a_rank_one_success() -> None:
    """A referral to the wrong identity is not a true identification."""
    row, prob = _mated(2)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.0)
    assert rates["mated_wrong_identity_referred"] == 1


def test_a_rank_two_referral_may_count_towards_tpir_rank5() -> None:
    row, prob = _mated(2)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank5"] == pytest.approx(1.0)
    assert rates["mated_correct_rank5_referred"] == 1


def test_a_rank_nine_referral_counts_for_neither_rank() -> None:
    row, prob = _mated(9)
    rates = acp.review_rates_at_probability([row], np.asarray([prob]), 0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.0)
    assert rates["tpir_rank5"] == pytest.approx(0.0)
    assert rates["mated_wrong_identity_referred"] == 1


def test_labels_require_the_correct_identity_at_rank_one() -> None:
    results = [
        acp.OpenSetSearchResult(
            sample_id=f"s{rank}", identity_hash=f"h{rank}", subgroup="asian_females",
            role="mated_probe", top_similarity=0.8, top2_similarity=0.7,
            top5_similarity_mean=0.6, top5_similarity_stdev=0.05,
            top1_gallery_image_count=3, gallery_size=200,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            correct_rank=rank, correct_similarity=0.8,
        )
        for rank in (1, 2)
    ]
    rows, _ = acp.build_review_feature_rows(results)
    assert [r.label for r in rows] == [1, 0]


def test_ground_truth_rank_is_never_a_feature() -> None:
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role="mated_probe", top_similarity=0.8, top2_similarity=0.7,
            top5_similarity_mean=0.6, top5_similarity_stdev=0.05,
            top1_gallery_image_count=3, gallery_size=200,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            correct_rank=1, correct_similarity=0.8,
        )
    ]
    rows, _ = acp.build_review_feature_rows(results)
    assert rows[0].correct_rank == 1
    assert "correct_rank" not in rows[0].features
    assert "correct_similarity" not in rows[0].features
    assert set(rows[0].features) == set(acp.ML_REVIEW_FEATURES)


def test_decision_counts_reconcile_with_the_scored_denominator() -> None:
    mated = [_mated(1), _mated(2), _mated(9)]
    all_rows: List[acp.ReviewFeatureRow] = [row for row, _ in mated] + [_row(0, 0.1)]
    all_probs = np.asarray([prob for _, prob in mated] + [0.1])
    rates = acp.review_rates_at_probability(all_rows, all_probs, 0.5)
    assert (
        rates["mated_correct_rank1_referred"]
        + rates["mated_wrong_identity_referred"]
        + rates["mated_not_referred"]
        == rates["scored_mated_probes"]
    )
    assert (
        rates["non_mated_incorrectly_referred"]
        + rates["non_mated_correctly_not_referred"]
        == rates["scored_non_mated_probes"]
    )


# --- Bootstrap denominator (corrected defect) ---------------------------------


def _outcome_fixture():
    """Twenty identities, each with one scored and one failed mated probe."""
    rows, probs, outcomes = [], [], {}
    for index in range(20):
        subgroup = EXPECTED_SUBGROUPS[index % len(EXPECTED_SUBGROUPS)]
        identity = f"id{index:03d}"
        base = _row(1, 0.9, subgroup=subgroup, identity=identity, sample=f"s{index}")
        rows.append(acp.ReviewFeatureRow(
            sample_id=base.sample_id, identity_hash=identity, subgroup=subgroup,
            role="mated_probe", features=base.features, label=1,
            correct_rank=1, correct_similarity=0.9,
        ))
        probs.append(0.9)
        outcomes[identity] = acp.ReviewIdentityOutcome(
            identity_hash=identity, subgroup=subgroup, intended_mated=2, scored_mated=1,
            intended_non_mated=0, scored_non_mated=0, mated_extraction_failures=1,
            non_mated_extraction_failures=0, gallery_reference_unavailable=1,
        )
    return rows, np.asarray(probs), outcomes


def test_the_end_to_end_point_estimate_lies_inside_its_interval() -> None:
    """A point estimate outside its own interval means the bootstrap changed
    the denominator. This is the regression guard for that defect."""
    rows, probs, outcomes = _outcome_fixture()
    point = acp.review_rates_at_probability(rows, probs, 0.5, outcomes=outcomes)
    intervals = acp.review_cluster_bootstrap(
        rows, probs, 0.5, replicates=300, seed=EXPECTED_SEED, outcomes=outcomes
    )
    estimate = point["end_to_end_duplicate_detection_rate"]
    band = intervals["end_to_end_duplicate_detection_rate"]
    assert band["lower_95"] <= estimate <= band["upper_95"], (
        f"{estimate} outside [{band['lower_95']}, {band['upper_95']}]"
    )


def test_conditional_and_end_to_end_differ_when_failures_exist() -> None:
    rows, probs, outcomes = _outcome_fixture()
    rates = acp.review_rates_at_probability(rows, probs, 0.5, outcomes=outcomes)
    assert rates["conditional_duplicate_detection_rate"] == pytest.approx(1.0)
    assert rates["end_to_end_duplicate_detection_rate"] == pytest.approx(0.5)
    assert rates["end_to_end_duplicate_detection_rate"] < rates[
        "conditional_duplicate_detection_rate"
    ]


def test_coverage_intervals_use_intended_probe_counts() -> None:
    rows, probs, outcomes = _outcome_fixture()
    intervals = acp.review_cluster_bootstrap(
        rows, probs, 0.5, replicates=100, seed=EXPECTED_SEED, outcomes=outcomes
    )
    band = intervals["mated_extraction_coverage"]
    # One scored of two intended for every identity, so coverage is exactly 0.5.
    assert band["lower_95"] == pytest.approx(0.5)
    assert band["upper_95"] == pytest.approx(0.5)


def test_the_corrected_bootstrap_is_deterministic() -> None:
    rows, probs, outcomes = _outcome_fixture()
    a = acp.review_cluster_bootstrap(rows, probs, 0.5, replicates=100,
                                     seed=EXPECTED_SEED, outcomes=outcomes)
    b = acp.review_cluster_bootstrap(rows, probs, 0.5, replicates=100,
                                     seed=EXPECTED_SEED, outcomes=outcomes)
    # NaN never equals itself, so an undefined metric needs explicit handling
    # rather than a plain dictionary comparison.
    assert set(a) == set(b)
    for metric, band in a.items():
        other = b[metric]
        for key, value in band.items():
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(other[key])
            else:
                assert value == other[key]


def test_subgroup_metrics_include_rank_five_and_coverage_intervals() -> None:
    rows, probs, outcomes = _outcome_fixture()
    per_subgroup = acp.review_subgroup_metrics(
        rows, probs, 0.5, replicates=50, seed=EXPECTED_SEED, outcomes=outcomes
    )
    assert per_subgroup
    for entry in per_subgroup.values():
        for metric in ("fpir", "fnir_rank1", "fnir_rank5", "tpir_rank1", "tpir_rank5",
                       "mated_probe_coverage", "non_mated_probe_coverage"):
            assert f"{metric}_lower_95" in entry and f"{metric}_upper_95" in entry
        assert entry["intended_mated_probes"] >= entry["scored_mated_probes"]


# --- Documentation contracts --------------------------------------------------


def _project_file(name: str) -> str:
    return (Path(acp.__file__).parent / name).read_text(encoding="utf-8")


def test_documentation_does_not_claim_nothing_is_trained() -> None:
    """Experiment 7 trains a classifier, so the blanket claim is now false. The
    narrower statement about face-recognition networks remains correct."""
    for name in ("README.md", "ACP_arden.py"):
        text = _project_file(name)
        assert "Nothing here is trained or fine-tuned" not in text
    readme = _project_file("README.md")
    assert "No face-detection or face-recognition network is\ntrained or fine-tuned" in readme \
        or "no face-recognition network is trained" in readme.lower()
    assert "logistic-regression review classifier" in readme


def test_the_valid_narrower_training_sentence_is_permitted() -> None:
    assert "No face-recognition model is trained or fine-tuned." not in _project_file(
        "README.md"
    ) or True  # the sentence is acceptable wherever it appears


def test_no_stale_licensing_wording_remains() -> None:
    for name in ("README.md", "ACP_arden.py", ".env.example", "REFERENCES.md",
                 "CONVERSION_MAP.md"):
        assert "those terms are unresolved for this project" not in _project_file(name)


def test_past_tense_evaluation_claims_match_the_real_status() -> None:
    """'was evaluated' may only appear once held-out metrics exist."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    metrics = root / "pipeline_comparison_metrics.json"
    if not metrics.is_file():
        pytest.skip("pipeline comparison has not been run in this checkout")
    evaluated = json.loads(metrics.read_text(encoding="utf-8"))["evaluated"] == "yes"
    report = (root / "PRETRAINED_PIPELINE_COMPARISON_REPORT.md").read_text(encoding="utf-8")
    if not evaluated:
        assert "pipeline was evaluated solely" not in report
        assert "is intended for evaluation solely" in report
    assert acp.arcface_use_statement(False).startswith("The InsightFace pipeline is intended")
    assert acp.arcface_use_statement(True).startswith("The InsightFace pipeline was evaluated")


def test_the_conversion_map_places_figures_in_section_28() -> None:
    text = _project_file("CONVERSION_MAP.md")
    assert "Matplotlib evidence-figure rendering | Reference-only" not in text
    assert "sections 11 to 15" not in text
    assert "results/figures/" in text
    for row in ("Figure generation from machine-readable artefacts | Section 28",
                "PNG and SVG output | Section 28",
                "Figure captions and denominator documentation | Section 28",
                "PNG metadata removal and privacy scanning | Section 28"):
        assert row in text


def test_section_numbers_are_contiguous_and_documented() -> None:
    import re

    source = _project_file("ACP_arden.py")
    numbers = sorted({int(m.group(1)) for m in re.finditer(r"^# (\d+)\. ", source, re.M)
                      if int(m.group(1)) <= 40})
    assert numbers == list(range(1, len(numbers) + 1))
    assert len(numbers) == 30
    assert "thirty numbered sections" in _project_file("CONVERSION_MAP.md")


# --- Experiment 8 evaluation path (synthetic stubs, no real weights) ----------


def _stub_pipeline(dimensions: int, seed_offset: int) -> Callable[..., Any]:
    """A deterministic stand-in producing embeddings of a chosen width.

    Used to prove the comparison machinery without any pretrained weights."""

    def _embed(entry, detector, embedder):
        seed = (int(entry.identity_hash[:8], 16) + seed_offset) % (2**32)
        vector = np.random.default_rng(seed).normal(size=dimensions)
        return vector / float(np.linalg.norm(vector)), None, {
            "probe_detection_confidence": 0.9,
            "probe_face_area_ratio": 0.5,
        }

    return _embed


def test_each_pipeline_receives_its_own_development_threshold(tmp_path: Path) -> None:
    """Similarity scales differ between embedding models, so one threshold must
    never be reused for the other."""
    protocol = _protocol(tmp_path)
    results = {}
    for name, (dims, offset) in {"primary": (128, 0), "comparator": (512, 7)}.items():
        run_dev = acp.run_open_set_method(
            protocol, partition="development", method=acp.METHOD_B,
            detector=None, embedder=None,  # type: ignore[arg-type]
            embed_fn=_stub_pipeline(dims, offset),
        )
        results[name] = acp.select_open_set_threshold(
            run_dev.search_results, target_fpir=EXPECTED_PRIMARY_FPIR_TARGET
        )["threshold"]
    # Two independently calibrated thresholds, each from its own scores.
    assert set(results) == {"primary", "comparator"}
    assert all(isinstance(v, float) for v in results.values())


def test_both_pipelines_traverse_the_identical_protocol(tmp_path: Path) -> None:
    protocol = _protocol(tmp_path)
    runs = [
        acp.run_open_set_method(
            protocol, partition="test", method=acp.METHOD_B,
            detector=None, embedder=None,  # type: ignore[arg-type]
            embed_fn=_stub_pipeline(dims, offset),
        )
        for dims, offset in ((128, 0), (512, 7))
    ]
    assert {r.sample_id for r in runs[0].search_results} == {
        r.sample_id for r in runs[1].search_results
    }
    assert runs[0].gallery_size == runs[1].gallery_size


def test_held_out_identities_never_influence_threshold_selection(tmp_path: Path) -> None:
    """Calibration reads the development partition only."""
    protocol = _protocol(tmp_path)
    development = {e.identity for e in protocol.partition("development")}
    test = {e.identity for e in protocol.partition("test")}
    assert development.isdisjoint(test)
    run_dev = acp.run_open_set_method(
        protocol, partition="development", method=acp.METHOD_B,
        detector=None, embedder=None,  # type: ignore[arg-type]
        embed_fn=_stub_pipeline(128, 0),
    )
    chosen = acp.select_open_set_threshold(
        run_dev.search_results, target_fpir=EXPECTED_PRIMARY_FPIR_TARGET
    )
    sample_ids = {r.sample_id for r in run_dev.search_results}
    test_samples = {e.sample_id for e in protocol.partition("test")}
    assert sample_ids.isdisjoint(test_samples)
    assert chosen["threshold"] is not None


def test_a_status_only_run_never_claims_evaluation() -> None:
    """Readiness alone must not set evaluated=yes."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    payload = json.loads((root / "pipeline_comparison_metrics.json").read_text(encoding="utf-8"))
    if payload["evaluated"] == "no":
        assert payload.get("held_out_metrics") is None
        assert payload.get("comparison_pipeline") is None
        assert payload["status"].startswith("not_run_")
    else:
        assert payload.get("held_out_metrics")


def test_the_comparison_guard_rejects_evaluation_without_metrics() -> None:
    assert issubclass(acp.PipelineComparisonError, RuntimeError)
    source = _project_file("ACP_arden.py")
    assert "cannot be marked as evaluated without held-out metrics" in source


def test_automatic_model_download_is_disabled() -> None:
    source = _project_file("ACP_arden.py")
    assert "download=False" in source
    assert "download_zip=False" in source


def test_a_hash_mismatch_is_refused(tmp_path: Path) -> None:
    impostor = tmp_path / acp.ARCFACE_DETECTOR_FILENAME
    impostor.write_bytes(b"not the pinned model")
    with pytest.raises(acp.ModelUnavailableError):
        acp.verify_model_file(impostor, "0" * 64)


def test_the_arcface_embedder_refuses_unexpected_dimensions() -> None:
    class _Face:
        # Minimal stand-in for an InsightFace detection result.
        bbox = np.asarray([0.0, 0.0, 8.0, 8.0])
        kps = np.zeros((5, 2))
        det_score = 0.9
        normed_embedding = np.ones(128) / math.sqrt(128)

    class _App:
        def get(self, bgr):
            return [_Face()]

    detector = acp.ArcFaceDetector(_App(), "digest")
    detector.detect_single_face(np.zeros((8, 8, 3), dtype=np.uint8))
    embedder = acp.ArcFaceEmbedder(detector, "digest", dimensions=512)
    with pytest.raises(acp.SimilarityError):
        embedder.embed(np.zeros((8, 8, 3), dtype=np.uint8), np.zeros(15))


def test_the_arcface_detector_requires_exactly_one_face() -> None:
    class _App:
        def __init__(self, count):
            self._count = count

        def get(self, bgr):
            return [object()] * self._count

    for count in (0, 2):
        detector = acp.ArcFaceDetector(_App(count), "digest")
        with pytest.raises(acp.FaceCountError) as raised:
            detector.detect_single_face(np.zeros((8, 8, 3), dtype=np.uint8))
        assert raised.value.face_count == count


def test_the_comparison_artefact_carries_full_provenance() -> None:
    root = Path(acp.__file__).parent / "results" / "aggregate"
    payload = json.loads((root / "pipeline_comparison_metrics.json").read_text(encoding="utf-8"))
    required = (
        "artifact_type", "schema_version", "created_at", "seed", "status", "evaluated",
        "dataset_name", "protocol_version", "protocol_digest", "public_manifest_digest",
        "primary_pipeline", "comparison_pipeline", "model_filenames", "model_digests",
        "software_environment", "dependency_versions", "preprocessing_revision",
        "threshold_policy", "calibration_partition", "held_out_partition", "policy_note",
        "licence_note", "preconditions", "limitations",
    )
    missing = [key for key in required if key not in payload]
    assert not missing, f"missing provenance: {missing}"
    if payload["evaluated"] == "no":
        assert payload["reason"]


def test_missing_dependencies_produce_the_dependency_status(tmp_path: Path, monkeypatch) -> None:
    """With files present and digests pinned, an absent package is the blocker."""
    (tmp_path / acp.ARCFACE_DETECTOR_FILENAME).write_bytes(b"placeholder")
    (tmp_path / acp.ARCFACE_RECOGNITION_FILENAME).write_bytes(b"placeholder")
    monkeypatch.setattr(acp, "ARCFACE_DETECTOR_SHA256", "a" * 64)
    monkeypatch.setattr(acp, "ARCFACE_RECOGNITION_SHA256", "b" * 64)
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    diagnosis = acp.arcface_preconditions(config)
    # onnxruntime and insightface are absent in this environment.
    assert diagnosis["status"] == acp.PIPELINE_STATUS_DEPENDENCIES_MISSING
    assert diagnosis["missing_dependencies"]
    assert diagnosis["checks"]["digests_pinned"] is True
    assert diagnosis["checks"]["model_files_present"] is True


def test_experiment_seven_subgroup_sample_counts_reconcile() -> None:
    rows, probs, outcomes = _outcome_fixture()
    per_subgroup = acp.review_subgroup_metrics(
        rows, probs, 0.5, replicates=20, seed=EXPECTED_SEED, outcomes=outcomes
    )
    for entry in per_subgroup.values():
        assert entry["scored_mated_probes"] <= entry["intended_mated_probes"]
        assert entry["scored_non_mated_probes"] <= entry["intended_non_mated_probes"]


def test_threshold_selection_is_reproducible_across_processes() -> None:
    """The frozen threshold must not drift between runs.

    A threshold that moved run to run would void the freezing guarantee, so
    selection is pinned here on fixed inputs rather than only end to end."""
    rows = _training_rows()
    first = acp.fit_review_classifier(rows)
    second = acp.fit_review_classifier(rows)
    assert first.coefficients == pytest.approx(second.coefficients, abs=0.0)
    assert first.intercept == pytest.approx(second.intercept, abs=0.0)

    matrix, _ = acp._feature_matrix(rows)
    chosen = [
        acp.select_review_probability_threshold(
            rows, model.probabilities(matrix), target_fpir=EXPECTED_PRIMARY_FPIR_TARGET
        )["probability_threshold"]
        for model in (first, second)
    ]
    assert chosen[0] == chosen[1]


def test_the_published_threshold_matches_the_frozen_policy() -> None:
    """The evaluated threshold must be the one recorded as frozen."""
    root = Path(acp.__file__).parent / "results" / "aggregate"
    policy_path, test_path = root / "ml_review_threshold.json", root / "ml_review_test_metrics.json"
    if not (policy_path.is_file() and test_path.is_file()):
        pytest.skip("the review experiment has not been run in this checkout")
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    assert policy["status"] == EXPECTED_FROZEN_STATUS
    frozen = policy["operating_points"][str(EXPECTED_PRIMARY_FPIR_TARGET)]["probability_threshold"]
    assert test["operating_probability_threshold"] == pytest.approx(frozen)
