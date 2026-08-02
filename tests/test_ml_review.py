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
from typing import Dict, List

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


def test_labels_follow_the_protocol_roles() -> None:
    results = [
        acp.OpenSetSearchResult(
            sample_id="a" * 32, identity_hash="h" * 32, subgroup="asian_females",
            role=role, top_similarity=0.8, top2_similarity=0.7,
            top5_similarity_mean=0.6, top5_similarity_stdev=0.05,
            top1_gallery_image_count=3, gallery_size=200,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
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
    assert rates["mated_probes_correctly_referred"] == 1
    assert rates["mated_probes_not_referred"] == 1
    assert rates["non_mated_probes_incorrectly_referred"] == 1
    assert rates["non_mated_probes_correctly_not_referred"] == 1
    assert rates["scored_mated_probes"] == 2
    assert rates["scored_non_mated_probes"] == 2
    assert rates["fpir"] == pytest.approx(0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.5)
    assert rates["fnir_rank1"] == pytest.approx(0.5)


def test_the_end_to_end_denominator_counts_every_intended_mated_probe() -> None:
    rows = [_row(1, 0.9), _row(0, 0.1)]
    probabilities = np.asarray([0.9, 0.1])
    rates = acp.review_rates_at_probability(rows, probabilities, 0.5, intended_mated=4)
    # One referral out of four intended, not out of the one that was scored.
    assert rates["end_to_end_duplicate_detection_rate"] == pytest.approx(0.25)
    assert rates["tpir_rank1"] == pytest.approx(1.0)


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
    assert "trains and fine-tunes nothing" in note or "trains nothing" in note
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
