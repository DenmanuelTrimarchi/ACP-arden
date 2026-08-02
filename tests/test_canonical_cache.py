"""Acceptance tests for the canonical run cache, its context validation and
the outcome digest.

The cache is what guarantees Experiments 6, 7 and 8 report identical figures
for the same method, so it must be rejected whenever any input capable of
changing its result differs. Matching probe identifiers is not sufficient: the
same images scored with a different model, detector setting or OpenCV build
produce different outcomes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

import ACP_arden as acp

EXPECTED_SEED = 20260727
EXPECTED_CACHE_SCHEMA_VERSION = 2


def _run(partition: str = "test") -> acp.OpenSetRunResult:
    """A small deterministic run standing in for a scored partition."""
    results: List[acp.OpenSetSearchResult] = []
    outcomes: List[acp.EnrolmentOutcome] = []
    for index in range(6):
        subgroup = acp.BFW_SUBGROUPS[index % len(acp.BFW_SUBGROUPS)]
        results.append(acp.OpenSetSearchResult(
            sample_id=f"m{index:031d}", identity_hash=f"{index:032d}", subgroup=subgroup,
            role="mated_probe", top_similarity=0.8, top2_similarity=0.7,
            correct_rank=1, correct_similarity=0.8, top5_similarity_mean=0.6,
            top5_similarity_stdev=0.05, top1_gallery_image_count=3, gallery_size=6,
            probe_detection_confidence=0.9, probe_face_area_ratio=0.5,
            top1_time_seconds=0.001, top5_time_seconds=0.002,
        ))
        results.append(acp.OpenSetSearchResult(
            sample_id=f"n{index:031d}", identity_hash=f"x{index:031d}", subgroup=subgroup,
            role="non_mated_probe", top_similarity=0.2, top1_time_seconds=0.001,
        ))
        outcomes.append(acp.EnrolmentOutcome(
            identity_hash=f"{index:032d}", subgroup=subgroup, enrolled=True,
            attempted_images=3, embedded_images=3, failure_code=None,
        ))
    return acp.OpenSetRunResult(
        method=acp.METHOD_B, partition=partition, enrolment_outcomes=outcomes,
        search_results=results, gallery_size=6, comparisons_per_probe=6,
        stage_times_seconds={"complete_pipeline_seconds": [0.01, 0.02]},
    )


def _context() -> Dict[str, Any]:
    return {
        "cache_schema_version": EXPECTED_CACHE_SCHEMA_VERSION,
        "partition": "test",
        "dataset_metadata_sha256": "a" * 64,
        "evaluated_image_set_sha256": "b" * 64,
        "protocol_version": acp.BFW_PROTOCOL_VERSION,
        "protocol_digest": "c" * 64,
        "public_manifest_digest": "c" * 64,
        "model_filenames": [acp.YUNET_FILENAME, acp.SFACE_FILENAME],
        "model_sha256": {"yunet": acp.YUNET_SHA256, "sface": acp.SFACE_SHA256},
        "preprocessing_revision": acp.PREPROCESSING_REVISION,
        "detector_input_size": [320, 320],
        "detector_score_threshold": acp.DETECTOR_SCORE_THRESHOLD,
        "embedding_dimensions": acp.EMBEDDING_DIMENSIONS,
        "gallery_images_per_identity": acp.MULTI_IMAGE_ENROLMENT,
        "seed": EXPECTED_SEED,
        "opencv_version": "4.13.0",
        "gallery_enrolment_samples": ["g1", "g2"],
        "mated_probe_samples": ["m1"],
        "non_mated_probe_samples": ["n1"],
    }


# --- Cache context validation -------------------------------------------------


def test_a_matching_context_is_accepted(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    assert acp.cache_invalidation_reason(path, _context()) is None


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_metadata_sha256", "f" * 64),
        ("protocol_digest", "f" * 64),
        ("preprocessing_revision", "changed-revision"),
        ("detector_score_threshold", 0.5),
        ("detector_input_size", [640, 640]),
        ("embedding_dimensions", 512),
        ("opencv_version", "5.0.0"),
        ("gallery_enrolment_samples", ["g1", "g3"]),
        ("mated_probe_samples", ["m2"]),
        ("non_mated_probe_samples", ["n2"]),
        ("seed", 1),
        ("gallery_images_per_identity", 1),
    ],
)
def test_any_changed_context_field_invalidates_the_cache(
    tmp_path: Path, field: str, value: Any
) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    changed = {**_context(), field: value}
    reason = acp.cache_invalidation_reason(path, changed)
    assert reason is not None, f"changing {field} did not invalidate the cache"
    assert field in reason


def test_a_changed_model_digest_invalidates_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    changed = {**_context(), "model_sha256": {"yunet": "0" * 64, "sface": acp.SFACE_SHA256}}
    reason = acp.cache_invalidation_reason(path, changed)
    assert reason is not None and "model_sha256" in reason


def test_a_changed_cache_schema_invalidates_the_cache(tmp_path: Path) -> None:
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["cache_schema_version"] = EXPECTED_CACHE_SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")
    reason = acp.cache_invalidation_reason(path, _context())
    assert reason is not None and "schema" in reason


def test_a_missing_or_contextless_cache_is_rejected(tmp_path: Path) -> None:
    assert acp.cache_invalidation_reason(tmp_path / "absent.json", _context()) is not None
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, None)
    assert acp.cache_invalidation_reason(path, _context()) is not None


def test_the_invalidation_reason_never_carries_context_values(tmp_path: Path) -> None:
    """Values can include private material, so only field names are reported."""
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    changed = {**_context(), "dataset_metadata_sha256": "f" * 64}
    reason = acp.cache_invalidation_reason(path, changed) or ""
    assert "f" * 64 not in reason and "a" * 64 not in reason


# --- Canonical outcome digest -------------------------------------------------


def test_timing_only_changes_do_not_alter_the_digest() -> None:
    run = _run()
    baseline = acp.canonical_run_digest(run)
    retimed = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=run.enrolment_outcomes,
        search_results=[
            acp.OpenSetSearchResult(**{
                **{f.name: getattr(r, f.name) for f in __import__("dataclasses").fields(r)},
                "top1_time_seconds": 99.0, "top5_time_seconds": 99.0,
            })
            for r in run.search_results
        ],
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
        stage_times_seconds={"complete_pipeline_seconds": [9.9]},
    )
    assert acp.canonical_run_digest(retimed) == baseline


def test_reordering_equivalent_records_does_not_alter_the_digest() -> None:
    run = _run()
    reordered = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=list(reversed(run.enrolment_outcomes)),
        search_results=list(reversed(run.search_results)),
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
        stage_times_seconds=run.stage_times_seconds,
    )
    assert acp.canonical_run_digest(reordered) == acp.canonical_run_digest(run)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("failure_code", "zero_faces"), ("top_similarity", 0.55),
        ("top2_similarity", 0.11), ("correct_rank", 4), ("correct_similarity", 0.33),
        ("top5_similarity_mean", 0.42), ("top5_similarity_stdev", 0.9),
        ("top1_gallery_image_count", 1), ("gallery_size", 99),
        ("probe_detection_confidence", 0.1), ("probe_face_area_ratio", 0.99),
        ("subgroup", "white_males"), ("role", "non_mated_probe"),
        ("top_identity_hash", "z" * 32),
    ],
)
def test_every_non_timing_search_field_affects_the_digest(field: str, value: Any) -> None:
    import dataclasses

    run = _run()
    baseline = acp.canonical_run_digest(run)
    first = run.search_results[0]
    mutated = acp.OpenSetSearchResult(**{
        **{f.name: getattr(first, f.name) for f in dataclasses.fields(first)}, field: value
    })
    changed = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=run.enrolment_outcomes,
        search_results=[mutated] + list(run.search_results[1:]),
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
    )
    assert acp.canonical_run_digest(changed) != baseline, f"{field} did not affect the digest"


@pytest.mark.parametrize(
    ("field", "value"),
    [("enrolled", False), ("attempted_images", 1), ("embedded_images", 0),
     ("failure_code", "insufficient_gallery_images"), ("subgroup", "black_males")],
)
def test_every_enrolment_field_affects_the_digest(field: str, value: Any) -> None:
    import dataclasses

    run = _run()
    baseline = acp.canonical_run_digest(run)
    first = run.enrolment_outcomes[0]
    mutated = acp.EnrolmentOutcome(**{
        **{f.name: getattr(first, f.name) for f in dataclasses.fields(first)}, field: value
    })
    changed = acp.OpenSetRunResult(
        method=run.method, partition=run.partition,
        enrolment_outcomes=[mutated] + list(run.enrolment_outcomes[1:]),
        search_results=run.search_results,
        gallery_size=run.gallery_size, comparisons_per_probe=run.comparisons_per_probe,
    )
    assert acp.canonical_run_digest(changed) != baseline, f"{field} did not affect the digest"


def test_the_cache_records_its_privacy_position_accurately(tmp_path: Path) -> None:
    """It holds derived face-comparison scores, so describing it as carrying no
    biometric record at all would be wrong."""
    path = tmp_path / "run.json"
    acp.save_canonical_run(_run(), path, _context())
    note = json.loads(path.read_text(encoding="utf-8"))["privacy_note"]
    assert "privacy-sensitive derived face-comparison" in note
    assert "no raw photographs, face embeddings or enrolled templates" in note
    assert "excluded from Git" in note


def test_the_cache_location_is_git_ignored() -> None:
    ignore = (Path(acp.__file__).parent / ".gitignore").read_text(encoding="utf-8")
    assert "results/raw/" in ignore
    assert str(acp.CANONICAL_RUN_CACHE).startswith(str(acp.RAW_ROOT))
