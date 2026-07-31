"""Tests for the BFW adapter, the identity-disjoint open-set protocol, the two
enrolment methods, the FPIR/FNIR metrics and the cluster bootstrap.

BFW itself is an access-gated external dataset and is not present in this
repository or in CI. Everything below therefore runs against a synthetic fixture
that reproduces the official layout — ``<subgroup>/<identity>/<image>.jpg`` with
the pinned datatable columns — so the protocol logic, the metric definitions and
the freezing guarantees are exercised without the real images.

That is the limit of what these tests establish: they pin behaviour, not
accuracy. No figure produced here is a research result.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

import ACP_arden as acp

EXPECTED_SEED = 20260727
EXPECTED_SUBGROUPS = (
    "asian_females", "asian_males", "black_females", "black_males",
    "indian_females", "indian_males", "white_females", "white_males",
)
EXPECTED_FPIR_TARGETS = (0.001, 0.003, 0.01)
EXPECTED_PRIMARY_FPIR_TARGET = 0.003
EXPECTED_FROZEN_STATUS = "open_set_frozen"
EXPECTED_ROLES = ("gallery_enrolment", "mated_probe", "non_mated_probe")


# --- Synthetic BFW fixture ----------------------------------------------------


def _make_bfw(tmp_path: Path, *, identities_per_subgroup: int = 6, images: int = 25):
    """Build a fixture matching the official BFW layout and datatable schema."""
    root = tmp_path / "bfw-images"
    rows: List[Dict[str, str]] = []
    for subgroup in EXPECTED_SUBGROUPS:
        for index in range(identities_per_subgroup):
            identity = f"n{index:06d}"
            folder = root / subgroup / identity
            folder.mkdir(parents=True, exist_ok=True)
            paths = []
            for image_index in range(images):
                name = f"{image_index:04d}_01.jpg"
                (folder / name).write_bytes(b"not-a-real-image")
                paths.append(f"{subgroup}/{identity}/{name}")
            # The datatable is a pair table; every image must appear at least once.
            for first, second in zip(paths, paths[1:] + paths[:1]):
                rows.append(
                    {"fold": "1", "p1": first, "p2": second, "label": "1",
                     "att1": subgroup, "att2": subgroup}
                )

    metadata = tmp_path / "bfw-v0.1.5-datatable.csv"
    with open(metadata, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold", "p1", "p2", "label", "att1", "att2"])
        writer.writeheader()
        writer.writerows(rows)
    return root, metadata


@pytest.fixture()
def bfw(tmp_path: Path) -> acp.BfwDataset:
    root, metadata = _make_bfw(tmp_path)
    return acp.load_bfw_dataset(root, metadata)


# --- Adapter ------------------------------------------------------------------


def test_the_adapter_reads_the_official_layout(bfw: acp.BfwDataset) -> None:
    grouped = bfw.by_identity()
    assert len(grouped) == len(EXPECTED_SUBGROUPS) * 6
    assert all(len(v) == 25 for v in grouped.values())
    assert set(bfw.subgroup_of_identity().values()) == set(EXPECTED_SUBGROUPS)


def test_a_missing_required_column_is_refused(tmp_path: Path) -> None:
    root, metadata = _make_bfw(tmp_path)
    text = metadata.read_text().replace("att1", "attribute_one", 1)
    metadata.write_text(text)
    with pytest.raises(acp.BfwDatasetError) as raised:
        acp.load_bfw_dataset(root, metadata)
    message = str(raised.value)
    assert "att1" in message and "found" in message


def test_a_path_outside_the_official_layout_is_refused(tmp_path: Path) -> None:
    root, metadata = _make_bfw(tmp_path)
    rows = metadata.read_text().splitlines()
    rows[1] = rows[1].replace("asian_females/n000000/0000_01.jpg", "loose_image.jpg")
    metadata.write_text("\n".join(rows) + "\n")
    with pytest.raises(acp.BfwDatasetError):
        acp.load_bfw_dataset(root, metadata)


def test_an_unknown_subgroup_is_refused(tmp_path: Path) -> None:
    root, metadata = _make_bfw(tmp_path)
    text = metadata.read_text().replace("asian_females/", "martian_females/")
    metadata.write_text(text)
    with pytest.raises(acp.BfwDatasetError) as raised:
        acp.load_bfw_dataset(root, metadata)
    assert "official subgroups" in str(raised.value)


def test_a_metadata_subgroup_contradicting_the_path_is_refused(tmp_path: Path) -> None:
    root, metadata = _make_bfw(tmp_path)
    lines = metadata.read_text().splitlines()
    header, first = lines[0], lines[1].split(",")
    first[4] = "white_males"  # att1 disagrees with the asian_females path
    metadata.write_text("\n".join([header, ",".join(first)] + lines[2:]) + "\n")
    with pytest.raises(acp.BfwDatasetError) as raised:
        acp.load_bfw_dataset(root, metadata)
    assert "Refusing to choose" in str(raised.value)


def test_a_missing_image_file_is_an_explicit_error(tmp_path: Path) -> None:
    root, metadata = _make_bfw(tmp_path)
    (root / "asian_females" / "n000000" / "0000_01.jpg").unlink()
    with pytest.raises(acp.BfwDatasetError) as raised:
        acp.load_bfw_dataset(root, metadata)
    assert "missing" in str(raised.value).lower()


def test_dataset_provenance_publishes_no_name_or_path(bfw: acp.BfwDataset) -> None:
    provenance = acp.bfw_dataset_provenance(bfw)
    rendered = repr(provenance)
    assert "n000000" not in rendered
    assert "/" not in provenance["metadata_filename"]
    assert provenance["total_identities"] == 48
    assert set(provenance["subgroup_identity_counts"]) == set(EXPECTED_SUBGROUPS)
    assert "does not publish" in provenance["protocol_provenance"].lower() or (
        "not publish" in provenance["protocol_provenance"].lower()
    )


# --- Protocol -----------------------------------------------------------------


def test_development_and_test_identities_are_disjoint(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    development = {e.identity for e in protocol.partition("development")}
    test = {e.identity for e in protocol.partition("test")}
    assert development and test
    assert development.isdisjoint(test)


def test_an_identity_is_never_both_enrolled_and_non_mated(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    for partition in ("development", "test"):
        enrolled = protocol.identities(partition, "gallery_enrolment")
        non_mated = protocol.identities(partition, "non_mated_probe")
        assert enrolled.isdisjoint(non_mated)


def test_one_image_cannot_hold_more_than_one_role(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    paths = [e.image_path for e in protocol.entries]
    assert len(paths) == len(set(paths))
    assert {e.role for e in protocol.entries} <= set(EXPECTED_ROLES)


def test_the_split_is_stratified_by_subgroup(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    subgroup_of = bfw.subgroup_of_identity()
    for partition in ("development", "test"):
        identities = {e.identity for e in protocol.partition(partition)}
        counts = {s: 0 for s in EXPECTED_SUBGROUPS}
        for identity in identities:
            counts[subgroup_of[identity]] += 1
        # Six identities per subgroup, halved: every subgroup contributes.
        assert all(count == 3 for count in counts.values()), counts


def test_the_partition_reproduces_under_the_research_seed(bfw: acp.BfwDataset) -> None:
    first = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    second = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    key = lambda p: sorted((e.partition, e.role, str(e.image_path)) for e in p.entries)
    assert key(first) == key(second)


def test_another_seed_changes_the_partition(bfw: acp.BfwDataset) -> None:
    first = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    other = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED + 1)
    key = lambda p: sorted((e.partition, str(e.image_path)) for e in p.entries)
    assert key(first) != key(other)


def test_the_partition_does_not_depend_on_the_identifier_key(tmp_path: Path) -> None:
    """Metrics must stay reproducible by someone who does not hold the secret
    key, so partitioning may depend only on the seed and the dataset."""
    root, metadata = _make_bfw(tmp_path)
    with acp.temporary_id_hmac_key("a" * 63 + "b"):
        first = acp.build_open_set_protocol(
            acp.load_bfw_dataset(root, metadata), seed=EXPECTED_SEED
        )
        first_key = sorted((e.partition, e.role, str(e.image_path)) for e in first.entries)
    with acp.temporary_id_hmac_key("c" * 63 + "d"):
        second = acp.build_open_set_protocol(
            acp.load_bfw_dataset(root, metadata), seed=EXPECTED_SEED
        )
        second_key = sorted((e.partition, e.role, str(e.image_path)) for e in second.entries)
    assert first_key == second_key


def test_the_public_protocol_summary_leaks_no_identity(bfw: acp.BfwDataset) -> None:
    summary = acp.open_set_protocol_summary(acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED))
    rendered = repr(summary)
    assert "n000000" not in rendered
    assert "/tmp" not in rendered
    assert summary["development"]["identities"] > 0
    assert summary["test"]["identities"] > 0


# --- Enrolment methods --------------------------------------------------------


def _embed_factory(
    fail_samples: Optional[Dict[str, str]] = None, *, noise: float = 0.0
):
    """Deterministic stub: one direction per identity, so mated probes match
    their own template and impostors do not."""
    fail_samples = fail_samples or {}

    def _embed(entry, detector, embedder):
        if entry.sample_id in fail_samples:
            return None, fail_samples[entry.sample_id]
        seed = int(entry.identity_hash[:8], 16)
        rng = np.random.default_rng(seed)
        base = rng.normal(size=32)
        if noise:
            jitter = np.random.default_rng(int(entry.sample_id[:8], 16)).normal(size=32)
            base = base + noise * jitter
        return base / float(np.linalg.norm(base)), None

    return _embed


def _run(protocol, method: str, partition: str = "development", **kwargs):
    return acp.run_open_set_method(
        protocol,
        partition=partition,
        method=method,
        detector=None,  # type: ignore[arg-type]
        embedder=None,  # type: ignore[arg-type]
        embed_fn=kwargs.pop("embed_fn", _embed_factory()),
        **kwargs,
    )


def test_template_averaging_returns_a_unit_vector() -> None:
    vectors = [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])]
    template = acp.build_identity_template(vectors)
    assert float(np.linalg.norm(template)) == pytest.approx(1.0)
    # Direction is the mean of the inputs, magnitude is renormalised to one.
    assert template.tolist() == pytest.approx([1 / math.sqrt(3)] * 3)


def test_both_methods_see_the_same_identity_split(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    control = _run(protocol, acp.METHOD_A)
    proposed = _run(protocol, acp.METHOD_B)
    assert {o.identity_hash for o in control.enrolment_outcomes} == {
        o.identity_hash for o in proposed.enrolment_outcomes
    }
    assert {r.sample_id for r in control.search_results} == {
        r.sample_id for r in proposed.search_results
    }


def test_the_single_image_method_enrols_one_image_per_identity(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    control = _run(protocol, acp.METHOD_A)
    assert all(o.attempted_images == 1 for o in control.enrolment_outcomes)


def test_the_three_image_method_enrols_up_to_three(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    proposed = _run(protocol, acp.METHOD_B)
    assert all(o.attempted_images == 3 for o in proposed.enrolment_outcomes)
    assert acp.MULTI_IMAGE_MINIMUM_ENROLMENT == 2


def test_insufficient_enrolment_images_are_explicit(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    rows = protocol.partition("development")
    victim = sorted(
        {e.identity_hash for e in rows if e.role == "gallery_enrolment"}
    )[0]
    # Fail two of the identity's three enrolment images, leaving one: below the
    # minimum of two for the three-image method.
    doomed = [e for e in rows if e.role == "gallery_enrolment" and e.identity_hash == victim]
    fail = {e.sample_id: "zero_faces" for e in sorted(doomed, key=lambda e: e.sample_id)[:2]}

    proposed = _run(protocol, acp.METHOD_B, embed_fn=_embed_factory(fail))
    outcome = next(o for o in proposed.enrolment_outcomes if o.identity_hash == victim)
    assert outcome.enrolled is False
    assert outcome.failure_code == "insufficient_gallery_images"
    assert outcome.embedded_images == 1

    # Its mated probe becomes a coverage failure, never a no-match decision.
    probe = next(
        r for r in proposed.search_results
        if r.role == "mated_probe" and r.identity_hash == victim
    )
    assert probe.failure_code == "gallery_reference_unavailable"
    assert probe.top_similarity is None


def test_an_extraction_failure_is_not_a_no_match_decision(bfw: acp.BfwDataset) -> None:
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    rows = protocol.partition("development")
    probe = next(e for e in rows if e.role == "non_mated_probe")
    proposed = _run(protocol, acp.METHOD_B, embed_fn=_embed_factory({probe.sample_id: "zero_faces"}))

    failed = next(r for r in proposed.search_results if r.sample_id == probe.sample_id)
    assert failed.failure_code == "zero_faces"
    assert failed.top_similarity is None
    # Excluded from the FPIR denominator rather than counted as a correct reject.
    coverage = acp.open_set_coverage(proposed)
    assert coverage["scored_non_mated_probes"] == coverage["intended_non_mated_probes"] - 1


# --- Metrics ------------------------------------------------------------------


def _result(role: str, *, top: Optional[float], rank=None, correct=None, fail=None, subgroup="asian_females"):
    return acp.OpenSetSearchResult(
        sample_id=f"s{abs(hash((role, top, rank, fail))) % 10**8:08d}",
        identity_hash="h" * 32,
        subgroup=subgroup,
        role=role,
        failure_code=fail,
        top_similarity=top,
        top_identity_hash="c" * 32,
        correct_rank=rank,
        correct_similarity=correct,
    )


def test_fpir_is_computed_from_non_mated_searches_only() -> None:
    results = [
        _result("non_mated_probe", top=0.9),   # above threshold -> false positive
        _result("non_mated_probe", top=0.1),
        _result("non_mated_probe", top=0.2),
        _result("non_mated_probe", top=0.3),
        _result("mated_probe", top=0.95, rank=1, correct=0.95),
    ]
    rates = acp.open_set_rates_at_threshold(results, 0.5)
    assert rates["fpir"] == pytest.approx(0.25)
    assert rates["scored_non_mated_probes"] == 4
    assert rates["false_reviews_per_1000_non_mated"] == pytest.approx(250.0)


def test_fnir_uses_the_correct_mate_and_its_rank() -> None:
    results = [
        _result("mated_probe", top=0.9, rank=1, correct=0.9),   # found at rank 1
        _result("mated_probe", top=0.9, rank=4, correct=0.7),   # found at rank 4
        _result("mated_probe", top=0.9, rank=9, correct=0.6),   # outside rank 5
        _result("mated_probe", top=0.9, rank=1, correct=0.2),   # rank 1 but below threshold
    ]
    rates = acp.open_set_rates_at_threshold(results, 0.5)
    assert rates["tpir_rank1"] == pytest.approx(0.25)
    assert rates["tpir_rank5"] == pytest.approx(0.5)
    assert rates["fnir_rank1"] == pytest.approx(0.75)
    assert rates["fnir_rank5"] == pytest.approx(0.5)
    # CMC ignores the threshold and asks only about rank.
    assert rates["cmc_rank1"] == pytest.approx(0.5)
    assert rates["cmc_rank5"] == pytest.approx(0.75)


def test_tpir_is_one_minus_fnir_at_the_same_rank() -> None:
    results = [
        _result("mated_probe", top=0.9, rank=1, correct=0.9),
        _result("mated_probe", top=0.9, rank=7, correct=0.9),
    ]
    rates = acp.open_set_rates_at_threshold(results, 0.5)
    assert rates["tpir_rank1"] + rates["fnir_rank1"] == pytest.approx(1.0)
    assert rates["tpir_rank5"] + rates["fnir_rank5"] == pytest.approx(1.0)


def test_a_failed_search_is_excluded_from_both_denominators() -> None:
    results = [
        _result("mated_probe", top=None, fail="zero_faces"),
        _result("mated_probe", top=0.9, rank=1, correct=0.9),
        _result("non_mated_probe", top=None, fail="gallery_reference_unavailable"),
        _result("non_mated_probe", top=0.1),
    ]
    rates = acp.open_set_rates_at_threshold(results, 0.5)
    assert rates["scored_mated_probes"] == 1
    assert rates["scored_non_mated_probes"] == 1
    assert rates["tpir_rank1"] == pytest.approx(1.0)
    assert rates["fpir"] == pytest.approx(0.0)


def test_undefined_rates_stay_nan_rather_than_zero() -> None:
    rates = acp.open_set_rates_at_threshold([], 0.5)
    assert math.isnan(rates["fpir"])
    assert math.isnan(rates["tpir_rank1"])


# --- Threshold development and freezing ---------------------------------------


def _development_results(count: int = 200) -> List[acp.OpenSetSearchResult]:
    rng = np.random.default_rng(EXPECTED_SEED)
    rows: List[acp.OpenSetSearchResult] = []
    for _ in range(count):
        rows.append(_result("non_mated_probe", top=float(rng.uniform(0.0, 0.5))))
    for _ in range(count):
        score = float(rng.uniform(0.6, 0.95))
        rows.append(_result("mated_probe", top=score, rank=1, correct=score))
    return rows


def test_the_targets_and_primary_operating_point_are_pinned() -> None:
    assert acp.FPIR_TARGETS == EXPECTED_FPIR_TARGETS
    assert acp.PRIMARY_FPIR_TARGET == EXPECTED_PRIMARY_FPIR_TARGET


def test_selection_respects_the_target_fpir_ceiling() -> None:
    development = _development_results()
    for target in EXPECTED_FPIR_TARGETS:
        chosen = acp.select_open_set_threshold(development, target_fpir=target)
        assert chosen["development_fpir"] <= target + 1e-12


def test_selection_follows_the_declared_tie_rule() -> None:
    """With several thresholds admissible and equal TPIR@1, the rule must pick
    the one with the lower FPIR, then the higher threshold."""
    rows = [
        _result("non_mated_probe", top=0.10),
        _result("non_mated_probe", top=0.20),
        _result("mated_probe", top=0.90, rank=1, correct=0.90),
        _result("mated_probe", top=0.95, rank=1, correct=0.95),
    ]
    chosen = acp.select_open_set_threshold(rows, target_fpir=0.0)
    # Every mated probe still found, no non-mated above threshold.
    assert chosen["development_fpir"] == pytest.approx(0.0)
    assert chosen["development_tpir_rank1"] == pytest.approx(1.0)
    assert chosen["threshold"] <= 0.90
    assert "highest development TPIR at rank 1" in chosen["selection_rule"]


def test_an_unreachable_target_is_refused_rather_than_approximated() -> None:
    rows = [
        _result("non_mated_probe", top=0.99),
        _result("mated_probe", top=0.99, rank=1, correct=0.99),
    ]
    # Sentinels make FPIR 0 reachable, so ask for something impossible instead.
    with pytest.raises(acp.OpenSetPolicyError):
        acp.select_open_set_threshold(rows, target_fpir=-0.5)


def test_test_scores_cannot_influence_threshold_selection() -> None:
    """Selection sees only what it is given; adding held-out rows with wildly
    different scores must not move the chosen threshold."""
    development = _development_results()
    baseline = acp.select_open_set_threshold(development, target_fpir=0.01)
    contaminant = [_result("non_mated_probe", top=0.999) for _ in range(500)]
    assert acp.select_open_set_threshold(development, target_fpir=0.01) == baseline
    # And the contaminated set genuinely would have changed it.
    assert (
        acp.select_open_set_threshold(development + contaminant, target_fpir=0.01)["threshold"]
        != baseline["threshold"]
    )


@pytest.mark.parametrize("status", ["open_set_development", "open_set_tested", None, "frozen"])
def test_a_non_frozen_policy_is_refused_for_held_out_evaluation(status) -> None:
    payload = {
        "status": status,
        "operating_points": {str(EXPECTED_PRIMARY_FPIR_TARGET): {"threshold": 0.5}},
    }
    with pytest.raises(acp.OpenSetPolicyError):
        acp.require_frozen_open_set_policy(payload)


def test_a_frozen_policy_is_accepted_and_returns_the_primary_threshold() -> None:
    payload = {
        "status": EXPECTED_FROZEN_STATUS,
        "operating_points": {str(EXPECTED_PRIMARY_FPIR_TARGET): {"threshold": 0.4242}},
    }
    assert acp.require_frozen_open_set_policy(payload) == pytest.approx(0.4242)


def test_a_frozen_policy_without_the_primary_target_is_refused() -> None:
    with pytest.raises(acp.OpenSetPolicyError):
        acp.require_frozen_open_set_policy(
            {"status": EXPECTED_FROZEN_STATUS, "operating_points": {"0.5": {"threshold": 0.4}}}
        )


# --- Cluster bootstrap --------------------------------------------------------


def _bootstrap_fixture() -> List[acp.OpenSetSearchResult]:
    rows: List[acp.OpenSetSearchResult] = []
    for index in range(40):
        subgroup = EXPECTED_SUBGROUPS[index % len(EXPECTED_SUBGROUPS)]
        identity = f"{index:032d}"
        rows.append(
            acp.OpenSetSearchResult(
                sample_id=f"m{index:031d}", identity_hash=identity, subgroup=subgroup,
                role="mated_probe", top_similarity=0.8, correct_rank=1, correct_similarity=0.8,
            )
        )
        rows.append(
            acp.OpenSetSearchResult(
                sample_id=f"n{index:031d}", identity_hash=f"x{index:031d}", subgroup=subgroup,
                role="non_mated_probe", top_similarity=0.2,
            )
        )
    return rows


def test_cluster_bootstrap_intervals_are_deterministic() -> None:
    rows = _bootstrap_fixture()
    first = acp.cluster_bootstrap_intervals(rows, threshold=0.5, replicates=50, seed=EXPECTED_SEED)
    second = acp.cluster_bootstrap_intervals(rows, threshold=0.5, replicates=50, seed=EXPECTED_SEED)
    assert first == second


def test_a_different_bootstrap_seed_changes_the_interval() -> None:
    """Needs genuine between-identity variance: if every identity behaves
    identically, every resample is identical and the interval is degenerate
    whatever the seed."""
    rng = np.random.default_rng(EXPECTED_SEED)
    rows: List[acp.OpenSetSearchResult] = []
    for index in range(40):
        subgroup = EXPECTED_SUBGROUPS[index % len(EXPECTED_SUBGROUPS)]
        score = float(rng.uniform(0.1, 0.9))
        rows.append(
            acp.OpenSetSearchResult(
                sample_id=f"m{index:031d}", identity_hash=f"{index:032d}", subgroup=subgroup,
                role="mated_probe", top_similarity=score, correct_rank=1,
                correct_similarity=score,
            )
        )
        other = float(rng.uniform(0.1, 0.9))
        rows.append(
            acp.OpenSetSearchResult(
                sample_id=f"n{index:031d}", identity_hash=f"x{index:031d}", subgroup=subgroup,
                role="non_mated_probe", top_similarity=other,
            )
        )
    first = acp.cluster_bootstrap_intervals(rows, threshold=0.5, replicates=50, seed=EXPECTED_SEED)
    other_seed = acp.cluster_bootstrap_intervals(
        rows, threshold=0.5, replicates=50, seed=EXPECTED_SEED + 1
    )
    assert first != other_seed


def test_the_bootstrap_reports_how_many_replicates_were_valid() -> None:
    intervals = acp.cluster_bootstrap_intervals(
        _bootstrap_fixture(), threshold=0.5, replicates=25, seed=EXPECTED_SEED
    )
    for name in ("fpir", "tpir_rank1", "extraction_coverage"):
        assert intervals[name]["requested_replicates"] == 25
        assert 0 <= intervals[name]["valid_replicates"] <= 25
    assert acp.BOOTSTRAP_REPLICATES == 2000


def test_an_undefined_metric_is_never_replaced_by_zero() -> None:
    """Only non-mated probes: every replicate leaves TPIR undefined, so the
    interval must be empty rather than a confident zero."""
    rows = [
        acp.OpenSetSearchResult(
            sample_id=f"n{i:031d}", identity_hash=f"x{i:031d}",
            subgroup="asian_females", role="non_mated_probe", top_similarity=0.2,
        )
        for i in range(8)
    ]
    intervals = acp.cluster_bootstrap_intervals(rows, threshold=0.5, replicates=10, seed=EXPECTED_SEED)
    assert intervals["tpir_rank1"]["valid_replicates"] == 0
    assert math.isnan(intervals["tpir_rank1"]["lower_95"])


# --- Subgroup reporting -------------------------------------------------------


def test_subgroup_metrics_are_reported_per_subgroup() -> None:
    per_subgroup = acp.subgroup_open_set_metrics(_bootstrap_fixture(), threshold=0.5)
    assert set(per_subgroup) == set(EXPECTED_SUBGROUPS)
    for row in per_subgroup.values():
        assert row["tpir_rank1"] == pytest.approx(1.0)
        assert row["fpir"] == pytest.approx(0.0)


def test_the_ratio_is_withheld_when_a_subgroup_records_zero_fpir() -> None:
    summary = acp.subgroup_disparity_summary(
        acp.subgroup_open_set_metrics(_bootstrap_fixture(), threshold=0.5)
    )
    assert summary["max_to_min_fpir_ratio"] is None
    assert "Undefined" in summary["max_to_min_fpir_ratio_note"]
    assert summary["absolute_fpir_range"] == pytest.approx(0.0)


def test_the_ratio_is_reported_when_the_denominator_is_non_zero() -> None:
    rows = _bootstrap_fixture()
    # Raise one subgroup's non-mated score above threshold to create spread.
    rows = [
        acp.OpenSetSearchResult(
            sample_id=r.sample_id, identity_hash=r.identity_hash, subgroup=r.subgroup,
            role=r.role,
            top_similarity=0.9 if r.role == "non_mated_probe" else r.top_similarity,
            correct_rank=r.correct_rank, correct_similarity=r.correct_similarity,
        )
        if r.subgroup == "asian_females" else r
        for r in rows
    ]
    # Give every other subgroup a non-zero but smaller FPIR.
    rows = [
        acp.OpenSetSearchResult(
            sample_id=r.sample_id, identity_hash=r.identity_hash, subgroup=r.subgroup,
            role=r.role,
            top_similarity=0.6 if (r.role == "non_mated_probe" and r.subgroup != "asian_females")
            else r.top_similarity,
            correct_rank=r.correct_rank, correct_similarity=r.correct_similarity,
        )
        for r in rows
    ]
    summary = acp.subgroup_disparity_summary(
        acp.subgroup_open_set_metrics(rows, threshold=0.5)
    )
    assert summary["max_to_min_fpir_ratio"] == pytest.approx(1.0)


# --- Success criteria ---------------------------------------------------------


def test_success_criteria_are_declared_before_any_test_result() -> None:
    assert acp.OPEN_SET_SUCCESS_CRITERIA["held_out_fpir_max"] == 0.01
    assert acp.OPEN_SET_SUCCESS_CRITERIA["target_fpir"] == EXPECTED_PRIMARY_FPIR_TARGET
    assert acp.OPEN_SET_SUCCESS_CRITERIA["tpir_rank1_min"] == 0.90
    assert acp.OPEN_SET_SUCCESS_CRITERIA["tpir_rank5_min"] == 0.95
    assert acp.OPEN_SET_SUCCESS_CRITERIA["gallery_enrolment_coverage_min"] == 0.90
    assert acp.OPEN_SET_SUCCESS_CRITERIA["probe_extraction_coverage_min"] == 0.90


def test_an_unmet_criterion_is_reported_as_not_achieved() -> None:
    verdicts = acp.evaluate_open_set_success_criteria(
        {"gallery_enrolment_coverage": 0.5, "mated_extraction_failure_rate": 0.5,
         "non_mated_extraction_failure_rate": 0.5},
        {"fpir": 0.4, "tpir_rank1": 0.2, "tpir_rank5": 0.3},
    )
    assert verdicts["fpir_at_or_below_1_percent"]["outcome"] == "not_achieved"
    assert verdicts["tpir_rank1_at_least_90_percent"]["outcome"] == "not_achieved"
    assert verdicts["gallery_enrolment_coverage_at_least_90_percent"]["outcome"] == "not_achieved"


def test_an_undefined_metric_is_not_measurable_rather_than_a_pass() -> None:
    verdicts = acp.evaluate_open_set_success_criteria(
        {"gallery_enrolment_coverage": float("nan"), "mated_extraction_failure_rate": float("nan"),
         "non_mated_extraction_failure_rate": float("nan")},
        {"fpir": float("nan"), "tpir_rank1": float("nan"), "tpir_rank5": float("nan")},
    )
    for name, verdict in verdicts.items():
        if name == "criteria_declared_before_test":
            continue
        assert verdict["outcome"] == "not_measurable"


# --- Optional extensions ------------------------------------------------------


def test_absent_optional_datasets_are_reported_as_skipped_not_fabricated() -> None:
    lines = acp.report_optional_dataset_status()
    joined = "\n".join(lines)
    assert "AgeDB" in joined
    assert "ArcFace" in joined or "buffalo_l" in joined
    assert "NOT RUN" in joined or "SKIPPED" in joined


def test_the_open_set_summary_reports_the_absence_of_results(tmp_path: Path) -> None:
    text = acp.render_open_set_summary(tmp_path)
    assert "No open-set results found" in text
    assert "FACE_BFW_ROOT" in text


# --- Privacy scanning of the new artefacts ------------------------------------


def test_the_identifier_key_is_detected_if_it_reaches_an_artifact(tmp_path: Path) -> None:
    key = "a" * 63 + "b"
    with acp.temporary_id_hmac_key(key):
        (tmp_path / "leaky.json").write_text(
            '{"note": "' + bytes.fromhex(key).hex() + '"}', encoding="utf-8"
        )
        with pytest.raises(acp.PrivacyLeakError) as raised:
            acp.assert_no_identifier_key_leak(tmp_path)
    message = str(raised.value)
    assert "leaky.json" in message
    # Reporting the key itself would be a second leak.
    assert key not in message
    assert bytes.fromhex(key).hex() not in message


def test_a_clean_artifact_directory_passes_the_key_scan(tmp_path: Path) -> None:
    with acp.temporary_id_hmac_key("a" * 63 + "b"):
        (tmp_path / "clean.json").write_text('{"fpir": 0.003}', encoding="utf-8")
        acp.assert_no_identifier_key_leak(tmp_path)


def test_optional_dataset_roots_join_the_forbidden_substring_set() -> None:
    for variable in ("FACE_BFW_ROOT", "FACE_BFW_METADATA_ROOT", "FACE_AGEDB_ROOT"):
        assert variable in acp.OPTIONAL_ENVIRONMENT_VARIABLES
    substrings = acp.default_forbidden_path_substrings(
        env={"FACE_BFW_ROOT": "/private/research/bfw"}
    )
    assert "/private/research/bfw" in substrings


def test_the_open_set_summary_never_shows_a_rate_without_coverage(tmp_path: Path) -> None:
    """FPIR is only interpretable next to the coverage it was measured over."""
    acp.write_json_artifact(
        tmp_path / "bfw_open_set_test_metrics.json",
        {
            "artifact_type": "bfw_open_set_test_metrics",
            "status": "open_set_tested",
            "operating_threshold": 0.62,
            "primary_fpir_target": EXPECTED_PRIMARY_FPIR_TARGET,
            "methods": {
                acp.METHOD_A: {"rates": {"fpir": 0.52, "tpir_rank1": 0.93}},
                acp.METHOD_B: {
                    "primary_operating_point": {
                        "fpir": 0.003, "tpir_rank1": 0.91, "tpir_rank5": 0.96,
                        "false_reviews_per_1000_non_mated": 3.0,
                    },
                    "coverage": {
                        "gallery_enrolment_coverage": 0.94,
                        "mated_extraction_failure_rate": 0.05,
                        "non_mated_extraction_failure_rate": 0.06,
                    },
                },
            },
            "success_criteria": {
                "criteria_declared_before_test": True,
                "fpir_at_or_below_1_percent": {"outcome": "achieved", "actual": 0.003, "target": 0.01},
            },
        },
    )
    acp.write_json_artifact(
        tmp_path / "open_set_confidence_intervals.json",
        {"intervals": {"fpir": {"lower_95": 0.001, "upper_95": 0.006},
                       "tpir_rank1": {"lower_95": 0.88, "upper_95": 0.94}}},
    )
    summary = acp.render_open_set_summary(tmp_path)
    assert "FPIR: 0.30%" in summary
    assert "Gallery enrolment coverage: 94.00%" in summary
    assert "Mated extraction failure: 5.00%" in summary
    assert "LIMITATION" in summary
    assert summary.index("FPIR") < summary.index("Gallery enrolment coverage")
    # The control must be labelled as such wherever it appears.
    assert "not a valid open-set operating threshold" in summary


def test_enrolment_image_choice_does_not_depend_on_the_identifier_key(tmp_path: Path) -> None:
    """Regression: enrolment candidates were once ordered by ``sample_id``, which
    is an HMAC under the secret key. That made the single-image method's chosen
    reference key-dependent, so its published metrics could not be reproduced by
    anyone holding a different key."""
    root, metadata = _make_bfw(tmp_path)
    chosen = []
    for key in ("a" * 63 + "b", "c" * 63 + "d"):
        with acp.temporary_id_hmac_key(key):
            protocol = acp.build_open_set_protocol(
                acp.load_bfw_dataset(root, metadata), seed=EXPECTED_SEED
            )
            run = acp.run_open_set_method(
                protocol,
                partition="development",
                method=acp.METHOD_A,
                detector=None,  # type: ignore[arg-type]
                embedder=None,  # type: ignore[arg-type]
                embed_fn=_recording_embed(seen := {}),
            )
            assert run.gallery_size > 0
            # Which *file* was enrolled, keyed by the private identity.
            chosen.append({identity: names for identity, names in sorted(seen.items())})
    assert chosen[0] == chosen[1]


def _recording_embed(seen: dict):
    """Stub that records the filename enrolled for each private identity."""

    def _embed(entry, detector, embedder):
        if entry.role == "gallery_enrolment":
            seen.setdefault(entry.identity, []).append(entry.image_path.name)
        seed = int(entry.identity_hash[:8], 16)
        vector = np.random.default_rng(seed).normal(size=16)
        return vector / float(np.linalg.norm(vector)), None

    return _embed


def test_top1_and_top5_latencies_are_measured_separately(bfw: acp.BfwDataset) -> None:
    """Both must be real measurements. The top-5 figure previously timed a list
    slice, which is free once the gallery is already ordered."""
    protocol = acp.build_open_set_protocol(bfw, seed=EXPECTED_SEED)
    run = _run(protocol, acp.METHOD_B)
    scored = [r for r in run.search_results if r.failure_code is None]
    assert scored
    for row in scored:
        assert row.top1_time_seconds is not None and row.top1_time_seconds > 0
        assert row.top5_time_seconds is not None
        # Retrieving five candidates cannot cost less than retrieving one.
        assert row.top5_time_seconds >= row.top1_time_seconds


# --- Optional pipeline comparison (Phase 11) ----------------------------------


def test_the_primary_pipeline_describes_the_whole_chain() -> None:
    description = acp.primary_pipeline_description()
    payload = description.as_dict()
    assert payload["detector_name"].startswith("OpenCV YuNet")
    assert payload["embedding_model_name"].startswith("OpenCV SFace")
    assert payload["embedding_dimensions"] == 128
    assert set(payload["model_sha256"]) == {"yunet", "sface"}
    # A difference between pipelines must never be attributed to the embedding
    # alone, because the detector and preprocessing differ too.
    assert "cannot be attributed to the embedding model alone" in payload["comparison_scope"]


def test_an_unconfigured_comparator_is_reported_not_run(tmp_path: Path) -> None:
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=None,
    )
    status = acp.pipeline_comparison_status(config)
    assert status["comparison_run"] is False
    assert status["substitute_model_used"] is False
    assert "non-commercial" in status["licence_note"]


def test_a_configured_comparator_without_pinned_digests_is_refused(tmp_path: Path) -> None:
    """Digests are pinned in source and never accepted as arguments, so an
    unverified weight file cannot reach a reportable evaluation."""
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=tmp_path,
    )
    with pytest.raises(acp.PipelineUnavailableError) as raised:
        acp.arcface_pipeline_description(config)
    assert "pinned" in str(raised.value).lower()
    assert acp.ARCFACE_DETECTOR_SHA256 is None
    assert acp.ARCFACE_RECOGNITION_SHA256 is None


def test_the_comparison_csv_records_absence_rather_than_omitting_it(tmp_path: Path) -> None:
    config = acp.EnvironmentConfig(
        data_root=None, protocol_root=None, model_root=None,
        cplfw_raw_root=None, cache_root=None, arcface_model_root=None,
    )
    out = tmp_path / "pretrained_pipeline_comparison.csv"
    acp.write_pipeline_comparison_csv(
        out, primary=acp.primary_pipeline_description(), config=config
    )
    rows = list(csv.DictReader(open(out)))
    assert len(rows) == 2
    assert rows[0]["evaluated"] == "yes"
    assert rows[1]["evaluated"] == "no" and rows[1]["note"]


# --- AgeDB transfer (Phase 10) ------------------------------------------------


def _make_agedb(tmp_path: Path, *, identities: int = 30, per_identity: int = 9) -> Path:
    """Official AgeDB layout: a flat directory of <index>_<name>_<age>_<gender>.jpg."""
    root = tmp_path / "AgeDB"
    root.mkdir(parents=True, exist_ok=True)
    index = 0
    for person in range(identities):
        for step in range(per_identity):
            index += 1
            age = 20 + step * 5
            (root / f"{index}_Subject{person:03d}_{age}_f.jpg").write_bytes(b"not-a-real-image")
    return root


def test_the_agedb_adapter_reads_the_official_naming(tmp_path: Path) -> None:
    ds = acp.load_agedb_dataset(_make_agedb(tmp_path))
    grouped = ds.by_identity()
    assert len(grouped) == 30
    assert all(len(v) == 9 for v in grouped.values())
    # Ordered by age, so "youngest enrols, oldest probes" is well defined.
    for images in grouped.values():
        assert [i.age for i in images] == sorted(i.age for i in images)


def test_a_file_not_matching_the_official_naming_is_refused(tmp_path: Path) -> None:
    root = _make_agedb(tmp_path)
    (root / "stray_photo.jpg").write_bytes(b"x")
    with pytest.raises(acp.AgeDbDatasetError) as raised:
        acp.load_agedb_dataset(root)
    message = str(raised.value)
    assert "official" in message
    # The offending filename must not be echoed: AgeDB names are real people.
    assert "stray_photo" not in message


def test_an_implausible_age_is_refused(tmp_path: Path) -> None:
    root = _make_agedb(tmp_path)
    (root / "9999_Someone_999_m.jpg").write_bytes(b"x")
    with pytest.raises(acp.AgeDbDatasetError) as raised:
        acp.load_agedb_dataset(root)
    assert "implausible age" in str(raised.value)


def test_agedb_identifiers_never_carry_the_subject_name(tmp_path: Path) -> None:
    ds = acp.load_agedb_dataset(_make_agedb(tmp_path))
    for image in ds.images:
        assert "Subject" not in image.sample_id
        assert "Subject" not in image.identity_hash
        assert len(image.identity_hash) == 32


def test_the_transfer_protocol_enrols_young_and_probes_old(tmp_path: Path) -> None:
    ds = acp.load_agedb_dataset(_make_agedb(tmp_path))
    protocol = acp.build_agedb_transfer_protocol(ds, gallery_size=10, seed=EXPECTED_SEED)
    age_of = {i.sample_id: i.age for i in ds.images}
    enrolled = [age_of[e.sample_id] for e in protocol.entries if e.role == "gallery_enrolment"]
    probes = [age_of[e.sample_id] for e in protocol.entries if e.role == "mated_probe"]
    assert max(enrolled) < max(probes)
    # Gallery and non-mated identities must not overlap.
    assert protocol.identities("test", "gallery_enrolment").isdisjoint(
        protocol.identities("test", "non_mated_probe")
    )


def test_the_transfer_protocol_reports_the_age_gap(tmp_path: Path) -> None:
    ds = acp.load_agedb_dataset(_make_agedb(tmp_path))
    protocol = acp.build_agedb_transfer_protocol(ds, gallery_size=10, seed=EXPECTED_SEED)
    gaps = acp.agedb_age_gap_distribution(ds, protocol)
    assert gaps["mated_probes_with_age_gap"] > 0
    assert gaps["age_gap_years_max"] >= gaps["age_gap_years_min"]
    assert gaps["age_gap_years_max"] > 0


def test_too_few_eligible_identities_is_an_explicit_error(tmp_path: Path) -> None:
    ds = acp.load_agedb_dataset(_make_agedb(tmp_path, identities=5))
    with pytest.raises(acp.AgeDbDatasetError) as raised:
        acp.build_agedb_transfer_protocol(ds, gallery_size=200, seed=EXPECTED_SEED)
    assert "fewer than the requested" in str(raised.value)


def test_agedb_transfer_is_skipped_not_fabricated_when_unconfigured(tmp_path: Path) -> None:
    assert acp.EnvironmentConfig.load().agedb_root is None
    assert acp.run_agedb_transfer(output_root=tmp_path) is None
    assert not (tmp_path / "agedb_transfer_metrics.json").exists()
