"""Saved report regeneration preserves measured values, tables and provenance."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import ACP_arden as a


@pytest.fixture()
def saved(tmp_path: Path) -> Path:
    """Copy public aggregate evidence; no image or pretrained model is accessed."""
    root = tmp_path / "aggregate"
    names = ["pipeline_comparison_metrics.json", "lfw_development_metrics.json", "lfw_final_metrics.json",
             "cplfw_metrics.json", "calibrated_threshold.json", "duplicate_gallery_metrics_v2.json"]
    classifier_files = ("ml_review_test_metrics.json", "ml_review_confidence_intervals.json",
                        "ml_review_subgroup_metrics.json", "ml_review_model.json")
    names += list(classifier_files)
    names += [f"{a.ARCFACE_REVIEW_DIRNAME}/{name}" for name in classifier_files]
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((a.AGGREGATE_ROOT / name).read_bytes())
    model_path = root / "ml_review_model.json"
    model = json.loads(model_path.read_text())
    model["model"]["coefficients"][0] = 17.1234
    model_path.write_text(json.dumps(model))
    comparison_path = root / "pipeline_comparison_metrics.json"
    comparison = json.loads(comparison_path.read_text())
    comparison["created_at"] = "2000-01-02T03:04:05+00:00"
    comparison_path.write_text(json.dumps(comparison))
    (root / "run_manifest.json").write_text('{"original_execution_environment":"preserve"}\n')
    (root / "metrics_summary.csv").write_text("original,measurement\n1,2\n")
    return root


def prohibit_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args, **kwargs):
        pytest.fail("Refreshing Markdown must not refit, reselect or infer")
    for name in ("load_models", "load_arcface_pipeline", "load_image_bgr", "fit_review_classifier",
                 "run_pipeline_comparison", "run_ml_review_experiment", "select_open_set_threshold",
                 "generate_threshold_sensitivity", "write_aggregate_reports"):
        monkeypatch.setattr(a, name, forbidden)


def test_saved_refresh_updates_all_three_reports_from_their_own_sources(saved: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prohibit_inference(monkeypatch)
    before = {path: path.read_bytes() for path in saved.rglob("*") if path.suffix in (".json", ".csv")}
    paths = a.refresh_saved_evaluation_reports(saved)
    assert {path.relative_to(saved).as_posix() for path in paths} == {
        "PRETRAINED_PIPELINE_COMPARISON_REPORT.md", "ML_REVIEW_EVALUATION_REPORT.md",
        f"{a.ARCFACE_REVIEW_DIRNAME}/ML_REVIEW_EVALUATION_REPORT.md"}
    assert all(path.read_bytes() == content for path, content in before.items())
    comparison = (saved / "PRETRAINED_PIPELINE_COMPARISON_REPORT.md").read_text()
    assert "2000-01-02T03:04:05+00:00" in comparison
    assert "Experiment 8" in comparison and "## Held-out results" in comparison
    baseline = (saved / "ML_REVIEW_EVALUATION_REPORT.md").read_text()
    arcface = (saved / a.ARCFACE_REVIEW_DIRNAME / "ML_REVIEW_EVALUATION_REPORT.md").read_text()
    assert "Experiment 7" in baseline and "+17.1234" in baseline
    assert "Experiment 11" in arcface and "+17.1234" not in arcface


def test_refresh_action_preserves_existing_json_csv_and_forwards_read_only_table_mode(saved: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    prohibit_inference(monkeypatch)
    before = {path: path.read_bytes() for path in saved.rglob("*") if path.suffix in (".json", ".csv")}
    calls = []

    def figures(*, aggregate_root, write_data_artifacts=True):
        calls.append((aggregate_root, write_data_artifacts))
        return []

    monkeypatch.setattr(a, "generate_figures", figures)
    assert a.action_refresh_reports(saved) == 0
    assert calls == [(saved, False)]
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (saved / "FINAL_EVALUATION_REPORT.md").is_file()
    assert (saved / "research_assessment.json").is_file()  # Explicitly derived interpretation.
    assert (saved / "RESEARCH_REPORT.md").is_file()
    assert (saved / a.ARCFACE_REVIEW_DIRNAME / "ML_REVIEW_EVALUATION_REPORT.md").is_file()


def test_incomplete_saved_classifier_sources_do_not_create_a_report(saved: Path) -> None:
    (saved / a.ARCFACE_REVIEW_DIRNAME / "ml_review_model.json").unlink()
    paths = a.refresh_saved_evaluation_reports(saved)
    assert len(paths) == 2
    assert not (saved / a.ARCFACE_REVIEW_DIRNAME / "ML_REVIEW_EVALUATION_REPORT.md").exists()


def test_invalid_numeric_classifier_model_does_not_partially_publish_reports(saved: Path) -> None:
    path = saved / "ml_review_model.json"
    model = json.loads(path.read_text())
    model["model"]["scaler_scale"][0] = 0
    path.write_text(json.dumps(model))
    with pytest.raises(a.MlReviewError, match="scaling"):
        a.refresh_saved_evaluation_reports(saved)
    assert not list(saved.rglob("*.md"))


def test_empty_refresh_has_no_fabricated_evaluation_report(tmp_path: Path) -> None:
    assert a.refresh_saved_evaluation_reports(tmp_path) == []
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("write_data", [False, True])
def test_figure_refresh_table_mode_is_forwarded_and_guards_layer_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_data: bool) -> None:
    root, figures = tmp_path / "aggregate", tmp_path / "figures"
    root.mkdir()
    calls = []

    def detector(active_root, active_figures, *, write_data_artifacts=True):
        calls.append(("detector", active_root, active_figures, write_data_artifacts))
        return None

    def layers(active_root):
        calls.append(("layers", active_root))

    monkeypatch.setattr(a, "write_detector_ablation_outputs", detector)
    monkeypatch.setattr(a, "write_implementation_layer_artefacts", layers)
    a.generate_figures(aggregate_root=root, figures_root=figures, write_data_artifacts=write_data)
    assert ("detector", root, figures, write_data) in calls
    assert (("layers", root) in calls) is write_data
    assert not list(root.glob("*.json")) and not list(root.glob("*.csv"))


@pytest.mark.parametrize("write_data", [False, True])
def test_detector_figure_refresh_preserves_or_rebuilds_its_summary_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, write_data: bool) -> None:
    root = tmp_path / "aggregate"
    directory = root / a.DETECTOR_ABLATION_DIR
    directory.mkdir(parents=True)
    name = "detector_settings_metrics.json"
    (directory / name).write_bytes((a.AGGREGATE_ROOT / a.DETECTOR_ABLATION_DIR / name).read_bytes())
    summary = directory / "detector_settings_summary.csv"
    summary.write_text("original measured table\n")
    axes = np.empty((2, 2), dtype=object)
    for index in range(4):
        axes.flat[index] = MagicMock()
    for axis in axes.flat:
        axis.get_legend_handles_labels.return_value = ([], [])
    backend = MagicMock()
    backend.subplots.return_value = (MagicMock(), axes)
    monkeypatch.setattr(a, "_figure_backend", lambda: backend)
    monkeypatch.setattr(a, "_save_figure", lambda *args: None)
    a.write_detector_ablation_outputs(root, tmp_path / "figures", write_data_artifacts=write_data)
    if write_data:
        assert "cohort,setting,intended_images" in summary.read_text()
    else:
        assert summary.read_text() == "original measured table\n"
