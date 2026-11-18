"""Controlled detector interventions, pairing and retained failures."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import ACP_arden as a


def settings():
    return [a.DetectorAblationSetting('base', 'YuNet', .9, None, 'baseline'),
            a.DetectorAblationSetting('changed', 'YuNet', .8, None, 'confidence'),
            a.DetectorAblationSetting('repeat', 'YuNet', .9, None, 'repeat_control')]


def test_planned_variants_change_only_one_factor():
    a.validate_detector_ablation_settings(a.detector_ablation_settings())
    bad = settings()
    bad[1] = replace(bad[1], canvas=320)
    with pytest.raises(a.ProtocolError, match='exactly'):
        a.validate_detector_ablation_settings(bad)


def test_equal_setting_repeat_must_actually_equal_baseline():
    bad = settings()
    bad[2] = replace(bad[2], confidence=.6)
    with pytest.raises(a.ProtocolError):
        a.validate_detector_ablation_settings(bad)


def test_cohort_is_balanced_deduplicated_and_order_independent():
    pool = [a.DetectorAblationSample(Path(f'{identity}/{image}.png'), identity, group)
            for group in ('a', 'b') for identity in (group+'1', group+'2', group+'3')
            for image in range(3)]
    chosen = a.select_detector_ablation_cohort(pool+pool, identities=4, images_per_identity=2)
    assert chosen == a.select_detector_ablation_cohort(list(reversed(pool)), identities=4, images_per_identity=2)
    assert len(chosen) == len({s.image_path for s in chosen}) == 8
    assert {g:sum(s.stratum==g for s in chosen) for g in ('a','b')} == {'a':4,'b':4}
    assert all(sum(s.identity==i for s in chosen)==2 for i in {s.identity for s in chosen})


def test_undersized_cohort_is_not_silently_shrunk():
    with pytest.raises(a.ProtocolError, match='Insufficient'):
        a.select_detector_ablation_cohort([], identities=2)


def test_canvas_preserves_aspect_and_pads_instead_of_stretching():
    image = np.full((10,20,3), 255, dtype=np.uint8)
    resized = a.detector_canvas(image, 40)
    assert resized.shape == (40,40,3)
    assert np.all(resized[:20] == 255)
    assert np.all(resized[20:] == 0)
    assert np.all(image == 255)


def make_rows(outcomes):
    return [{'sample_id':str(i), 'identity_hash':str(i//2), 'stratum':'all',
             'outcome':outcome, 'face_count':None,
             'detector_ms':None if outcome=='load_failure' else 1.0}
            for i,outcome in enumerate(outcomes)]


def test_paired_effect_keeps_failed_images_and_uses_identity_clusters():
    base = make_rows(['zero_faces','zero_faces','one_face','load_failure'])
    changed = make_rows(['one_face','one_face','one_face','load_failure'])
    result = a.summarise_detector_ablation({'base':base,'changed':changed,'repeat':base}, settings(), replicates=2000)
    assert result['independent_identities'] == 2
    assert result['intended_images'] == 4
    assert result['settings']['base']['one_face_coverage'] == .25
    assert result['settings']['changed']['unprocessed_per_1000_intended'] == 250
    delta, repeat = result['paired_comparisons']
    assert delta['coverage_change'] == .5
    # Both changes belong to one identity. Sampling images independently
    # would underestimate this uncertainty and fail to retain that dependence.
    assert (delta['lower_95'],delta['upper_95']) == (0,1)
    assert delta['transitions']['zero_faces->one_face'] == 2
    assert delta['transitions']['load_failure->load_failure'] == 1
    assert sum(delta['transitions'].values()) == 4
    assert repeat['coverage_change'] == repeat['lower_95'] == repeat['upper_95'] == 0


def test_pairing_rejects_different_images_or_identities():
    base = make_rows(['zero_faces','one_face','one_face','one_face'])
    other = [dict(r) for r in base]
    other[0]['identity_hash'] = 'different'
    with pytest.raises(a.ProtocolError, match='same ordered'):
        a.summarise_detector_ablation({'base':base,'changed':other,'repeat':base}, settings())


def test_load_failures_are_recorded_for_every_setting_without_detector_calls(monkeypatch):
    def fail(path):
        raise a.ImageLoadError('synthetic decode failure')
    monkeypatch.setattr(a, 'load_image_bgr', fail)
    def counter(image):
        pytest.fail('A decode failure must never be sent to a detector')
    samples=[a.DetectorAblationSample(Path('missing.png'),'one','all')]
    rows=a.measure_detector_ablation_cohort(samples, {'a':counter,'b':counter})
    assert all(r[0]['outcome']=='load_failure' and r[0]['detector_ms'] is None for r in rows.values())


def test_detector_exception_is_not_mislabelled_as_no_face(monkeypatch):
    monkeypatch.setattr(a,'load_image_bgr',lambda path:a.LoadedImage(np.zeros((4,4,3),dtype=np.uint8),4,4,path))
    def fail(image):
        raise RuntimeError('backend failure')
    with pytest.raises(RuntimeError,match='backend failure'):
        a.measure_detector_ablation_cohort([a.DetectorAblationSample(Path('x'),'i','g')],{'a':fail})


def test_missing_experiment_is_reported_as_missing(tmp_path):
    assert 'not been evaluated' in a.render_detector_ablation_report(tmp_path)


def test_cli_exposes_run_and_read_only_summary():
    for mode in ('detector-settings','detector-settings-summary'):
        assert a.build_argument_parser().parse_args(['--mode',mode]).mode == mode
