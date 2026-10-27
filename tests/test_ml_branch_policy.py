import numpy as np
import pytest

from signalwatch.ml_branch_policy import apply_branch_policy, apply_to_scored, screen_candidate


def test_classifier_is_immediate_while_single_residual_pulses_remain_pending():
    classifier = np.array([.2, 1.5, .2, .2, .2])
    residual = np.array([.2, .2, 1.5, .2, 1.6])
    result = apply_branch_policy(classifier, residual, np.arange(5.), np.zeros(5, bool))
    np.testing.assert_array_equal(result['decision_score'] > 1., [False, True, False, False, False])
    np.testing.assert_array_equal(result['pending'], [False, False, True, False, True])
    np.testing.assert_array_equal(result['raw_score'], np.maximum(classifier, residual))


def test_elapsed_confirmation_emits_current_observation_without_backfill_and_resets():
    times = np.array([0., .4, 1., 2., 3., 4., 5., 6.])
    residual = np.array([2., 2., 2., 1., 2., 2., 2., 2.])
    gaps = np.array([False, False, False, False, False, True, False, False])
    result = apply_branch_policy(np.full(8, .2), residual, times, gaps)
    np.testing.assert_array_equal(result['decision_score'] > 1., [False, False, True, False, False, False, True, True])
    assert result['raw_episode_start'][6] == 5
    assert result['confirmation_seconds'] == 1.


def test_either_branch_nan_preserves_finite_mask_and_cannot_accumulate_confirmation():
    classifier = np.array([.2, np.nan, .2, .2, .2])
    residual = np.array([2., 2., 2., np.nan, 2.])
    result = apply_branch_policy(classifier, residual, np.arange(5.), np.zeros(5, bool))
    finite = np.isfinite(classifier) & np.isfinite(residual)
    np.testing.assert_array_equal(np.isfinite(result['decision_score']), finite)
    assert not (result['decision_score'][finite] > 1.).any()
    assert not result['pending'][~finite].any()


def test_wrapper_preserves_raw_branches_baselines_prefix_labels_and_input_arrays():
    classifier = np.array([np.nan, .2, 1.5, .2, .2, .2])
    residual = np.array([np.nan, 2., .2, 2., 2., .2])
    raw = np.maximum(classifier, residual)
    base = {'scores': {'learned': raw.copy(), 'robust': np.arange(6.), 'persistence': np.arange(6.) * 2},
            'thresholds': dict.fromkeys(['learned', 'robust', 'persistence'], 1.),
            'normalized_branch_scores': {'classifier': classifier.copy(), 'innovation': residual.copy()}}
    record = {'id': 'one', 'labels': np.zeros(6, bool), 'elapsed': np.arange(6.), 'gap_before': np.zeros(6, bool)}
    result = apply_to_scored(base, record)
    np.testing.assert_array_equal(base['scores']['learned'], raw)
    np.testing.assert_array_equal(result['raw_scores']['learned'], raw)
    np.testing.assert_array_equal(result['scores']['robust'], base['scores']['robust'])
    np.testing.assert_array_equal(result['normalized_branch_scores']['innovation'], residual)
    altered = dict(record, id='not-feature', labels=~record['labels'])
    np.testing.assert_array_equal(result['scores']['learned'], apply_to_scored(base, altered)['scores']['learned'])
    prefix = {key: (value[:4] if isinstance(value, np.ndarray) else value) for key, value in record.items()}
    short = dict(base, scores={key: value[:4] for key, value in base['scores'].items()},
                 normalized_branch_scores={key: value[:4] for key, value in base['normalized_branch_scores'].items()})
    np.testing.assert_array_equal(apply_to_scored(short, prefix)['scores']['learned'], result['scores']['learned'][:4])


def test_single_eligible_residual_event_is_lost_while_classifier_event_survives():
    from signalwatch.evaluation import inspect_recording
    labels = np.array([False, True, False, True, False, False])
    classifier = np.array([.2, .2, .2, 1.2, .2, .2])
    residual = np.array([.2, 1.2, .2, 1.2, .2, .2])
    result = apply_branch_policy(classifier, residual, np.arange(6.), np.zeros(6, bool))
    record = {'id': 'short', 'labels': labels, 'elapsed': np.arange(6.), 'gap_before': np.zeros(6, bool), 'values': np.zeros((6, 8))}
    traces = {'scores': dict.fromkeys(['learned', 'robust', 'persistence'], result['decision_score']),
              'thresholds': dict.fromkeys(['learned', 'robust', 'persistence'], 1.)}
    events = inspect_recording(record, traces)['events']
    truth = {event['id']: event['status'] for event in events if event['status'] in ['detected', 'missed']}
    assert truth == {'short-event-0': 'missed', 'short-event-1': 'detected'}


def test_candidate_screen_uses_project_proxy_and_rejects_count_or_exposure_failures():
    reference = {'false_episodes': 3, 'false_alert_seconds': 37., 'observed_normal_seconds': 9179.}
    metrics = {'detected_events': 27, 'false_episodes': 0, 'false_episodes_per_hour': 0.,
               'false_alert_seconds': 34., 'observed_normal_seconds': 9179.}
    assert screen_candidate(metrics, reference) == []
    assert 'development_recall_floor' in screen_candidate(dict(metrics, detected_events=25), reference)
    assert 'no_strict_false_count_improvement' in screen_candidate(dict(metrics, false_episodes=3), reference)
    assert 'reference_normal_alert_time' in screen_candidate(dict(metrics, false_alert_seconds=38.), reference)
    with pytest.raises(ValueError, match='exposure'):
        screen_candidate(dict(metrics, observed_normal_seconds=9178.), reference)


def test_invalid_vector_shapes_and_timestamps_fail():
    with pytest.raises(ValueError):
        apply_branch_policy(np.zeros(2), np.zeros(3), np.arange(3.), np.zeros(3, bool))
    with pytest.raises(ValueError):
        apply_branch_policy(np.zeros(2), np.zeros(2), np.array([0., 0.]), np.zeros(2, bool))
