import numpy as np
import pytest

from signalwatch.ml_supervised import causal_features as base_features
from signalwatch.ml_supervised_weighted import sample_weights
from signalwatch.ml_trends import (
    causal_features, feature_names, fit_model, classifier_scores, hard_normal_weights,
    choose_hard_normals, inner_splits, select_candidate, screen_candidate,
    mine_hard_normals,
)


def record(key='unit', n=40, elapsed=None):
    elapsed = np.arange(n, dtype=float) if elapsed is None else np.asarray(elapsed, float)
    values = np.column_stack([2. * elapsed + channel for channel in range(8)])
    return {'id': key, 'values': values, 'elapsed': elapsed, 'labels': np.arange(n) >= n // 2,
            'gap_before': np.zeros(n, bool), 'cadence_seconds': 1., 'group': 'unit'}


def test_exact_feature_names_ramps_actual_time_and_constant_channel():
    r = record(elapsed=[0., .5, 2., 3., 5., 8., 9., 12.], n=8)
    r['values'][:, 1] = 7.
    X, indices, reference = causal_features(r)
    names = feature_names()
    assert X.shape == (6, 186) and len(names) == len(set(names)) == 186
    base, old_indices, old_reference = base_features(r)
    np.testing.assert_array_equal(indices, old_indices)
    np.testing.assert_array_equal(X[:, :120], base)
    np.testing.assert_array_equal(reference, old_reference)
    for horizon in [30, 120]:
        assert X[-1, names.index(f'trend_slope{horizon}s:Accelerometer1RMS')] == pytest.approx(2.)
        assert X[-1, names.index(f'trend_direction{horizon}s:Accelerometer1RMS')] == 1.
        assert X[-1, names.index(f'trend_r2{horizon}s:Accelerometer1RMS')] == pytest.approx(1.)
        for statistic in ['slope', 'net', 'direction', 'r2']:
            assert X[-1, names.index(f'trend_{statistic}{horizon}s:Accelerometer2RMS')] == 0.


def test_history_is_past_only_prefix_gap_labels_id_time_origin_and_eligibility():
    r = record(n=50); r['gap_before'][28] = True
    X, indices, _ = causal_features(r)
    assert 28 not in indices and 29 not in indices and 30 in indices
    changed = dict(r, id='different', labels=~r['labels'], elapsed=r['elapsed'] + 100000.)
    np.testing.assert_allclose(causal_features(changed)[0], X, atol=1e-9, rtol=0)
    prefix = {key: (value[:35] if isinstance(value, np.ndarray) else value) for key, value in r.items()}
    pX, pindices, _ = causal_features(prefix)
    np.testing.assert_array_equal(pindices, indices[indices < 35])
    np.testing.assert_allclose(pX, X[indices < 35], atol=0, rtol=0)
    now = 20; altered = dict(r, values=r['values'].copy()); altered['values'][now] += 100.
    changed_X, _, _ = causal_features(altered)
    np.testing.assert_array_equal(changed_X[indices == now, 120:], X[indices == now, 120:])


def test_partial_and_sparse_windows_do_not_add_warmup_or_nonfinite_features():
    r = record(n=5, elapsed=[0., 60., 120., 180., 240.])
    X, indices, _ = causal_features(r); names = feature_names()
    np.testing.assert_array_equal(indices, [2, 3, 4])
    assert np.isfinite(X).all()
    assert np.all(X[:, names.index('trend_slope30s:Accelerometer1RMS')] == 0.)
    assert np.all(X[:, names.index('trend_coverage30s')] == 0.)
    assert np.all(X[:, names.index('trend_slope120s:Accelerometer1RMS')] == pytest.approx(2.))
    short = record(n=3); sX, _, _ = causal_features(short)
    assert sX[0, names.index('trend_coverage120s')] == pytest.approx(1. / 120.)


def test_isolated_current_spike_does_not_invent_a_historical_ramp():
    r = record(n=20); r['values'][:] = 4.; r['values'][15] = 20.
    X, indices, _ = causal_features(r); row = X[indices == 15][0]
    assert np.all(row[120:152] == 0.)
    assert np.all(row[152:184] == 0.)
    assert row[0] == 20.


def test_nonbinary_constant_float_with_irregular_elapsed_has_exact_zero_trends():
    elapsed = np.cumsum(np.resize([.1, .2, .3, .7, 1.3], 55))
    r = record(n=55, elapsed=elapsed); r['values'][:] = .1
    X, _, _ = causal_features(r)
    np.testing.assert_array_equal(X[:, 120:184], np.zeros((53, 64)))


def test_inner_mining_scores_each_disturbed_record_only_when_record_disjoint():
    rows = [record('m' + str(i), n=36) for i in range(3)]
    anchor = record('normal_train_anchor', n=36); anchor['labels'][:] = False
    counter = {'fits': 0}
    hard, evidence, scores = mine_hard_normals(rows, anchor, forbidden_ids=['outer-held'], fit_counter=counter)
    assert counter['fits'] == evidence['hgb_fits'] == 3
    assert evidence['anchor_mined'] is False and set(hard) == {r['id'] for r in rows}
    for fold in evidence['inner_folds']:
        assert not set(fold['fit']['training_ids']) & set(fold['heldout_ids'])
        assert 'outer-held' not in fold['fit']['training_ids']
        assert 'normal_train_anchor' in fold['fit']['training_ids']
    for row in rows:
        indices = causal_features(row)[1]
        assert len(hard[row['id']]) == 2
        np.testing.assert_array_equal(hard[row['id']], choose_hard_normals(row, scores[row['id']], indices))
    with pytest.raises(ValueError, match='Forbidden'):
        mine_hard_normals(rows, anchor, forbidden_ids=['m0'])


def test_hard_normal_exact_count_stable_ties_and_record_class_mass_anchor_unchanged():
    a = record('a', n=40); anchor = record('normal_train_anchor', n=40); anchor['labels'][:] = False
    records = [anchor, a]; indices = [base_features(r)[1] for r in records]
    scores = np.full(40, np.nan); scores[2:] = .4
    selected = choose_hard_normals(a, scores, indices[1])
    np.testing.assert_array_equal(selected, [2, 3])
    base, _ = sample_weights(records, indices, 'record_equal_class_balanced')
    updated, evidence = hard_normal_weights(records, indices, {'a': selected})
    np.testing.assert_array_equal(updated[:len(indices[0])], base[:len(indices[0])])
    offset = len(indices[0]); normal = ~a['labels'][indices[1]]
    assert updated[offset:][normal].sum() == pytest.approx(base[offset:][normal].sum())
    assert updated[offset:][~normal].sum() == pytest.approx(base[offset:][~normal].sum())
    assert updated.sum() == pytest.approx(base.sum())
    assert evidence['anchor_mined'] is False
    with pytest.raises(ValueError, match='normal'):
        hard_normal_weights(records, indices, {'a': np.array([25])})
    with pytest.raises(ValueError, match='anchor'):
        hard_normal_weights(records, indices, {'a': selected, 'normal_train_anchor': np.array([2])})


def test_inner_folds_are_whole_record_disjoint_stable_and_cover_once():
    ids = ['r' + str(i) for i in range(8)]
    folds = inner_splits(ids)
    assert folds == inner_splits(ids[::-1])
    assert sorted(k for f in folds for k in f['heldout_ids']) == ids
    for fold in folds:
        assert not set(fold['train_ids']) & set(fold['heldout_ids'])


def test_real_hgb_training_prediction_invariance_and_forbidden_record_guard():
    rows = [record('r0', n=50), record('r1', n=50)]
    anchor = record('normal_train_anchor', n=50); anchor['labels'][:] = False
    fitted, evidence = fit_model([anchor] + rows, forbidden_ids=['outside'])
    assert fitted['feature_names'] == feature_names() and evidence['feature_count'] == 186
    r = record('outside', n=44)
    before = classifier_scores(fitted, r)['scores']
    after = classifier_scores(fitted, dict(r, id='changed', labels=None))['scores']
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(np.isfinite(before), np.isfinite(np.r_[np.nan, np.nan, np.zeros(42)]))
    with pytest.raises(ValueError, match='forbidden'):
        fit_model([anchor] + rows, forbidden_ids=['r0'])


def test_selection_requires_progressive_retention_and_explicit_no_feasible():
    reference = {'false_episodes': 3, 'false_alert_seconds': 37., 'observed_normal_seconds': 9179.}
    metric = {'detected_events': 27, 'false_episodes': 1, 'false_episodes_per_hour': .4,
              'false_alert_seconds': 10., 'observed_normal_seconds': 9179., 'median_delay_seconds': 3., 'mean_delay_seconds': 5.}
    assert screen_candidate(metric, reference, []) == []
    assert 'progressive_subset_retention' in screen_candidate(metric, reference, ['other-6-event-0'])
    assert select_candidate([]) is None
    points = [{'arm': 'B', 'threshold_order': 0, 'metrics': metric, 'feasible': False},
              {'arm': 'C', 'threshold_order': 0, 'metrics': metric, 'feasible': True}]
    assert select_candidate(points)['arm'] == 'C'
