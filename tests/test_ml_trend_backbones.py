import numpy as np
import pytest

from signalwatch.ml_trends import causal_features, feature_names
from signalwatch.ml_trend_backbones import fit_model, classifier_scores, select_candidate, evaluate_candidates


def record(key, n=44):
    elapsed = np.arange(n, dtype=float)
    labels = np.arange(n) >= n // 2
    rng = np.random.default_rng(sum(map(ord, key)))
    values = rng.normal(size=(n, 8)) * .1 + labels[:, None] * 3.
    return {'id': key, 'values': values, 'elapsed': elapsed, 'labels': labels,
            'gap_before': np.zeros(n, bool), 'cadence_seconds': 1., 'group': 'unit'}


def test_fixed_learners_fit_actual_estimators_same_preprocessing_and_class_mass():
    anchor = record('normal_train_anchor'); anchor['labels'][:] = False
    rows = [anchor, record('r0'), record('r1')]
    fitted = {}; counter = {'fits': 0}
    for learner in ['HGB31', 'ExtraTrees256']:
        model, evidence = fit_model(rows, learner, forbidden_ids=['held'], fit_counter=counter)
        fitted[learner] = model
        assert evidence['feature_count'] == 186
        assert model['feature_names'] == feature_names()
        assert evidence['sample_weights']['scheme'] == 'record_equal_class_balanced'
        assert set(evidence['training_ids']) == {'normal_train_anchor', 'r0', 'r1'}
    assert counter['fits'] == 2
    assert fitted['HGB31']['model'].get_params()['max_leaf_nodes'] == 31
    assert fitted['HGB31']['model'].get_params()['max_iter'] == 150
    assert fitted['ExtraTrees256']['model'].get_params()['n_estimators'] == 256
    assert fitted['ExtraTrees256']['model'].get_params()['min_samples_leaf'] == 4
    assert fitted['ExtraTrees256']['model'].get_params()['max_features'] == 1.
    np.testing.assert_array_equal(fitted['HGB31']['feature_center'], fitted['ExtraTrees256']['feature_center'])
    np.testing.assert_array_equal(fitted['HGB31']['feature_scale'], fitted['ExtraTrees256']['feature_scale'])
    r = record('held', n=50); r['gap_before'][30] = True
    for model in fitted.values():
        predicted = classifier_scores(model, r)['scores']
        _, indices, _ = causal_features(r)
        expected = np.zeros(50, bool); expected[indices] = True
        np.testing.assert_array_equal(np.isfinite(predicted), expected)
        changed = dict(r, id='not-a-feature', labels=None)
        np.testing.assert_allclose(classifier_scores(model, changed)['scores'], predicted, atol=1e-12, rtol=0, equal_nan=True)
        prefix = {key: (value[:25] if isinstance(value, np.ndarray) else value) for key, value in r.items()}
        np.testing.assert_allclose(classifier_scores(model, prefix)['scores'], predicted[:25], atol=1e-12, rtol=0, equal_nan=True)


def test_forbidden_records_unknown_learner_and_missing_classes_reject_before_fit():
    rows = [record('r0'), record('r1')]; counter = {'fits': 0}
    with pytest.raises(ValueError, match='forbidden'):
        fit_model(rows, 'HGB31', forbidden_ids=['r0'], fit_counter=counter)
    with pytest.raises(ValueError, match='learner'):
        fit_model(rows, 'not-frozen', fit_counter=counter)
    rows[0]['labels'][:] = False; rows[1]['labels'][:] = False
    with pytest.raises(ValueError, match='classes'):
        fit_model(rows, 'ExtraTrees256', fit_counter=counter)
    assert counter['fits'] == 0


def test_selection_keeps_gate_and_predeclared_learner_tie_order():
    metrics = {'detected_events': 27, 'false_episodes': 1, 'false_alert_seconds': 20.,
               'median_delay_seconds': 3., 'mean_delay_seconds': 4.}
    points = [{'arm': 'ExtraTrees256', 'threshold_order': 0, 'metrics': metrics, 'feasible': True},
              {'arm': 'HGB31', 'threshold_order': 1, 'metrics': metrics, 'feasible': True}]
    assert select_candidate(points)['arm'] == 'HGB31'
    assert select_candidate([]) is None
    assert select_candidate([dict(points[0], feasible=False)]) is None


def test_evaluator_preserves_false_rate_and_progressive_guards_without_renaming_models():
    from signalwatch.ml_trends import PROGRESSIVE_EVENTS
    rows = []
    for name in ['other-6', 'other-9']:
        r = record(name, n=120); r['labels'][:] = False; r['labels'][70:90] = True
        rows.append(r)
    source = {r['id']: np.where(r['labels'], 2., .1) for r in rows}
    source_classifier = {r['id']: np.where(r['labels'], .95, .1) for r in rows}
    candidate = {name: {r['id']: source_classifier[r['id']].copy() for r in rows}
                 for name in ['HGB31', 'ExtraTrees256']}
    diagnostics = evaluate_candidates(rows, candidate, source, source_classifier, .8)
    assert set(row['arm'] for row in diagnostics['candidate_audits']) == {'HGB31', 'ExtraTrees256'}
    assert diagnostics['selected'] is None
    assert all('development_recall_floor' in row['rejection_reasons'] for row in diagnostics['candidate_audits'])
    assert all(not row['lost_progressive_event_ids'] for row in diagnostics['candidate_audits'])
    assert diagnostics['progressive_event_ids'] == PROGRESSIVE_EVENTS
