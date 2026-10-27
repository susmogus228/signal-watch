import copy
import numpy as np
import pytest

from signalwatch.ml_trends import causal_features as trend_features
from signalwatch.ml_regimes import (
    balanced_normal_bank, regime_features, feature_names, fit_model, classifier_scores,
    fit_outer_arm, select_candidate, new_counters,
    evaluate_candidates,
)


def record(key, n=40):
    rng = np.random.default_rng(sum(map(ord, key)))
    labels = np.arange(n) >= n // 2
    values = rng.normal(size=(n, 8)) * .2 + labels[:, None] * 2.
    return {'id': key, 'values': values, 'labels': labels, 'elapsed': np.arange(n, dtype=float),
            'gap_before': np.zeros(n, bool), 'cadence_seconds': 1., 'group': 'unit', 'sha256': key + '-hash'}


def test_exact_balanced_normal_bank_with_endpoints_anchor_and_train_exclusion():
    a = record('a', 40); b = record('b', 60); anchor = record('normal_train_anchor', 80); anchor['labels'][:] = False
    rows = [anchor, a, b]
    bank, evidence = balanced_normal_bank(rows, forbidden_ids=['held'])
    assert evidence['m'] == 18 and bank.shape == (54, 8)
    for r in rows:
        eligible = trend_features(r)[1]; normal = eligible[~r['labels'][eligible]]
        expected = normal[np.floor(np.linspace(0, len(normal) - 1, 18)).astype(int)]
        np.testing.assert_array_equal(evidence['selected_indices'][r['id']], expected)
        assert len(set(expected)) == 18 and expected[0] == normal[0] and expected[-1] == normal[-1]
    with pytest.raises(ValueError, match='forbidden'):
        balanced_normal_bank(rows, forbidden_ids=['a'])


def test_too_small_or_empty_normal_bank_invalidates_without_dropping_record():
    a = record('a', 10); a['labels'][:] = True; a['labels'][2] = False
    anchor = record('normal_train_anchor', 20); anchor['labels'][:] = False
    with pytest.raises(ValueError, match='at least two'):
        balanced_normal_bank([anchor, a])
    a['labels'][:] = True
    with pytest.raises(ValueError, match='normal'):
        balanced_normal_bank([anchor, a])


def test_actual_two_stage_fit_dimensions_scalers_bank_and_prefix_gap_label_invariance():
    anchor = record('normal_train_anchor', 50); anchor['labels'][:] = False
    rows = [anchor, record('a', 50), record('b', 50)]
    counters = new_counters()
    model, evidence = fit_model(rows, 2, forbidden_ids=['held'], counters=counters)
    assert model is not None and evidence['valid']
    assert len(model['feature_names']) == len(feature_names()) == 198
    assert model['feature_center'].shape == model['feature_scale'].shape == (198,)
    assert counters['gmm_attempted'] == counters['gmm_completed'] == counters['gmm_converged'] == 1
    assert counters['hgb_attempted'] == counters['hgb_completed'] == 1
    designs = np.concatenate([trend_features(r)[0] for r in rows])
    center = np.median(designs, axis=0)
    scale = np.maximum(np.quantile(designs, .75, axis=0) - np.quantile(designs, .25, axis=0),
                       np.maximum(designs.std(axis=0) * .05, 1e-6))
    np.testing.assert_array_equal(model['feature_center'][:186], center)
    np.testing.assert_array_equal(model['feature_scale'][:186], scale)
    r = record('held', 55); r['gap_before'][30] = True
    before = classifier_scores(model, r)['scores']
    indices = trend_features(r)[1]; eligible = np.zeros(55, bool); eligible[indices] = True
    np.testing.assert_array_equal(np.isfinite(before), eligible)
    changed = dict(r, id='not-feature', labels=None)
    np.testing.assert_allclose(classifier_scores(model, changed)['scores'], before, atol=1e-12, rtol=0, equal_nan=True)
    prefix = {key: (value[:25] if isinstance(value, np.ndarray) else value) for key, value in r.items()}
    np.testing.assert_allclose(classifier_scores(model, prefix)['scores'], before[:25], atol=1e-12, rtol=0, equal_nan=True)
    assert not np.any(np.isfinite(before[:2])) and np.isfinite(before[2])


def test_regime_formulas_and_component_permutation_invariance():
    anchor = record('normal_train_anchor', 60); anchor['labels'][:] = False
    model, _ = fit_model([anchor, record('a', 60), record('b', 60)], 2)
    regime = model['regime']; values = record('query', 15)['values']
    original = regime_features(regime, values)
    assert original.shape == (15, 12) and np.isfinite(original).all()
    z = (values - regime['bank_center']) / regime['bank_scale']
    gmm = regime['model']; posterior = gmm.predict_proba(z)
    residual = (z[:, None, :] - gmm.means_[None, :, :]) / np.sqrt(gmm.covariances_[None, :, :])
    squared = np.sum(residual * residual, axis=2)
    expected = np.column_stack([-gmm.score_samples(z), squared.min(axis=1),
                                np.sum(posterior * squared, axis=1), posterior.max(axis=1),
                                np.sum(posterior[:, :, None] * residual, axis=1)])
    np.testing.assert_allclose(original, expected, atol=1e-12, rtol=0)
    permuted = copy.deepcopy(regime)
    for attribute in ['weights_', 'means_', 'covariances_', 'precisions_', 'precisions_cholesky_']:
        setattr(permuted['model'], attribute, getattr(permuted['model'], attribute)[::-1].copy())
    np.testing.assert_allclose(regime_features(permuted, values), original, atol=1e-10, rtol=1e-12)
    invalid = copy.deepcopy(regime); invalid['model'].covariances_[0, 0] = 0.
    with pytest.raises(ValueError, match='variance'):
        regime_features(invalid, values)


def test_constant_bank_channels_remain_finite_with_covariance_regularization():
    rows = [record('normal_train_anchor', 30), record('a', 30)]
    rows[0]['labels'][:] = False
    for r in rows:
        r['values'][:, 0] = .1
    model, evidence = fit_model(rows, 1)
    assert model is not None and evidence['mixture']['converged']
    assert model['regime']['bank_scale'][0] == 1e-6
    assert np.isfinite(regime_features(model['regime'], rows[1]['values'])).all()


def test_nonconverged_mixture_invalidates_before_any_classifier_fit(monkeypatch):
    import signalwatch.ml_regimes as module
    original = module.GaussianMixture
    class Nonconverged(original):
        def fit(self, X, y=None):
            fitted = super().fit(X, y)
            self.converged_ = False
            return fitted
    monkeypatch.setattr(module, 'GaussianMixture', Nonconverged)
    anchor = record('normal_train_anchor'); anchor['labels'][:] = False
    counters = new_counters()
    model, evidence = fit_model([anchor, record('a')], 1, counters=counters)
    assert model is None and not evidence['valid']
    assert evidence['reason'] == 'mixture_nonconvergence'
    assert counters['gmm_attempted'] == counters['gmm_completed'] == 1
    assert counters['gmm_converged'] == counters['hgb_attempted'] == 0


def test_invalid_outer_fold_discards_partial_oof_and_skips_remaining_arm(monkeypatch):
    import signalwatch.ml_regimes as module
    rows = [record('r' + str(i)) for i in range(3)]
    anchor = record('normal_train_anchor'); anchor['labels'][:] = False
    folds = [{'fold': i, 'train_ids': [r['id'] for j, r in enumerate(rows) if j != i],
              'heldout_ids': [rows[i]['id']]} for i in range(3)]
    calls = []
    def fake_fit(records, components, forbidden_ids=None, counters=None):
        calls.append(forbidden_ids)
        counters['gmm_attempted'] += 1; counters['gmm_completed'] += 1
        if len(calls) == 2:
            return None, {'valid': False, 'reason': 'mixture_nonconvergence'}
        counters['gmm_converged'] += 1; counters['hgb_attempted'] += 1; counters['hgb_completed'] += 1
        return {'fake': True}, {'valid': True}
    monkeypatch.setattr(module, 'fit_model', fake_fit)
    monkeypatch.setattr(module, 'classifier_scores', lambda bundle, r: {'scores': np.r_[np.nan, np.nan, np.full(len(r['values']) - 2, .2)]})
    monkeypatch.setattr(module, 'normal_density_summary', lambda bundle, records: {})
    result = fit_outer_arm(rows, anchor, folds, 2, counters=new_counters())
    assert not result['valid'] and result['scores'] == {}
    assert len(calls) == 2 and result['skipped_folds'] == [2]
    assert result['discarded_partial_record_ids'] == ['r0']


def test_explicit_no_feasible_and_predeclared_k_tie_order():
    metric = {'detected_events': 27, 'false_episodes': 1, 'false_alert_seconds': 20.,
              'median_delay_seconds': 4., 'mean_delay_seconds': 5.}
    points = [{'arm': arm, 'threshold_order': 0, 'metrics': metric, 'feasible': True} for arm in ['K4', 'K2', 'K1']]
    assert select_candidate(points)['arm'] == 'K1'
    assert select_candidate([]) is None
    assert select_candidate([dict(points[0], feasible=False)]) is None


def test_positive_classifier_is_immediate_at_original_first_eligible_row(monkeypatch):
    import signalwatch.ml_regimes as module
    class PositiveClassifier:
        classes_ = np.array([0, 1])
        def predict_proba(self, X):
            assert X.shape[1] == 198
            return np.column_stack([np.zeros(len(X)), np.ones(len(X))])
    monkeypatch.setattr(module, 'regime_features', lambda regime, values: np.zeros((len(values), 12)))
    bundle = {'model': PositiveClassifier(), 'regime': {}, 'feature_center': np.zeros(198),
              'feature_scale': np.ones(198), 'sensor_scale': np.ones(8), 'threshold': .8}
    scores = classifier_scores(bundle, record('query', 10))['scores']
    assert np.isnan(scores[:2]).all()
    assert scores[2] > bundle['threshold']
    assert (scores[2:] > bundle['threshold']).all()


def test_all_invalid_arms_give_no_candidate_and_partial_oof_is_rejected():
    rows = [record(name, 120) for name in ['other-6', 'other-9']]
    fusion = {r['id']: np.where(r['labels'], 2., .1) for r in rows}
    classifier = {r['id']: np.where(r['labels'], .95, .1) for r in rows}
    result = evaluate_candidates(rows, {}, fusion, classifier, .8)
    assert result['selected'] is None and result['candidate_audits'] == []
    with pytest.raises(ValueError, match='Partial'):
        evaluate_candidates(rows, {'K2': {'other-6': classifier['other-6']}}, fusion, classifier, .8)


def test_selection_manifest_uses_shared_diagnostic_oof_hash_contract(tmp_path):
    import hashlib
    import json
    import signalwatch.ml_regimes as module
    selected = {'arm': 'K2', 'threshold': .9}
    (tmp_path / 'candidate_audits.json').write_text(json.dumps([selected]))
    np.savez_compressed(tmp_path / 'K2_oof_scores.npz', sample=np.array([np.nan, .8, .95]))
    module.freeze_selection(tmp_path, selected, 'frozen-plan-hash', ['K2'], 18)
    manifest = json.loads((tmp_path / 'selection_before_final_fit.json').read_text())
    assert manifest['selected'] == selected and manifest['plan_sha256'] == 'frozen-plan-hash'
    assert manifest['oof_hashes']['K2'] == hashlib.sha256((tmp_path / 'K2_oof_scores.npz').read_bytes()).hexdigest()
    assert manifest['candidate_audits_sha256'] == hashlib.sha256((tmp_path / 'candidate_audits.json').read_bytes()).hexdigest()
