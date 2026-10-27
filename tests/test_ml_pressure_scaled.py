import copy
import hashlib
import json

import numpy as np
import pytest

from signalwatch.dataset import SENSORS
from signalwatch.ml_trends import causal_features as trend_features
from signalwatch import ml_pressure_scaled as module


PRESSURE = next(i for i, sensor in enumerate(SENSORS) if sensor['key'] == 'Pressure')


def record(key, n=50):
    rng = np.random.default_rng(sum(map(ord, key)))
    labels = np.arange(n) >= n // 2
    values = rng.normal(size=(n, 8)) * .2 + labels[:, None] * 2.
    return {'id': key, 'values': values, 'labels': labels, 'elapsed': np.arange(n, dtype=float),
            'gap_before': np.zeros(n, bool), 'cadence_seconds': 1., 'group': 'unit', 'sha256': key + '-hash'}


def test_sparse_pressure_zero_iqr_uses_population_std_and_preserves_other_seven():
    bank = np.random.default_rng(4).normal(size=(1000, 8))
    bank[:, PRESSURE] = 0.; bank[-1, PRESSURE] = 1.6396386
    center, scale, evidence = module.bank_scaling(bank)
    original_center = np.median(bank, axis=0)
    original_scale = np.maximum(np.quantile(bank, .75, axis=0) - np.quantile(bank, .25, axis=0), 1e-6)
    other = np.arange(8) != PRESSURE
    np.testing.assert_array_equal(center, original_center)
    np.testing.assert_array_equal(scale[other], original_scale[other])
    assert original_scale[PRESSURE] == 1e-6
    assert scale[PRESSURE] == bank[:, PRESSURE].std(ddof=0)
    assert np.max(np.abs((bank[:, PRESSURE] - center[PRESSURE]) / scale[PRESSURE])) < 128.
    assert evidence['pressure_sensor_key'] == 'Pressure'
    assert evidence['pressure_index'] == PRESSURE
    assert evidence['pressure_std_ddof0'] == scale[PRESSURE]
    assert evidence['bank_pressure_max_abs_z'] < 128.


def test_constant_pressure_floor_and_no_adaptive_rescaling_of_queries():
    bank = np.zeros((40, 8)); bank[:, PRESSURE] = .1
    center, scale, evidence = module.bank_scaling(bank)
    assert center[PRESSURE] == .1 and scale[PRESSURE] == 1e-6
    assert evidence['bank_pressure_max_abs_z'] == 0.
    np.testing.assert_array_equal(scale, np.full(8, 1e-6))


def test_bank_guard_rejects_before_any_estimator_and_never_clips():
    bank = np.zeros((20000, 8)); bank[-1, PRESSURE] = 1.
    with pytest.raises(ValueError, match='128'):
        module.bank_scaling(bank)
    with pytest.raises(ValueError, match='finite'):
        module.bank_scaling(np.full((10, 8), np.nan))


def test_actual_fit_preserves_scalers_masks_prefix_labels_and_training_exclusion():
    anchor = record('normal_train_anchor', 60); anchor['labels'][:] = False
    rows = [anchor, record('a', 60), record('b', 60)]
    for r in rows:
        r['values'][:, PRESSURE] = 0.
        r['values'][4, PRESSURE] = 1.6
    counters = module.new_counters()
    model, evidence = module.fit_model(rows, 2, forbidden_ids=['held'], counters=counters)
    assert model is not None and evidence['valid']
    assert model['version'] == 'pressure-scaled-normal-regimes-13.0.0'
    assert len(model['feature_names']) == 198
    designs = np.concatenate([trend_features(r)[0] for r in rows])
    expected_center = np.median(designs, axis=0)
    expected_scale = np.maximum(np.quantile(designs, .75, axis=0) - np.quantile(designs, .25, axis=0),
                                np.maximum(designs.std(axis=0) * .05, 1e-6))
    np.testing.assert_array_equal(model['feature_center'][:186], expected_center)
    np.testing.assert_array_equal(model['feature_scale'][:186], expected_scale)
    assert counters['gmm_attempted'] == counters['hgb_attempted'] == 1
    assert evidence['normal_bank']['forbidden_ids'] == ['held']
    assert evidence['numerical_audits']['bank']['rows'] == evidence['normal_bank']['rows']
    assert evidence['numerical_audits']['training']['rows'] == sum(len(trend_features(r)[1]) for r in rows)
    assert evidence['numerical_audits']['bank']['absolute_nll_difference']['max'] < 1e-8
    assert set(evidence['training_numerical_audits_by_recording']) == {r['id'] for r in rows}
    held = record('held', 55); held['values'][:, PRESSURE] = 0.; held['gap_before'][30] = True
    original = module.classifier_scores(model, held)['scores']
    eligible = np.zeros(55, bool); eligible[trend_features(held)[1]] = True
    np.testing.assert_array_equal(np.isfinite(original), eligible)
    np.testing.assert_allclose(module.classifier_scores(model, dict(held, id='irrelevant', labels=None))['scores'],
                               original, equal_nan=True, atol=1e-12, rtol=0)
    prefix = {key: value[:25] if isinstance(value, np.ndarray) else value for key, value in held.items()}
    np.testing.assert_allclose(module.classifier_scores(model, prefix)['scores'], original[:25], equal_nan=True,
                               atol=1e-12, rtol=0)
    with pytest.raises(ValueError, match='forbidden'):
        module.fit_model(rows, 2, forbidden_ids=['a'])


def test_fixed_parameter_nll_audit_is_direct_and_component_permutation_invariant():
    anchor = record('normal_train_anchor', 60); anchor['labels'][:] = False
    bundle, _ = module.fit_model([anchor, record('a', 60)], 2)
    values = record('query', 40)['values']
    audit = module.numerical_audit(bundle['regime'], values)
    assert audit['rows'] == 40 and audit['absolute_nll_difference']['max'] < 1e-8
    assert audit['minimum_variance'] > 0 and audit['reg_covar'] == .001
    assert audit['minimum_variance_minus_reg_covar'] >= -1e-10
    permuted = copy.deepcopy(bundle['regime'])
    for attr in ['weights_', 'means_', 'covariances_', 'precisions_', 'precisions_cholesky_']:
        setattr(permuted['model'], attr, getattr(permuted['model'], attr)[::-1].copy())
    after = module.numerical_audit(permuted, values)
    assert abs(after['direct_nll']['median'] - audit['direct_nll']['median']) < 1e-10
    zero_weight = copy.deepcopy(bundle['regime'])
    zero_weight['model'].weights_ = np.array([1., 0.])
    zero_audit = module.numerical_audit(zero_weight, values)
    assert zero_audit['finite_audit_values'] and zero_audit['numerically_accepted']
    assert zero_audit['absolute_nll_difference']['max'] < 1e-8
    actual_feature_nll = module.regime_features(bundle['regime'], values)[:, 0]
    changed_audit = module.numerical_audit(bundle['regime'], values, installed_nll=actual_feature_nll + 1e-4)
    assert not changed_audit['numerically_accepted']
    assert changed_audit['absolute_nll_difference']['min'] > .00009999
    with pytest.raises(ValueError, match='match'):
        module.numerical_audit(bundle['regime'], values, installed_nll=actual_feature_nll[:-1])
    nonfinite = actual_feature_nll.copy(); nonfinite[0] = np.nan
    failed = module.numerical_audit(bundle['regime'], values, installed_nll=nonfinite)
    assert not failed['finite_audit_values'] and not failed['numerically_accepted']


def test_no_feasible_partial_oof_and_shared_selection_identity_contract(tmp_path):
    assert module.select_candidate([]) is None
    rows = [record(name, 120) for name in ['other-6', 'other-9']]
    fusion = {r['id']: np.where(r['labels'], 2., .1) for r in rows}
    classifier = {r['id']: np.where(r['labels'], .95, .1) for r in rows}
    result = module.evaluate_candidates(rows, {}, fusion, classifier, .8)
    assert result['selected'] is None
    with pytest.raises(ValueError, match='Partial'):
        module.evaluate_candidates(rows, {'K2': {'other-6': classifier['other-6']}}, fusion, classifier, .8)
    selected = {'arm': 'K2', 'threshold': .9}
    (tmp_path / 'candidate_audits.json').write_text(json.dumps([selected]))
    np.savez_compressed(tmp_path / 'K2_oof_scores.npz', sample=np.array([np.nan, .8, .95]))
    module.freeze_selection(tmp_path, selected, 'plan-hash', ['K2'], 18)
    evidence = json.loads((tmp_path / 'selection_before_final_fit.json').read_text())
    assert evidence['selected'] == selected and evidence['plan_sha256'] == 'plan-hash'
    assert evidence['oof_hashes']['K2'] == hashlib.sha256((tmp_path / 'K2_oof_scores.npz').read_bytes()).hexdigest()


def test_numeric_failure_rejects_original_quality_winner_without_fallback():
    metric = {'detected_events': 27, 'false_episodes': 1, 'false_alert_seconds': 20.,
              'median_delay_seconds': 4., 'mean_delay_seconds': 5.}
    points = [{'arm': arm, 'threshold_order': 0, 'metrics': metric, 'feasible': True}
              for arm in ['K2', 'K1']]
    winner = module.select_candidate(points)
    assert winner['arm'] == 'K1'
    decision = module.quality_decision(winner, {'K1': {'numerically_accepted': False},
                                               'K2': {'numerically_accepted': True}})
    assert decision['development_selected'] == winner and decision['selected'] is None
    assert decision['status'] == 'numerical_check_rejected'
    assert module.quality_decision(None, {})['status'] == 'no_feasible_candidate'


def test_unchanged_fold_requires_exact_mask_and_tolerant_score_identity():
    detail = {'normal_bank': {'selected_indices': {'a': [2, 3]}}, 'bank_center': [0.] * 8,
              'bank_scale': [1.] * 8, 'feature_center': [0.] * 198, 'feature_scale': [1.] * 198}
    scores = {'held': np.array([np.nan, np.nan, .7, .9])}
    reference = {'held': scores['held'].copy()}
    check = module.check_v12_fold_integrity(detail, copy.deepcopy(detail), scores, reference)
    assert check['pressure_scale_unchanged'] and check['unchanged_fold_score_identity_passed']
    changed = copy.deepcopy(scores); changed['held'][2] += 1e-8
    with pytest.raises(ValueError, match='unchanged'):
        module.check_v12_fold_integrity(detail, copy.deepcopy(detail), changed, reference)
    changed = copy.deepcopy(scores); changed['held'][1] = .1
    with pytest.raises(ValueError, match='mask'):
        module.check_v12_fold_integrity(detail, copy.deepcopy(detail), changed, reference)
    modified = copy.deepcopy(detail); modified['bank_scale'][0] = 2.
    with pytest.raises(ValueError, match='seven'):
        module.check_v12_fold_integrity(modified, detail, scores, reference)


@pytest.mark.parametrize('nonconverged', [False, True])
def test_final_rejection_preserves_selection_and_writes_no_model(tmp_path, monkeypatch, nonconverged):
    selected = {'arm': 'K1', 'components': 1, 'threshold': .9}
    (tmp_path / 'selection_before_final_fit.json').write_text(json.dumps({'selected': selected}))
    before = (tmp_path / 'selection_before_final_fit.json').read_bytes()
    def rejected(records, components, counters=None):
        counters['gmm_attempted'] += 1; counters['hgb_attempted'] += 1
        if nonconverged:
            return None, {'valid': False, 'reason': 'mixture_nonconvergence'}
        return {'not_serialized': True}, {'valid': True, 'numerically_accepted': False}
    monkeypatch.setattr(module, 'fit_model', rejected)
    result = module.fit_final_candidate(tmp_path, [], selected, module.new_counters(),
                                        'plan-hash', hashlib.sha256(before).hexdigest())
    assert result['selected'] is None and result['status'] == 'final_fit_rejected'
    assert result['final_fit_rejection'] == ('mixture_nonconvergence' if nonconverged else 'final_numerical_check_failed')
    assert not result['final_model_available'] and 'model_sha256' not in result
    assert not (tmp_path / 'selected_bundle.joblib').exists()
    assert (tmp_path / 'selection_before_final_fit.json').read_bytes() == before


def test_final_success_embeds_plan_selection_and_probability_cutoff(tmp_path, monkeypatch):
    selected = {'arm': 'K2', 'components': 2, 'threshold': .917}
    path = tmp_path / 'selection_before_final_fit.json'; path.write_text(json.dumps({'selected': selected}))
    selection_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    def accepted(records, components, counters=None):
        counters['gmm_attempted'] += 1; counters['hgb_attempted'] += 1
        return {'arm': 'K2', 'feature_names': module.feature_names()}, {'valid': True, 'numerically_accepted': True}
    monkeypatch.setattr(module, 'fit_model', accepted)
    result = module.fit_final_candidate(tmp_path, [], selected, module.new_counters(), 'plan-hash', selection_sha)
    import joblib
    bundle = joblib.load(tmp_path / 'selected_bundle.joblib')
    assert result['final_model_available'] and result['selected'] == selected
    assert bundle['threshold'] == .917 and bundle['plan_sha256'] == 'plan-hash'
    assert bundle['selection_before_final_fit_sha256'] == selection_sha
    assert len(bundle['feature_names']) == 198


def test_failed_bank_guard_records_zero_estimator_calls(monkeypatch):
    bank = np.zeros((20000, 8)); bank[-1, PRESSURE] = 1.
    monkeypatch.setattr(module, 'balanced_normal_bank', lambda records, forbidden_ids: (bank, {}))
    counters = module.new_counters()
    model, evidence = module.fit_model([record('a')], 1, counters=counters)
    assert model is None and evidence['reason'] == 'invalid_normal_bank_scaling'
    assert not evidence['classifier_fitted']
    assert counters['gmm_attempted'] == counters['hgb_attempted'] == 0


@pytest.mark.parametrize('individual_only', [False, True])
def test_finite_failed_numeric_flags_do_not_discard_outer_oof(monkeypatch, individual_only):
    rows = [record('r' + str(i)) for i in range(3)]
    anchor = record('normal_train_anchor'); anchor['labels'][:] = False
    folds = [{'fold': i, 'train_ids': [r['id'] for j, r in enumerate(rows) if j != i],
              'heldout_ids': [rows[i]['id']]} for i in range(3)]
    def fitted(records, components, forbidden_ids=None, counters=None):
        for key in counters:
            counters[key] += 1
        return {'regime': {}}, {'valid': True, 'numerical_audits': {'bank': {'numerically_accepted': True}},
                                'pressure_scaling': {'new_pressure_scale': 1.}}
    monkeypatch.setattr(module, 'fit_model', fitted)
    monkeypatch.setattr(module, 'classifier_scores', lambda model, r: {'scores': np.r_[np.nan, np.nan, np.full(len(r['values']) - 2, .2)]})
    monkeypatch.setattr(module, 'normal_density_summary', lambda model, rows: {})
    audit_calls = []
    def audit(regime, values):
        audit_calls.append(len(values))
        return {'numerically_accepted': len(audit_calls) != 1 if individual_only else False}
    monkeypatch.setattr(module, 'numerical_audit', audit)
    result = module.fit_outer_arm(rows, anchor, folds, 2, module.new_counters())
    assert result['valid'] and not result['numerically_accepted']
    assert set(result['scores']) == {r['id'] for r in rows}
    assert result['skipped_top_level_fit_calls'] == 0
    assert result['discarded_scores'] == {}


def test_frontiers_use_identical_declared_rules_without_becoming_selection():
    def point(order, detected, false, seconds, lost):
        return {'arm': 'K1', 'threshold_order': order, 'lost_progressive_event_ids': lost,
                'metrics': {'detected_events': detected, 'false_episodes': false,
                            'false_alert_seconds': seconds, 'false_episodes_per_hour': false / 2.5,
                            'median_delay_seconds': 3., 'mean_delay_seconds': 4.}}
    points = [point(0, 27, 5, 37., []), point(1, 26, 4, 36., []), point(2, 26, 1, 28., ['other-6-event-0'])]
    frontiers = module.descriptive_frontiers(points)
    assert frontiers['progressive_recall_frontier']['threshold_order'] == 1
    assert frontiers['burden_limited_recall_frontier']['threshold_order'] == 2
    assert frontiers['burden_limited_recall_frontier']['lost_progressive_event_ids'] == ['other-6-event-0']


def test_new_plan_freezes_generic_diagnostic_sources_even_if_absent_in_v12(tmp_path):
    prior = tmp_path / 'artifacts/regime-development'; prior.mkdir(parents=True)
    plan = {'dependencies': {}, 'normal_bank': {'sensor_center': 'bankmedian'}}
    (prior / 'plan.json').write_text(json.dumps(plan))
    (prior / 'candidate_audits.json').write_text('[]')
    (prior / 'selection_before_final_fit.json').write_text('{"selected":null}')
    digest = lambda name: hashlib.sha256((prior / name).read_bytes()).hexdigest()
    report = {'plan_sha256': digest('plan.json'), 'score_artifacts_sha256': {}, 'fit_evidence_sha256': {},
              'candidate_audits_sha256': digest('candidate_audits.json'),
              'selection_before_final_fit_sha256': digest('selection_before_final_fit.json')}
    (prior / 'report.json').write_text(json.dumps(report))
    protected = ['signalwatch/trend_diagnostics.py', 'tests/test_trend_diagnostics.py', 'scripts/check_trend_model.py']
    for name in [module.PROTOCOL, 'signalwatch/ml_pressure_scaled.py', 'tests/test_ml_pressure_scaled.py',
                 'scripts/study_pressure_scaled.py', *protected]:
        path = tmp_path / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
    result = module.study_plan(tmp_path)
    assert set(protected) <= set(result['dependencies'])
    assert all(result['dependencies'][name] == hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() for name in protected)

