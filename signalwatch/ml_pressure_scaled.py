"""V13 training-bank Pressure scaling correction; immediate learned decisions."""
from __future__ import annotations

import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from signalwatch.dataset import SENSORS
from signalwatch.ml_ablation import probability_diagnostics
from signalwatch.ml_supervised import causal_features as base_features
from signalwatch.ml_supervised_weighted import sample_weights
from signalwatch.ml_trends import (
    ACTIVE_FILES, PARAMETERS, _load_development, _reports, _events, _groups,
    causal_features as trend_features,
)
from signalwatch.ml_regimes import (
    COMPONENTS, MIXTURE_PARAMETERS, SEED, NormalBankError, _valid_mixture,
    _hash, _read, _write, _arrays, _fit_call_count,
    balanced_normal_bank, new_counters, feature_names, regime_features,
    classifier_scores, normal_density_summary, select_candidate,
    evaluate_candidates, freeze_selection,
)

PROTOCOL = 'docs/ML_V13_PRESSURE_SCALING_PLAN.md'
TARGET = 'artifacts/pressure-scaled-development'
MAX_PRESSURE_Z = 128.
MAX_NLL_DIFFERENCE = 1e-7
VARIANCE_TOLERANCE = 1e-9


class BankScalingError(NormalBankError):
    """A training-bank numerical preflight failure invalidates the whole arm."""


def bank_scaling(bank: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    bank = np.asarray(bank, dtype=np.float64)
    if bank.ndim != 2 or bank.shape[1] != 8 or not len(bank) or not np.isfinite(bank).all():
        raise BankScalingError('Normal bank must contain finite eight-sensor rows.')
    pressure = [i for i, sensor in enumerate(SENSORS) if sensor['key'] == 'Pressure']
    if len(pressure) != 1:
        raise BankScalingError('The sensor schema must contain exactly one Pressure column.')
    index = pressure[0]
    center = np.median(bank, axis=0)
    iqr = np.quantile(bank, .75, axis=0) - np.quantile(bank, .25, axis=0)
    original = np.maximum(iqr, 1e-6)
    scale = original.copy()
    std = float(np.std(bank[:, index], ddof=0, dtype=np.float64))
    scale[index] = max(iqr[index], std, 1e-6)
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise BankScalingError('Normal-bank scales and centers must be finite and scales positive.')
    transformed = (bank - center) / scale
    maximum = float(np.max(np.abs(transformed[:, index])))
    if not np.isfinite(transformed).all() or maximum > MAX_PRESSURE_Z:
        raise BankScalingError('Training-bank maximum absolute standardized Pressure exceeds128 or is nonfinite.')
    evidence = {'pressure_sensor_key': 'Pressure', 'pressure_index': index,
                'pressure_iqr': float(iqr[index]), 'pressure_std_ddof0': std,
                'original_bank_scale': original.tolist(), 'new_bank_scale': scale.tolist(),
                'original_pressure_scale': float(original[index]), 'new_pressure_scale': float(scale[index]),
                'changed_column_indices': np.flatnonzero(scale != original).tolist(),
                'pressure_min': float(bank[:, index].min()), 'pressure_max': float(bank[:, index].max()),
                'bank_pressure_max_abs_z': maximum,
                'bank_max_abs_z_by_sensor': np.max(np.abs(transformed), axis=0).tolist(),
                'guard_bound': MAX_PRESSURE_Z, 'guard_passed': True,
                'old_pressure_physical_reg_covar': float(MIXTURE_PARAMETERS['reg_covar'] * original[index] ** 2),
                'new_pressure_physical_reg_covar': float(MIXTURE_PARAMETERS['reg_covar'] * scale[index] ** 2)}
    return center, scale, evidence


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    if not len(values):
        return {'count': 0, 'min': None, 'median': None, 'p95': None, 'p99': None, 'max': None}
    return {'count': len(values), 'min': float(values.min()), 'median': float(np.median(values)),
            'p95': float(np.quantile(values, .95)), 'p99': float(np.quantile(values, .99)),
            'max': float(values.max())}


def numerical_audit(regime: dict, values: np.ndarray, installed_nll: np.ndarray | None = None) -> dict:
    """Compare density arithmetic at fixed parameters; never change model features."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError('Numerical audit requires finite eight-sensor rows.')
    model = regime['model']
    z = (values - regime['bank_center']) / regime['bank_scale']
    minimum = float(model.covariances_.min()); reg = float(model.reg_covar)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        squared = np.sum((z[:, None, :] - model.means_[None, :, :]) ** 2 /
                         model.covariances_[None, :, :], axis=2)
        log_weights = np.where(model.weights_ > 0, np.log(model.weights_), -np.inf)
        log_components = log_weights[None, :] - .5 * (
            8 * np.log(2 * np.pi) + np.log(model.covariances_).sum(axis=1)[None, :] + squared)
        direct = -logsumexp(log_components, axis=1)
        if installed_nll is None:
            with threadpool_limits(limits=2):
                installed = -model.score_samples(z) if len(z) else np.empty(0)
        else:
            installed = np.asarray(installed_nll, dtype=np.float64)
            if installed.shape != (len(values),):
                raise ValueError('Supplied actual-feature NLL must match the audited rows.')
        difference = np.abs(direct - installed)
        relative = difference / np.maximum(1., np.abs(direct))
    finite = bool(np.isfinite(z).all() and np.isfinite(direct).all() and
                  np.isfinite(installed).all() and np.isfinite(difference).all() and np.isfinite(minimum))
    maximum = float(difference.max()) if len(difference) else 0.
    accepted = finite and maximum <= MAX_NLL_DIFFERENCE and minimum >= reg - VARIANCE_TOLERANCE
    return {'rows': len(values), 'sensor_keys': [s['key'] for s in SENSORS],
            'max_abs_z_by_sensor': (np.max(np.abs(z), axis=0) if len(z) else np.zeros(8)).tolist(),
            'direct_nll': _summary(direct), 'installed_nll': _summary(installed),
            'absolute_nll_difference': _summary(difference), 'relative_nll_difference': _summary(relative),
            'minimum_variance': minimum, 'reg_covar': reg,
            'minimum_variance_minus_reg_covar': minimum - reg,
            'minimum_variance_over_reg_covar': minimum / reg,
            'finite_audit_values': finite, 'numerically_accepted': bool(accepted),
            'installed_nll_source': 'actual feature values' if installed_nll is not None else 'same-shape score_samples call',
            'limits': {'max_absolute_nll_difference': MAX_NLL_DIFFERENCE,
                       'minimum_variance': reg - VARIANCE_TOLERANCE},
            'interpretation': 'Fixed-parameter arithmetic audit, not an EM-error measurement or alarm-quality criterion.'}


def fit_model(records: list[dict], components: int, forbidden_ids: list[str] | None = None,
              counters: dict | None = None) -> tuple[dict | None, dict]:
    if components not in COMPONENTS:
        raise ValueError('Component count is outside the frozen K1/K2/K4 study.')
    counters = new_counters() if counters is None else counters
    ids = [record['id'] for record in records]
    if len(ids) != len(set(ids)) or set(ids) & set(forbidden_ids or []):
        raise ValueError('Duplicate or forbidden record entered Pressure-corrected fitting.')
    try:
        bank, bank_evidence = balanced_normal_bank(records, forbidden_ids)
        bank_center, bank_scale, scaling = bank_scaling(bank)
    except NormalBankError as exc:
        return None, {'valid': False, 'reason': 'invalid_normal_bank_scaling', 'detail': str(exc),
                      'training_ids': ids, 'components': components, 'classifier_fitted': False}
    mixture = GaussianMixture(n_components=components, **MIXTURE_PARAMETERS)
    counters['gmm_attempted'] += 1; counters['gmm_internal_initializations'] += 1
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with threadpool_limits(limits=2):
            mixture.fit((bank - bank_center) / bank_scale)
    counters['gmm_completed'] += 1
    mixture_evidence = {'parameters': mixture.get_params(deep=False), 'converged': bool(mixture.converged_),
                        'iterations': int(mixture.n_iter_), 'lower_bound': float(mixture.lower_bound_),
                        'weights': mixture.weights_.tolist(), 'means': mixture.means_.tolist(),
                        'variances': mixture.covariances_.tolist(),
                        'warnings': [{'category': warning.category.__name__, 'message': str(warning.message)} for warning in caught],
                        'initialization': 'One internal kmeans initialization inside this top-level GaussianMixture.fit call.'}
    evidence = {'valid': False, 'components': components, 'training_ids': ids,
                'forbidden_ids': sorted(forbidden_ids or []), 'normal_bank': bank_evidence,
                'bank_center': bank_center.tolist(), 'bank_scale': bank_scale.tolist(),
                'pressure_scaling': scaling, 'mixture': mixture_evidence, 'classifier_fitted': False}
    if not mixture.converged_:
        return None, dict(evidence, reason='mixture_nonconvergence')
    counters['gmm_converged'] += 1
    if not _valid_mixture(mixture):
        return None, dict(evidence, reason='mixture_nonfinite_or_invalid_variance')
    regime = {'model': mixture, 'bank_center': bank_center, 'bank_scale': bank_scale}
    designs = []; appended = []; targets = []; indices_by_record = []; raw_values = []; counts = {}; training_audits = {}
    for record in records:
        X, indices, _ = trend_features(record)
        values = record['values'][indices]; labels = np.asarray(record['labels'], int)[indices]
        extra = regime_features(regime, values)
        designs.append(X); appended.append(extra)
        targets.append(labels); indices_by_record.append(indices)
        raw_values.append(values); counts[record['id']] = len(indices)
        training_audits[record['id']] = {
            'all_eligible': numerical_audit(regime, values, installed_nll=extra[:, 0]),
            'labelled_normal': numerical_audit(regime, values[labels == 0])}
    base = np.concatenate(designs); extra = np.concatenate(appended); y = np.concatenate(targets)
    if len(np.unique(y)) != 2:
        raise ValueError('Both pooled classifier training classes are required.')
    base_center = np.median(base, axis=0)
    base_scale = np.maximum(np.quantile(base, .75, axis=0) - np.quantile(base, .25, axis=0),
                            np.maximum(base.std(axis=0) * .05, 1e-6))
    extra_center = np.median(extra, axis=0)
    extra_scale = np.maximum(np.quantile(extra, .75, axis=0) - np.quantile(extra, .25, axis=0), 1e-6)
    center = np.r_[base_center, extra_center]; scale = np.r_[base_scale, extra_scale]
    X = np.column_stack([base, extra]); sensors = np.concatenate(raw_values)
    weights, weighting = sample_weights(records, indices_by_record, 'record_equal_class_balanced')
    sensor_scale = np.maximum(np.quantile(sensors, .75, axis=0) - np.quantile(sensors, .25, axis=0),
                              np.maximum(sensors.std(axis=0) * .05, 1e-6))
    audits = {'bank': numerical_audit(regime, bank), 'training': numerical_audit(regime, sensors),
              'training_normal': numerical_audit(regime, sensors[y == 0])}
    classifier = HistGradientBoostingClassifier(**PARAMETERS, random_state=SEED)
    counters['hgb_attempted'] += 1
    with threadpool_limits(limits=2):
        classifier.fit((X - center) / scale, y, sample_weight=weights)
    counters['hgb_completed'] += 1
    bundle = {'model': classifier, 'regime': regime, 'arm': f'K{components}', 'components': components,
              'feature_center': center, 'feature_scale': scale, 'sensor_scale': sensor_scale,
              'feature_names': feature_names(), 'feature_count': 198, 'threshold': .5, 'window': 2,
              'algorithm': 'GaussianMixtureNormalFeatures+HistGradientBoostingClassifier',
              'version': 'pressure-scaled-normal-regimes-13.0.0', 'objective': 'supervised_known_anomaly_classification'}
    evidence.update(valid=True, classifier_fitted=True, training_rows=len(y),
                    training_rows_by_recording=counts, normal_rows=int((y == 0).sum()), anomalous_rows=int((y == 1).sum()),
                    feature_count=198, feature_center=center.tolist(), feature_scale=scale.tolist(),
                    classifier_parameters=classifier.get_params(deep=False), sample_weights=weighting,
                    numerical_audits=audits, training_numerical_audits_by_recording=training_audits,
                    numerically_accepted=(all(a['numerically_accepted'] for a in audits.values()) and
                                          all(a['numerically_accepted'] for group in training_audits.values() for a in group.values())))
    return bundle, evidence


def check_v12_fold_integrity(detail: dict, reference: dict, outputs: dict, reference_oof: dict) -> dict:
    if detail['normal_bank'] != reference['normal_bank']:
        raise ValueError('The corrected model changed the preserved V12 normal-bank selection or source identities.')
    if not np.array_equal(detail['bank_center'], reference['bank_center']):
        raise ValueError('The corrected model changed preserved V12 bank centers.')
    pressure = next(i for i, sensor in enumerate(SENSORS) if sensor['key'] == 'Pressure')
    other = np.arange(8) != pressure
    scale = np.asarray(detail['bank_scale']); previous = np.asarray(reference['bank_scale'])
    if not np.array_equal(scale[other], previous[other]):
        raise ValueError('One of the seven protected non-Pressure bank scales changed.')
    for name in ['feature_center', 'feature_scale']:
        if not np.array_equal(detail[name][:186], reference[name][:186]):
            raise ValueError('The corrected model changed original186 classifier preprocessing.')
    unchanged = bool(scale[pressure] == previous[pressure]); differences = {}
    for key, values in outputs.items():
        old = reference_oof[key]
        if values.shape != old.shape or not np.array_equal(np.isfinite(values), np.isfinite(old)):
            raise ValueError('V13 and preserved V12 score availability mask differs.')
        mask = np.isfinite(old)
        differences[key] = float(np.max(np.abs(values[mask] - old[mask]))) if mask.any() else 0.
        if unchanged and not np.allclose(values[mask], old[mask], atol=1e-12, rtol=0):
            raise ValueError('An unchanged Pressure scale did not reproduce the preserved same-K V12 scores.')
    return {'normal_bank_identical': True, 'bank_centers_identical': True,
            'other_seven_scales_identical': True, 'original186_scalers_identical': True,
            'finite_mask_identical': True, 'pressure_scale_unchanged': unchanged,
            'unchanged_fold_score_identity_passed': True if unchanged else None,
            'same_k_max_absolute_score_difference_by_recording': differences,
            'score_identity_tolerance': {'atol': 1e-12, 'rtol': 0.}}


def fit_outer_arm(records: list[dict], anchor: dict, folds: list[dict], components: int,
                  counters: dict, expected_base_folds: list[dict] | None = None,
                  expected_v12_folds: list[dict] | None = None, reference_oof: dict | None = None) -> dict:
    by_id = {record['id']: record for record in records}; expected = set(by_id)
    before = counters.copy(); outputs = {}; evidence = []
    for position, fold in enumerate(folds):
        if set(fold['train_ids']) & set(fold['heldout_ids']) or set(fold['train_ids']) | set(fold['heldout_ids']) != expected:
            raise ValueError('Invalid original whole-recording outer fold.')
        fitting = [anchor] + [by_id[key] for key in fold['train_ids']]
        bundle, detail = fit_model(fitting, components, forbidden_ids=fold['heldout_ids'], counters=counters)
        detail = dict(detail, fold=fold['fold'], outer_train_ids=fold['train_ids'], outer_heldout_ids=fold['heldout_ids'])
        evidence.append(detail)
        if bundle is None:
            attempted = _fit_call_count(counters) - _fit_call_count(before)
            return {'valid': False, 'reason': detail['reason'], 'scores': {}, 'discarded_scores': outputs,
                    'folds': evidence, 'numerically_accepted': False,
                    'skipped_folds': [item['fold'] for item in folds[position + 1:]],
                    'discarded_partial_record_ids': sorted(outputs),
                    'skipped_top_level_fit_calls': 2 * len(folds) - attempted,
                    'counters': {key: counters[key] - before[key] for key in counters}}
        if expected_base_folds is not None:
            old = next(item['B_fit'] for item in expected_base_folds if item['fold'] == fold['fold'])
            np.testing.assert_array_equal(bundle['feature_center'][:186], old['feature_center'])
            np.testing.assert_array_equal(bundle['feature_scale'][:186], old['feature_scale'])
        detail['train_normal_density'] = normal_density_summary(bundle, fitting)
        heldout = [by_id[key] for key in fold['heldout_ids']]
        detail['heldout_normal_density'] = normal_density_summary(bundle, heldout)
        held_values = []; held_labels = []; per_record_audits = {}
        fold_outputs = {}
        for record in heldout:
            indices = base_features(record)[1]
            values = record['values'][indices]; labels = np.asarray(record['labels'])[indices]
            held_values.append(values); held_labels.append(labels)
            per_record_audits[record['id']] = {
                'all_eligible': numerical_audit(bundle['regime'], values),
                'labelled_normal': numerical_audit(bundle['regime'], values[labels == 0])}
            fold_outputs[record['id']] = classifier_scores(bundle, record)['scores']
        values = np.concatenate(held_values); labels = np.concatenate(held_labels)
        detail['numerical_audits'].update(heldout=numerical_audit(bundle['regime'], values),
                                         heldout_normal=numerical_audit(bundle['regime'], values[labels == 0]))
        detail['heldout_numerical_audits_by_recording'] = per_record_audits
        detail['numerically_accepted'] = (
            all(a['numerically_accepted'] for a in detail['numerical_audits'].values()) and
            all(a['numerically_accepted'] for audits in detail.get('training_numerical_audits_by_recording', {}).values()
                for a in audits.values()) and
            all(a['numerically_accepted'] for audits in per_record_audits.values() for a in audits.values()))
        if expected_v12_folds is not None:
            old = next(item for item in expected_v12_folds if item['fold'] == fold['fold'])
            detail['v12_integrity'] = check_v12_fold_integrity(detail, old, fold_outputs, reference_oof)
        outputs.update(fold_outputs)
        print(json.dumps({'arm': f'K{components}', 'outer_fold_complete': fold['fold'], 'counters': counters,
                          'numerically_accepted': detail['numerically_accepted'],
                          'pressure_scale': detail['pressure_scaling']['new_pressure_scale']}), flush=True)
    return {'valid': True, 'scores': outputs, 'folds': evidence, 'skipped_folds': [], 'discarded_scores': {},
            'numerically_accepted': all(f['numerically_accepted'] for f in evidence),
            'discarded_partial_record_ids': [], 'skipped_top_level_fit_calls': 0,
            'counters': {key: counters[key] - before[key] for key in counters}}


def quality_decision(selected: dict | None, arms: dict) -> dict:
    if selected is None:
        return {'development_selected': None, 'selected': None, 'status': 'no_feasible_candidate'}
    if not arms[selected['arm']]['numerically_accepted']:
        return {'development_selected': selected, 'selected': None, 'status': 'numerical_check_rejected',
                'numerical_rejection': 'The sole original-quality winner failed fixed numerical checks; no alternative winner.'}
    return {'development_selected': selected, 'selected': selected, 'status': 'selected_development_candidate'}


def fit_final_candidate(target: Path, records: list[dict], selected: dict, counters: dict,
                        plan_sha256: str, selection_sha256: str) -> dict:
    if _hash(target / 'selection_before_final_fit.json') != selection_sha256:
        raise ValueError('Pre-final selection identity changed before conditional final fitting.')
    bundle, evidence = fit_model(records, selected['components'], counters=counters)
    _write(target / 'final_fit.json', evidence)
    result = {'final_fit': evidence, 'final_fit_sha256': _hash(target / 'final_fit.json'),
              'final_model_available': False}
    if bundle is None or not evidence['numerically_accepted']:
        return dict(result, selected=None, status='final_fit_rejected',
                    final_fit_rejection=evidence['reason'] if bundle is None else 'final_numerical_check_failed')
    bundle.update(threshold=selected['threshold'], plan_sha256=plan_sha256,
                  selection_before_final_fit_sha256=selection_sha256)
    if bundle['arm'] != selected['arm'] or len(bundle['feature_names']) != 198:
        raise ValueError('The sole final model differs from selected component/feature identity.')
    joblib.dump(bundle, target / 'selected_bundle.joblib', compress=3)
    return dict(result, selected=selected, status='selected_development_candidate', final_model_available=True,
                model_sha256=_hash(target / 'selected_bundle.joblib'))


def _delay_rank(point: dict) -> tuple:
    metric = point['metrics']
    return (metric['median_delay_seconds'] if metric['median_delay_seconds'] is not None else float('inf'),
            metric['mean_delay_seconds'] if metric['mean_delay_seconds'] is not None else float('inf'),
            point['threshold_order'])


def descriptive_frontiers(points: list[dict]) -> dict:
    progressive = [p for p in points if not p['lost_progressive_event_ids'] and p['metrics']['detected_events'] >= 26]
    burden = [p for p in points if p['metrics']['false_episodes'] < 3 and
              p['metrics']['false_episodes_per_hour'] <= 2 and p['metrics']['false_alert_seconds'] <= 37]
    return {
        'progressive_recall_frontier': min(progressive, key=lambda p: (
            p['metrics']['false_episodes'], p['metrics']['false_alert_seconds'], -p['metrics']['detected_events'],
            *_delay_rank(p))) if progressive else None,
        'burden_limited_recall_frontier': min(burden, key=lambda p: (
            -p['metrics']['detected_events'], p['metrics']['false_episodes'], p['metrics']['false_alert_seconds'],
            *_delay_rank(p))) if burden else None,
    }


def _paired_frontiers(old: dict | None, new: dict | None) -> dict:
    if old is None or new is None:
        return {'comparable': False, 'v12_available': old is not None, 'v13_available': new is not None}
    old_rows = {row['id']: row for row in old['per_recording']}
    new_rows = {row['id']: row for row in new['per_recording']}
    return {'comparable': True,
            'per_recording': [{'id': key, 'group': row['group'],
                               'v12_false_episodes': old_rows[key]['metrics']['false_episodes'],
                               'v13_false_episodes': row['metrics']['false_episodes'],
                               'v12_normal_alert_seconds': old_rows[key]['metrics']['false_alert_seconds'],
                               'v13_normal_alert_seconds': row['metrics']['false_alert_seconds']}
                              for key, row in new_rows.items()],
            'v12_family': old['group_slices'], 'v13_family': new['group_slices'],
            'v12_progressive': old['progressive_subset'], 'v13_progressive': new['progressive_subset'],
            'interpretation': 'Descriptive predeclared frontiers at their own operating cutoffs, not a new model selection.'}


def same_k_comparisons(points: list[dict], old_points: list[dict]) -> dict:
    comparisons = {}
    for components in COMPONENTS:
        arm = f'K{components}'
        old = descriptive_frontiers([point for point in old_points if point['arm'] == arm])
        new = descriptive_frontiers([point for point in points if point['arm'] == arm])
        comparisons[arm] = {'v12': old, 'v13': new,
                            'paired': {key: _paired_frontiers(old[key], new[key]) for key in old}}
    return comparisons


def study_plan(root: Path) -> dict:
    root = Path(root); prior = root / 'artifacts/regime-development'
    plan = _read(prior / 'plan.json'); report = _read(prior / 'report.json')
    if _hash(prior / 'plan.json') != report['plan_sha256']:
        raise ValueError('Frozen V12 source plan hash mismatch.')
    if any(_hash(root / name) != digest for name, digest in plan['dependencies'].items()):
        raise ValueError('A preserved V12 or earlier dependency changed.')
    for name, digest in report['score_artifacts_sha256'].items():
        if _hash(prior / name) != digest:
            raise ValueError('A preserved V12 OOF array changed.')
    for name, expected in [('candidate_audits.json', report['candidate_audits_sha256']),
                           ('selection_before_final_fit.json', report['selection_before_final_fit_sha256'])]:
        if _hash(prior / name) != expected:
            raise ValueError('Preserved V12 candidate/selection identity changed.')
    for arm, expected in report['fit_evidence_sha256'].items():
        if _hash(prior / f'{arm}_fit_evidence.json') != expected:
            raise ValueError('Preserved V12 fit evidence changed.')
    paths = list(plan['dependencies']) + [PROTOCOL, 'signalwatch/ml_pressure_scaled.py',
             'tests/test_ml_pressure_scaled.py', 'scripts/study_pressure_scaled.py',
             'signalwatch/trend_diagnostics.py', 'tests/test_trend_diagnostics.py', 'scripts/check_trend_model.py']
    paths += [str(path.relative_to(root)).replace('\\', '/') for path in sorted(prior.iterdir()) if path.is_file()]
    return dict(plan, study='pressure_scaled_normal_regimes_v13', protocol_path=PROTOCOL,
                protocol_sha256=_hash(root / PROTOCOL),
                dependencies={name: _hash(root / name) for name in dict.fromkeys(paths)},
                preserved_v12_plan_sha256=report['plan_sha256'],
                preserved_v12_report_sha256=_hash(prior / 'report.json'),
                normal_bank=dict(plan['normal_bank'], sensor_scale='Pressure=max(bankIQR,bankstd_ddof0,1e-6); other7=max(bankIQR,1e-6)'),
                numerical_preflight={'pressure_key': 'Pressure', 'max_bank_abs_pressure_z': MAX_PRESSURE_Z},
                numerical_flags={'max_absolute_nll_difference': MAX_NLL_DIFFERENCE,
                                 'variance_tolerance_below_reg_covar': VARIANCE_TOLERANCE,
                                 'summaries': ['count', 'min', 'median', 'p95', 'p99', 'max'],
                                 'policy': 'Original quality rank first; reject sole winner if numerical flags fail, no fallback.'},
                unchanged_fold_check={'atol': 1e-12, 'rtol': 0., 'finite_mask': 'exact'},
                v12_comparisons='Saved same-K predeclared progressive/recall and burden-limited frontiers; no new V12 scan.')


def freeze_study(root: Path) -> dict:
    root = Path(root); target = root / TARGET; target.mkdir(parents=True, exist_ok=True)
    plan = study_plan(root); path = target / 'plan.json'
    if path.exists():
        if _read(path) != plan:
            raise ValueError('Frozen Pressure-scaling protocol or dependency identity changed.')
    else:
        _write(path, plan)
    return plan


def run_study(root: Path) -> dict:
    root = Path(root); target = root / TARGET
    if (target / 'execution_started.json').exists() or (target / 'report.json').exists():
        raise ValueError('One-shot Pressure-scaling study has already started; never silently retry.')
    started = time.perf_counter(); plan = freeze_study(root)
    active = {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}
    _write(target / 'execution_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
            'dependencies': plan['dependencies'], 'active_hashes': active,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
            'authorization': 'One bounded Pressure-scaling study and conditional sole final GMM/HGB fit; no diagnostic inference.'})
    records, anchor = _load_development(root, plan)
    source = root / 'artifacts/v5/development'; source_report = _read(source / 'report.json')
    fusion = _arrays(source / 'selected_oof_scores.npz'); classifier_A = _arrays(source / 'classifier_oof_scores.npz')
    v12 = root / 'artifacts/regime-development'; v12_report = _read(v12 / 'report.json')
    old_scores = {f'K{k}': _arrays(v12 / f'K{k}_oof_scores.npz') for k in COMPONENTS}
    saved_controls = {'A120': classifier_A,
                      'V10B186': _arrays(root / 'artifacts/trend-development/B_oof_scores.npz'),
                      'V11HGB31': _arrays(root / 'artifacts/trend-backbones-development/HGB31_oof_scores.npz'),
                      'V11ExtraTrees256': _arrays(root / 'artifacts/trend-backbones-development/ExtraTrees256_oof_scores.npz'),
                      **{'V12' + name: values for name, values in old_scores.items()}}
    expected = set(plan['development_ids'])
    if any(set(values) != expected for values in [fusion, *saved_controls.values()]):
        raise ValueError('Saved source controls differ from exact development ID set.')
    for record in records:
        _, indices, _ = trend_features(record); eligible = np.zeros(len(record['values']), bool); eligible[indices] = True
        if any(values[record['id']].shape != eligible.shape or not np.array_equal(np.isfinite(values[record['id']]), eligible)
               for values in [fusion, *saved_controls.values()]):
            raise ValueError('Saved source controls differ from original eligible rows.')
    original, _ = _reports(records, fusion, 1.)
    if original != source_report['metrics']:
        raise ValueError('Original V5 development operating point does not reproduce.')
    v10_report = _read(root / 'artifacts/trend-development/report.json')
    counters = new_counters(); valid_scores = {}; arms = {}; fit_files = {}
    for components in COMPONENTS:
        arm = f'K{components}'
        result = fit_outer_arm(records, anchor, plan['folds'], components, counters,
                               expected_base_folds=v10_report['folds'],
                               expected_v12_folds=v12_report['arms'][arm]['folds'], reference_oof=old_scores[arm])
        scores = result.pop('scores'); discarded = result.pop('discarded_scores')
        if result['valid']:
            if set(scores) != expected:
                raise ValueError('A supposedly valid arm lacks complete OOF predictions.')
            for key, score in scores.items():
                if not np.array_equal(np.isfinite(score), np.isfinite(fusion[key])):
                    raise ValueError('Pressure-corrected classifier changed finite scored exposure.')
            valid_scores[arm] = scores
            np.savez_compressed(target / f'{arm}_oof_scores.npz', **scores)
        elif discarded:
            np.savez_compressed(target / f'{arm}_discarded_partial_oof_scores.npz', **discarded)
        arms[arm] = result
        _write(target / f'{arm}_fit_evidence.json', result)
        fit_files[arm] = _hash(target / f'{arm}_fit_evidence.json')
        if counters['gmm_attempted'] > 9 or counters['hgb_attempted'] > 9 or _fit_call_count(counters) > 18:
            raise ValueError('Outer estimator fit budget exceeded.')
    development_counters = counters.copy()
    evaluated = evaluate_candidates(records, valid_scores, fusion, classifier_A, source_report['selected']['classifier_threshold'])
    points = evaluated.pop('candidate_audits'); _write(target / 'candidate_audits.json', points)
    winner = evaluated.pop('selected'); selection_path = target / 'selection_before_final_fit.json'
    freeze_selection(target, winner, _hash(target / 'plan.json'), list(valid_scores), _fit_call_count(counters))
    report = {'schema_version': 1, 'scope': 'pressure_scaled_normal_regime_grouped_oof_development_selection',
              'final_model_available': False,
              'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
              'selection_before_final_fit_sha256': _hash(selection_path),
              'candidate_audits_sha256': _hash(target / 'candidate_audits.json'),
              'arms': arms, 'development_counters': development_counters,
              'development_skipped_fit_calls': 18 - _fit_call_count(development_counters),
              'fit_evidence_sha256': fit_files, 'feature_count': 198,
              'control_source_ids': plan['development_ids'],
              'saved_control_probability_diagnostics': {name: probability_diagnostics(records, values) for name, values in saved_controls.items()},
              'v12_same_k_comparisons': same_k_comparisons(points, _read(v12 / 'candidate_audits.json')),
              'old_test_scored': False, 'teaser_scored': False, 'activation': False,
              'limits': ['Repeated development selection with benchmark-normal labels, not fresh validation or verified physical safety.',
                         'Scaling changes effective Pressure regularization and geometry as well as conditioning.',
                         'Numerical checks are separate from quality rank; no alternate winner after numerical rejection.',
                         'No normalization of test streams, clipping, mixture retries, delay rule or residual OR.'],
              **evaluated, **quality_decision(winner, arms)}
    if report['selected'] is not None:
        report.update(fit_final_candidate(target, [anchor] + records, report['selected'], counters,
                                          report['plan_sha256'], report['selection_before_final_fit_sha256']))
    else:
        report['final_fit'] = None
    if _fit_call_count(counters) > 20 or _fit_call_count(counters) - _fit_call_count(development_counters) > 2:
        raise ValueError('Conditional estimator fit budget exceeded.')
    if active != {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}:
        raise ValueError('Protected active V5 artifacts changed.')
    if any(_hash(root / name) != digest for name, digest in plan['dependencies'].items()):
        raise ValueError('A frozen source, prior artifact or data dependency changed during the Pressure-scaling study.')
    report.update(counters=counters, actual_top_level_fit_calls=_fit_call_count(counters),
                  active_files_unchanged=True, dependencies_unchanged=True,
                  score_artifacts_sha256={path.name: _hash(path) for path in target.glob('*.npz')},
                  study_seconds=round(time.perf_counter() - started, 3))
    _write(target / 'report.json', report)
    return report
