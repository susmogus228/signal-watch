"""Training-only normal-regime features followed by an immediate learned classifier."""
from __future__ import annotations

import hashlib
import json
import platform
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits

from signalwatch.dataset import SENSORS, SNAPSHOT, manifest
from signalwatch.ml_ablation import probability_diagnostics
from signalwatch.ml_supervised import causal_features as base_features
from signalwatch.ml_supervised_weighted import sample_weights
from signalwatch.ml_trends import (
    ACTIVE_FILES, PARAMETERS, PROGRESSIVE_EVENTS, QUANTILES, _load_development,
    causal_features as trend_features, feature_names as trend_feature_names,
    screen_candidate, _reports, _events, _groups,
)

COMPONENTS = [1, 2, 4]
PROTOCOL = 'docs/ML_V12_NORMAL_REGIME_PLAN.md'
SEED = 20261023
MIXTURE_PARAMETERS = {'covariance_type': 'diag', 'tol': .001, 'reg_covar': .001,
                      'max_iter': 200, 'n_init': 1, 'init_params': 'kmeans',
                      'random_state': SEED, 'warm_start': False}


class NormalBankError(ValueError):
    """An invalid balanced training normal bank invalidates a component arm."""


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _plain(value):
    if isinstance(value, dict): return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_plain(item) for item in value]
    if isinstance(value, np.ndarray): return _plain(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value): return None
    if isinstance(value, np.generic): return value.item()
    return value


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, allow_nan=False), encoding='utf-8')


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def new_counters() -> dict:
    return {'gmm_attempted': 0, 'gmm_completed': 0, 'gmm_converged': 0,
            'gmm_internal_initializations': 0, 'hgb_attempted': 0, 'hgb_completed': 0}


def feature_names() -> list[str]:
    return trend_feature_names() + ['normal_negative_log_density', 'normal_min_mahalanobis_squared',
            'normal_posterior_mean_mahalanobis_squared', 'normal_max_posterior'] + [
            'normal_posterior_signed_standardized_residual:' + sensor['key'] for sensor in SENSORS]


def balanced_normal_bank(records: list[dict], forbidden_ids: list[str] | None = None) -> tuple[np.ndarray, dict]:
    ids = [record['id'] for record in records]
    if len(ids) != len(set(ids)) or set(ids) & set(forbidden_ids or []):
        raise ValueError('Duplicate or forbidden record entered normal-bank construction.')
    eligible = {}; normal_counts = {}
    for record in records:
        if record.get('labels') is None or not np.isin(record['labels'], [0, 1]).all():
            raise ValueError('Normal-bank construction requires explicit binary training labels.')
        indices = base_features(record)[1]
        normal = indices[np.asarray(record['labels'])[indices] == 0]
        if not len(normal):
            raise NormalBankError('Every training record must provide eligible normal observations.')
        eligible[record['id']] = normal; normal_counts[record['id']] = len(normal)
    if not records:
        raise NormalBankError('A normal bank requires training records.')
    m = min(512, min(normal_counts.values()))
    if m < 2:
        raise NormalBankError('Every training record needs at least two eligible normal observations.')
    selected = {}; parts = []
    for record in records:
        indices = eligible[record['id']]
        positions = np.floor(np.linspace(0, len(indices) - 1, m)).astype(int)
        chosen = indices[positions]
        if len(np.unique(chosen)) != m:
            raise ValueError('The frozen bank sampling rule generated duplicate positions.')
        selected[record['id']] = chosen.tolist(); parts.append(np.asarray(record['values'], float)[chosen])
    bank = np.concatenate(parts)
    if bank.shape != (len(records) * m, 8) or not np.isfinite(bank).all():
        raise ValueError('Balanced normal-bank sensor values must be finite eight-column rows.')
    return bank, {'m': m, 'rows': len(bank), 'normal_counts': normal_counts,
                  'selected_indices': selected, 'training_ids': ids,
                  'source_hashes': {record['id']: record.get('sha256') for record in records},
                  'forbidden_ids': sorted(forbidden_ids or []),
                  'bank_sha256_float64': hashlib.sha256(np.ascontiguousarray(bank).tobytes()).hexdigest(),
                  'sampling': 'I_r[floor(linspace(0,n_r-1,m))], equal m perrecord including anchor, no replacement'}


def _valid_mixture(model) -> bool:
    values = [model.weights_, model.means_, model.covariances_, model.precisions_cholesky_]
    return (all(np.isfinite(value).all() for value in values) and np.isfinite(model.lower_bound_)
            and np.all(model.covariances_ > 0) and np.all(model.weights_ >= 0)
            and float(model.weights_.sum()) > 0)


def regime_features(regime: dict, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float); model = regime['model']
    if values.ndim != 2 or values.shape[1] != 8 or not np.isfinite(values).all():
        raise ValueError('Regime features require finite eight-sensor rows.')
    if not _valid_mixture(model):
        raise ValueError('Normal mixture has non-finite parameters or non-positive variance.')
    if not np.isfinite(regime['bank_center']).all() or not np.isfinite(regime['bank_scale']).all() or np.any(regime['bank_scale'] <= 0):
        raise ValueError('Normal-bank sensor scaling is invalid.')
    z = (values - regime['bank_center']) / regime['bank_scale']
    with threadpool_limits(limits=2):
        posterior = model.predict_proba(z)
        negative_density = -model.score_samples(z)
    signed = (z[:, None, :] - model.means_[None, :, :]) / np.sqrt(model.covariances_[None, :, :])
    squared = np.sum(signed * signed, axis=2)
    features = np.column_stack([negative_density, squared.min(axis=1), np.sum(posterior * squared, axis=1),
                                posterior.max(axis=1), np.sum(posterior[:, :, None] * signed, axis=1)])
    if features.shape != (len(values), 12) or not np.isfinite(features).all():
        raise ValueError('Normal-regime features exceed the finite numerical range.')
    return features


def fit_model(records: list[dict], components: int, forbidden_ids: list[str] | None = None,
              counters: dict | None = None) -> tuple[dict | None, dict]:
    if components not in COMPONENTS:
        raise ValueError('Component count is outside the frozen K1/K2/K4 study.')
    counters = new_counters() if counters is None else counters
    ids = [record['id'] for record in records]
    if len(ids) != len(set(ids)) or set(ids) & set(forbidden_ids or []):
        raise ValueError('Duplicate or forbidden record entered regime model fitting.')
    try:
        bank, bank_evidence = balanced_normal_bank(records, forbidden_ids)
    except NormalBankError as exc:
        return None, {'valid': False, 'reason': 'invalid_normal_bank', 'detail': str(exc),
                      'training_ids': ids, 'components': components, 'classifier_fitted': False}
    bank_center = np.median(bank, axis=0)
    bank_scale = np.maximum(np.quantile(bank, .75, axis=0) - np.quantile(bank, .25, axis=0), 1e-6)
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
                'mixture': mixture_evidence, 'classifier_fitted': False}
    if not mixture.converged_:
        evidence['reason'] = 'mixture_nonconvergence'
        return None, evidence
    counters['gmm_converged'] += 1
    if not _valid_mixture(mixture):
        evidence['reason'] = 'mixture_nonfinite_or_invalid_variance'
        return None, evidence
    regime = {'model': mixture, 'bank_center': bank_center, 'bank_scale': bank_scale}
    designs = []; appended = []; targets = []; indices_by_record = []; raw_values = []; counts = {}
    for record in records:
        X, indices, _ = trend_features(record)
        designs.append(X); appended.append(regime_features(regime, record['values'][indices]))
        targets.append(np.asarray(record['labels'], int)[indices]); indices_by_record.append(indices)
        raw_values.append(record['values'][indices]); counts[record['id']] = len(indices)
    base = np.concatenate(designs); extra = np.concatenate(appended); y = np.concatenate(targets)
    if len(np.unique(y)) != 2:
        raise ValueError('Both pooled classifier training classes are required.')
    base_center = np.median(base, axis=0)
    base_scale = np.maximum(np.quantile(base, .75, axis=0) - np.quantile(base, .25, axis=0),
                            np.maximum(base.std(axis=0) * .05, 1e-6))
    extra_center = np.median(extra, axis=0)
    extra_scale = np.maximum(np.quantile(extra, .75, axis=0) - np.quantile(extra, .25, axis=0), 1e-6)
    center = np.r_[base_center, extra_center]; scale = np.r_[base_scale, extra_scale]
    X = np.column_stack([base, extra])
    weights, weighting = sample_weights(records, indices_by_record, 'record_equal_class_balanced')
    sensors = np.concatenate(raw_values)
    sensor_scale = np.maximum(np.quantile(sensors, .75, axis=0) - np.quantile(sensors, .25, axis=0),
                              np.maximum(sensors.std(axis=0) * .05, 1e-6))
    classifier = HistGradientBoostingClassifier(**PARAMETERS, random_state=SEED)
    counters['hgb_attempted'] += 1
    with threadpool_limits(limits=2):
        classifier.fit((X - center) / scale, y, sample_weight=weights)
    counters['hgb_completed'] += 1
    bundle = {'model': classifier, 'regime': regime, 'arm': f'K{components}', 'components': components,
              'feature_center': center, 'feature_scale': scale, 'sensor_scale': sensor_scale,
              'feature_names': feature_names(), 'feature_count': 198, 'threshold': .5, 'window': 2,
              'algorithm': 'GaussianMixtureNormalFeatures+HistGradientBoostingClassifier',
              'version': 'learned-normal-regimes-12.0.0', 'objective': 'supervised_known_anomaly_classification'}
    evidence.update(valid=True, classifier_fitted=True, training_rows=len(y),
                    training_rows_by_recording=counts, normal_rows=int((y == 0).sum()), anomalous_rows=int((y == 1).sum()),
                    feature_count=198, feature_center=center.tolist(), feature_scale=scale.tolist(),
                    classifier_parameters=classifier.get_params(deep=False), sample_weights=weighting)
    return bundle, evidence


def classifier_scores(bundle: dict, recording: dict) -> dict:
    base, indices, reference = trend_features(recording)
    scores = np.full(len(recording['values']), np.nan)
    contributions = np.full((len(scores), 8), np.nan)
    if len(indices):
        extra = regime_features(bundle['regime'], recording['values'][indices])
        X = np.column_stack([base, extra])
        with threadpool_limits(limits=2):
            probability = bundle['model'].predict_proba((X - bundle['feature_center']) / bundle['feature_scale'])
        positive = np.flatnonzero(bundle['model'].classes_ == 1)
        if len(positive) != 1:
            raise ValueError('Classifier requires a unique positive known-anomaly class.')
        scores[indices] = probability[:, positive[0]]
        contributions[indices] = np.abs(reference) / bundle['sensor_scale']
    return {'scores': scores, 'contributions': contributions, 'warmup_samples': 2,
            'threshold': float(bundle['threshold'])}


def _summary(values: np.ndarray) -> dict:
    values = np.asarray(values, float)
    if not len(values):
        return {'count': 0, 'min': None, 'q25': None, 'median': None, 'q75': None, 'max': None}
    return {'count': len(values), 'min': float(values.min()), 'q25': float(np.quantile(values, .25)),
            'median': float(np.median(values)), 'q75': float(np.quantile(values, .75)), 'max': float(values.max())}


def normal_density_summary(bundle: dict, records: list[dict]) -> dict:
    rows = {}; pieces = []
    for record in records:
        indices = base_features(record)[1]
        normal = indices[np.asarray(record['labels'])[indices] == 0]
        features = regime_features(bundle['regime'], record['values'][normal]) if len(normal) else np.empty((0, 12))
        rows[record['id']] = {'negative_log_density': _summary(features[:, 0]),
                             'min_mahalanobis_squared': _summary(features[:, 1])}
        pieces.append(features)
    combined = np.concatenate(pieces) if pieces else np.empty((0, 12))
    return {'per_recording': rows, 'pooled': {'negative_log_density': _summary(combined[:, 0]),
            'min_mahalanobis_squared': _summary(combined[:, 1])},
            'scope': 'Descriptive labelled-normal rows under fixed fitted transforms; never a fitted correction or selection gate.'}


def fit_outer_arm(records: list[dict], anchor: dict, folds: list[dict], components: int,
                  counters: dict, expected_base_folds: list[dict] | None = None) -> dict:
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
            attempted = (counters['gmm_attempted'] - before['gmm_attempted']) + (counters['hgb_attempted'] - before['hgb_attempted'])
            return {'valid': False, 'reason': detail['reason'], 'scores': {}, 'discarded_scores': outputs, 'folds': evidence,
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
        for record in heldout:
            outputs[record['id']] = classifier_scores(bundle, record)['scores']
        print(json.dumps({'arm': f'K{components}', 'outer_fold_complete': fold['fold'], 'counters': counters}), flush=True)
    return {'valid': True, 'scores': outputs, 'folds': evidence, 'skipped_folds': [],
            'discarded_scores': {},
            'discarded_partial_record_ids': [], 'skipped_top_level_fit_calls': 0,
            'counters': {key: counters[key] - before[key] for key in counters}}


def select_candidate(points: list[dict]) -> dict | None:
    feasible = [point for point in points if point['feasible']]
    def rank(point):
        metric = point['metrics']
        return (-metric['detected_events'], metric['false_episodes'], metric['false_alert_seconds'],
                metric['median_delay_seconds'] if metric['median_delay_seconds'] is not None else float('inf'),
                metric['mean_delay_seconds'] if metric['mean_delay_seconds'] is not None else float('inf'),
                COMPONENTS.index(int(point['arm'][1:])), point['threshold_order'])
    return min(feasible, key=rank) if feasible else None


def evaluate_candidates(records: list[dict], scores_by_arm: dict, reference_scores: dict,
                         classifier_A: dict, classifier_cutoff: float) -> dict:
    reference_metric, reference_reports = _reports(records, reference_scores, 1.)
    reference_events = _events(reference_reports)
    reference_hits = {key for key, event in reference_events.items() if event['status'] == 'detected'}
    if not set(PROGRESSIVE_EVENTS) <= reference_hits:
        raise ValueError('The source progressive reference events are not both detected.')
    classifier_metric, classifier_reports = _reports(records, classifier_A, classifier_cutoff)
    points = []
    for components in COMPONENTS:
        arm = f'K{components}'
        if arm not in scores_by_arm:
            continue
        scores = scores_by_arm[arm]
        if set(scores) != {record['id'] for record in records}:
            raise ValueError('Partial OOF arrays may not enter candidate selection.')
        normal = np.concatenate([scores[record['id']][~record['labels'] & np.isfinite(scores[record['id']])]
                                 for record in records])
        cutoffs = np.unique(np.r_[np.quantile(normal, QUANTILES), np.nextafter(normal.max(), np.inf)])
        cutoffs = cutoffs[np.isfinite(cutoffs) & (cutoffs > 0)]
        for order, cutoff in enumerate(cutoffs):
            metric, reports = _reports(records, scores, float(cutoff)); events = _events(reports)
            hits = {key for key, event in events.items() if event['status'] == 'detected'}
            lost = sorted(reference_hits - hits); gained = sorted(hits - reference_hits)
            lost_progressive = sorted(set(PROGRESSIVE_EVENTS) - hits)
            reasons = screen_candidate(metric, reference_metric, lost_progressive)
            pairs = [{'id': key, 'reference_status': reference_events[key]['status'], 'candidate_status': event['status'],
                      'reference_delay_seconds': reference_events[key]['delay_seconds'],
                      'candidate_delay_seconds': event['delay_seconds'],
                      'reference_first_alert_seconds': reference_events[key]['first_alert_seconds'],
                      'candidate_first_alert_seconds': event['first_alert_seconds']}
                     for key, event in events.items()]
            points.append({'arm': arm, 'components': components, 'threshold': float(cutoff), 'threshold_order': order,
                           'metrics': metric, 'feasible': not reasons, 'rejection_reasons': reasons,
                           'lost_event_ids': lost, 'gained_event_ids': gained,
                           'lost_progressive_event_ids': lost_progressive,
                           'progressive_subset': [row for row in pairs if row['id'] in PROGRESSIVE_EVENTS],
                           'paired_outcomes': pairs, 'group_slices': _groups(records, reports),
                           'per_recording': [{'id': record['id'], 'group': record['group'],
                                              'metrics': report['metrics']['learned'],
                                              'false_onset_seconds': [event['start_seconds'] for event in report['events'] if event['status'] == 'false_alert']}
                                             for record, report in zip(records, reports)]})
    if len(points) > 306:
        raise ValueError('The fixed mixture-arm scan exceeds its306point bound.')
    return {'candidate_audits': points, 'selected': select_candidate(points),
            'reference_fusion': {'metrics': reference_metric, 'group_slices': _groups(records, reference_reports)},
            'reference_classifier_A': {'metrics': classifier_metric, 'threshold': classifier_cutoff,
                                      'group_slices': _groups(records, classifier_reports)},
            'probability_diagnostics': {arm: probability_diagnostics(records, scores) for arm, scores in scores_by_arm.items()}}


def study_plan(root: Path) -> dict:
    root = Path(root); prior = root / 'artifacts/trend-backbones-development'
    v11_plan = _read(prior / 'plan.json'); v11_report = _read(prior / 'report.json')
    if _hash(prior / 'plan.json') != v11_report['plan_sha256']:
        raise ValueError('Frozen V11 source plan hash mismatch.')
    if any(_hash(root / name) != digest for name, digest in v11_plan['dependencies'].items()):
        raise ValueError('A preserved V10/V11 scientific dependency changed.')
    for name, digest in v11_report['score_artifacts_sha256'].items():
        if _hash(prior / name) != digest:
            raise ValueError('A preserved V11 OOF array changed.')
    if _hash(prior / 'candidate_audits.json') != v11_report['candidate_audits_sha256']:
        raise ValueError('The preserved V11 candidate audit changed.')
    paths = list(v11_plan['dependencies']) + [PROTOCOL, 'docs/ML_V12_NORMAL_REGIME_PROPOSAL.md',
            'signalwatch/ml_regimes.py', 'tests/test_ml_regimes.py', 'scripts/study_regimes.py',
            'data/manifest.json', 'data/SKAB-LICENSE', 'data/SKAB-README.md', 'data/SKAB-DATA-README.md']
    paths += [str(path.relative_to(root)).replace('\\', '/') for path in sorted(prior.iterdir()) if path.is_file()]
    entries = {entry['id']: entry for entry in manifest(root)['recordings']}
    for key, expected in v11_plan['source_hashes'].items():
        path = root / entries[key]['path']
        if _hash(path) != expected:
            raise ValueError('A pinned development/normal source CSV changed.')
        paths.append(entries[key]['path'])
    return {'study': 'normal_regime_features_v12', 'schema_version': 1, 'source_snapshot': SNAPSHOT,
            'protocol_path': PROTOCOL, 'protocol_sha256': _hash(root / PROTOCOL),
            'dependencies': {name: _hash(root / name) for name in dict.fromkeys(paths)},
            'development_ids': v11_plan['development_ids'], 'source_hashes': v11_plan['source_hashes'],
            'folds': v11_plan['folds'], 'normal_train_anchor': v11_plan['normal_train_anchor'],
            'feature_count': 198, 'feature_names': feature_names(), 'components': COMPONENTS,
            'normal_bank': {'m': 'min(512,min eligible normal count per trainingrecord)',
                            'indices': 'I_r[floor(linspace(0,n_r-1,m))]', 'minimum_m': 2,
                            'sensor_center': 'bankmedian', 'sensor_scale': 'max(bankIQR,1e-6)',
                            'anchor': 'Same m as every disturbed record; fit-only.'},
            'mixture_parameters': MIXTURE_PARAMETERS,
            'classifier_parameters': dict(PARAMETERS, random_state=SEED),
            'appended_features': ['negative_logdensity', 'minimum_squared_mahalanobis', 'posterior_mean_squared_mahalanobis',
                                  'max_posterior', 'eight_posterior_mean_signed_standardized_residuals'],
            'classifier_scaling': 'Original186 identical V10B; appended12 train-all-rowmedian/IQRfloor1e-6.',
            'max_outer_fit_calls': {'gmm': 9, 'hgb': 9}, 'max_conditional_final_fit_calls': 2, 'max_total_fit_calls': 20,
            'invalid_arm_behavior': 'Invalidate entireK, discard partialOOF fromselection, skipremainingfolds, neverretryparameters.',
            'counter_semantics': 'Attempted invocation increments beforefit; completed means returned; GMM convergence separatelycounted; internal kmeans initialization is not an extra top-levelfit.',
            'threshold_quantiles': QUANTILES, 'additional_threshold': 'nextafter(maxnormal,+inf)', 'maximum_scan_points': 306,
            'constraints': v11_plan['constraints'], 'progressive_event_ids': PROGRESSIVE_EVENTS,
            'ranking': 'Mostevents,fewerfalse,lessnormalalerttime,smallermedian/mean delay,K1/K2/K4,thresholdorder.',
            'inference': 'Immediate rawclass1 probability; frozen GMM/scalers; no labels/IDs or test normalization.',
            'final_failure': 'Preserve development_selected and selectionfile, selectednull, final_fit_rejected, no model or diagnostics.',
            'old_test_access': False, 'teaser_access': False, 'activation': False}


def freeze_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/regime-development'; target.mkdir(parents=True, exist_ok=True)
    plan = study_plan(root); path = target / 'plan.json'
    if path.exists():
        if _read(path) != plan:
            raise ValueError('Frozen regime protocol or dependency identity changed.')
    else:
        _write(path, plan)
    return plan


def _fit_call_count(counters: dict) -> int:
    return counters['gmm_attempted'] + counters['hgb_attempted']


def freeze_selection(target: Path, selected: dict | None, plan_sha256: str,
                      valid_arms: list[str], fit_calls: int) -> dict:
    selection = {'selected': selected, 'plan_sha256': plan_sha256,
                 'candidate_audits_sha256': _hash(target / 'candidate_audits.json'),
                 'oof_hashes': {arm: _hash(target / f'{arm}_oof_scores.npz') for arm in valid_arms},
                 'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'fit_calls_before_final': fit_calls}
    _write(target / 'selection_before_final_fit.json', selection)
    return selection


def run_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/regime-development'; started = time.perf_counter()
    plan = freeze_study(root)
    if (target / 'execution_started.json').exists() or (target / 'report.json').exists():
        raise ValueError('The sole frozen regime study already started; refusing another execution.')
    active = {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}
    _write(target / 'execution_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
            'dependencies': plan['dependencies'], 'active_hashes': active,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__},
            'authorization': 'One bounded K1/K2/K4 outer study and conditional sole final GMM/HGB fit; no diagnostic inference.'})
    records, anchor = _load_development(root, plan)
    source = root / 'artifacts/v5/development'; source_report = _read(source / 'report.json')
    fusion = _arrays(source / 'selected_oof_scores.npz'); classifier_A = _arrays(source / 'classifier_oof_scores.npz')
    saved_controls = {'A120': classifier_A,
                      'V10B186': _arrays(root / 'artifacts/trend-development/B_oof_scores.npz'),
                      'V11HGB31': _arrays(root / 'artifacts/trend-backbones-development/HGB31_oof_scores.npz'),
                      'V11ExtraTrees256': _arrays(root / 'artifacts/trend-backbones-development/ExtraTrees256_oof_scores.npz')}
    expected = set(plan['development_ids'])
    if any(set(values) != expected for values in [fusion, *saved_controls.values()]):
        raise ValueError('Saved source controls differ from the exact development ID set.')
    for record in records:
        _, indices, _ = trend_features(record); eligible = np.zeros(len(record['values']), bool); eligible[indices] = True
        if any(values[record['id']].shape != eligible.shape or not np.array_equal(np.isfinite(values[record['id']]), eligible)
               for values in [fusion, *saved_controls.values()]):
            raise ValueError('Saved source controls differ from original eligible rows.')
    original, _ = _reports(records, fusion, 1.)
    if original != source_report['metrics']:
        raise ValueError('Original V5 development operating point does not reproduce.')
    v10_report = _read(root / 'artifacts/trend-development/report.json')
    counters = new_counters(); valid_scores = {}; arms = {}; density_files = {}
    for components in COMPONENTS:
        arm = f'K{components}'
        result = fit_outer_arm(records, anchor, plan['folds'], components, counters, expected_base_folds=v10_report['folds'])
        scores = result.pop('scores'); discarded = result.pop('discarded_scores')
        if result['valid']:
            if set(scores) != expected:
                raise ValueError('A supposedly valid arm lacks complete OOF predictions.')
            for key, score in scores.items():
                if not np.array_equal(np.isfinite(score), np.isfinite(fusion[key])):
                    raise ValueError('Regime classifier changed finite scored exposure.')
            valid_scores[arm] = scores
            np.savez_compressed(target / f'{arm}_oof_scores.npz', **scores)
        elif discarded:
            np.savez_compressed(target / f'{arm}_discarded_partial_oof_scores.npz', **discarded)
        arms[arm] = result
        _write(target / f'{arm}_fit_evidence.json', result)
        density_files[arm] = _hash(target / f'{arm}_fit_evidence.json')
        if counters['gmm_attempted'] > 9 or counters['hgb_attempted'] > 9 or _fit_call_count(counters) > 18:
            raise ValueError('Outer estimator fit budget exceeded.')
    development_counters = counters.copy()
    evaluated = evaluate_candidates(records, valid_scores, fusion, classifier_A, source_report['selected']['classifier_threshold'])
    points = evaluated.pop('candidate_audits'); _write(target / 'candidate_audits.json', points)
    selected = evaluated['selected']; selection_path = target / 'selection_before_final_fit.json'
    freeze_selection(target, selected, _hash(target / 'plan.json'), list(valid_scores), _fit_call_count(counters))
    report = {'schema_version': 1, 'scope': 'normal_regime_grouped_oof_development_selection',
              'status': 'selected_development_candidate' if selected else 'no_feasible_candidate',
              'development_selected': selected, 'final_model_available': False,
              'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
              'selection_before_final_fit_sha256': _hash(selection_path),
              'candidate_audits_sha256': _hash(target / 'candidate_audits.json'),
              'arms': arms, 'development_counters': development_counters,
              'development_skipped_fit_calls': 18 - _fit_call_count(development_counters),
              'fit_evidence_sha256': density_files, 'feature_count': 198,
              'control_source_ids': plan['development_ids'],
              'saved_control_probability_diagnostics': {name: probability_diagnostics(records, values) for name, values in saved_controls.items()},
              'old_test_scored': False, 'teaser_scored': False, 'activation': False,
              'limits': ['Repeated development selection with benchmark-normal labels, not fresh validation or verified physical safety.',
                         'A trained normal-bank density can distinguish in-sample normality from held-out novelty; density summaries do not correct it.',
                         'No mixture retry, adaptive normalization, temporal policy or residualOR.',
                         'Counters count top-level fit calls; one internal kmeans initialization occurs inside each mixture call.'], **evaluated}
    if selected is not None:
        final_model, evidence = fit_model([anchor] + records, selected['components'], counters=counters)
        _write(target / 'final_fit.json', evidence)
        report['final_fit'] = evidence; report['final_fit_sha256'] = _hash(target / 'final_fit.json')
        if final_model is None:
            report.update(status='final_fit_rejected', selected=None, final_fit_rejection=evidence['reason'])
        else:
            final_model.update(threshold=selected['threshold'], plan_sha256=report['plan_sha256'],
                               selection_before_final_fit_sha256=report['selection_before_final_fit_sha256'])
            joblib.dump(final_model, target / 'selected_bundle.joblib', compress=3)
            report.update(final_model_available=True, model_sha256=_hash(target / 'selected_bundle.joblib'))
    else:
        report['final_fit'] = None
    if _fit_call_count(counters) > 20 or _fit_call_count(counters) - _fit_call_count(development_counters) > 2:
        raise ValueError('Conditional estimator fit budget exceeded.')
    if active != {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}:
        raise ValueError('Protected active V5 artifacts changed.')
    if any(_hash(root / name) != digest for name, digest in plan['dependencies'].items()):
        raise ValueError('A frozen source, prior artifact or data dependency changed during the regime study.')
    report.update(counters=counters, actual_top_level_fit_calls=_fit_call_count(counters),
                  active_files_unchanged=True, dependencies_unchanged=True,
                  score_artifacts_sha256={path.name: _hash(path) for path in target.glob('*.npz')},
                  study_seconds=round(time.perf_counter() - started, 3))
    _write(target / 'report.json', report)
    return report
