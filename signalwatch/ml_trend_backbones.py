"""Bounded tree-learner comparison on the frozen 186-feature trend design."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from signalwatch.ml_ablation import probability_diagnostics
from signalwatch.ml_supervised_weighted import sample_weights
from signalwatch.ml_trends import (
    ACTIVE_FILES, PROGRESSIVE_EVENTS, SNAPSHOT, causal_features, feature_names,
    classifier_scores, evaluate_grid, _load_development,
)

LEARNERS = ['HGB31', 'ExtraTrees256']
PROTOCOL = 'docs/ML_V11_TREND_BACKBONE_PLAN.md'
SEED = 20261023
CONFIGS = {
    'HGB31': {'max_iter': 150, 'max_leaf_nodes': 31, 'min_samples_leaf': 20,
              'learning_rate': .05, 'l2_regularization': 1., 'early_stopping': False, 'random_state': SEED},
    'ExtraTrees256': {'n_estimators': 256, 'min_samples_leaf': 4, 'min_samples_split': 2,
                      'max_features': 1., 'criterion': 'gini', 'max_depth': None,
                      'bootstrap': False, 'class_weight': None, 'n_jobs': 2, 'random_state': SEED},
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def fit_model(records: list[dict], learner: str, forbidden_ids: list[str] | None = None,
              fit_counter: dict | None = None) -> tuple[dict, dict]:
    if learner not in LEARNERS:
        raise ValueError('Unknown learner outside the frozen two-learner comparison.')
    ids = [record['id'] for record in records]
    if len(ids) != len(set(ids)) or set(ids) & set(forbidden_ids or []):
        raise ValueError('Duplicate or forbidden record entered learner fitting.')
    designs = []; targets = []; indices_by_record = []; sensor_values = []; counts = {}
    for record in records:
        if record.get('labels') is None or not np.isin(record['labels'], [0, 1]).all():
            raise ValueError('Explicit binary labels are required for model fitting.')
        X, indices, _ = causal_features(record)
        designs.append(X); targets.append(np.asarray(record['labels'], int)[indices])
        indices_by_record.append(indices); sensor_values.append(record['values'][indices]); counts[record['id']] = len(indices)
    X = np.concatenate(designs); y = np.concatenate(targets)
    if len(np.unique(y)) != 2:
        raise ValueError('Both pooled training classes are required.')
    weights, weighting = sample_weights(records, indices_by_record, 'record_equal_class_balanced')
    center = np.median(X, axis=0)
    scale = np.maximum(np.quantile(X, .75, axis=0) - np.quantile(X, .25, axis=0),
                       np.maximum(X.std(axis=0) * .05, 1e-6))
    sensors = np.concatenate(sensor_values)
    sensor_scale = np.maximum(np.quantile(sensors, .75, axis=0) - np.quantile(sensors, .25, axis=0),
                              np.maximum(sensors.std(axis=0) * .05, 1e-6))
    estimator = HistGradientBoostingClassifier(**CONFIGS[learner]) if learner == 'HGB31' else ExtraTreesClassifier(**CONFIGS[learner])
    if fit_counter is not None:
        fit_counter['fits'] += 1
    with threadpool_limits(limits=2):
        estimator.fit((X - center) / scale, y, sample_weight=weights)
    bundle = {'model': estimator, 'arm': learner, 'feature_center': center, 'feature_scale': scale,
              'sensor_scale': sensor_scale, 'feature_names': feature_names(), 'feature_count': 186,
              'threshold': .5, 'window': 2, 'version': 'learned-trend-backbones-11.0.0',
              'algorithm': type(estimator).__name__, 'objective': 'supervised_known_anomaly_classification'}
    evidence = {'learner': learner, 'training_ids': ids, 'forbidden_ids': sorted(forbidden_ids or []),
                'training_rows_by_recording': counts, 'training_rows': len(y),
                'normal_rows': int((y == 0).sum()), 'anomalous_rows': int((y == 1).sum()),
                'feature_count': 186, 'sample_weights': weighting,
                'feature_center': center.tolist(), 'feature_scale': scale.tolist(),
                'estimator_parameters': estimator.get_params(deep=False), 'hgb_or_et_fits': 1}
    return bundle, evidence


def select_candidate(points: list[dict]) -> dict | None:
    feasible = [point for point in points if point['feasible']]
    def rank(point):
        metric = point['metrics']
        return (-metric['detected_events'], metric['false_episodes'], metric['false_alert_seconds'],
                metric['median_delay_seconds'] if metric['median_delay_seconds'] is not None else float('inf'),
                metric['mean_delay_seconds'] if metric['mean_delay_seconds'] is not None else float('inf'),
                LEARNERS.index(point['arm']), point['threshold_order'])
    return min(feasible, key=rank) if feasible else None


def evaluate_candidates(records: list[dict], candidates: dict, reference_fusion: dict,
                         classifier_A: dict, classifier_threshold: float) -> dict:
    if set(candidates) != set(LEARNERS):
        raise ValueError('Both and only the fixed learners must enter the candidate comparison.')
    # Reuse exactly the frozen V10 evaluator/gates; its internal B/C score slots
    # receive the new learners, with output identities restored before ranking.
    mapping = {'B': 'HGB31', 'C': 'ExtraTrees256'}
    evaluated = evaluate_grid(records, {'B': candidates['HGB31'], 'C': candidates['ExtraTrees256']},
                              reference_fusion, classifier_A, classifier_threshold)
    for point in evaluated['candidate_audits']:
        point['arm'] = mapping[point['arm']]
    evaluated['selected'] = select_candidate(evaluated['candidate_audits'])
    evaluated['probability_diagnostics'] = {mapping.get(key, key): value for key, value in evaluated['probability_diagnostics'].items()}
    evaluated['progressive_event_ids'] = list(PROGRESSIVE_EVENTS)
    return evaluated


def study_plan(root: Path) -> dict:
    root = Path(root); source = root / 'artifacts/trend-development'
    v10_plan = _read(source / 'plan.json'); v10_report = _read(source / 'report.json')
    if _hash(source / 'plan.json') != v10_report['plan_sha256']:
        raise ValueError('Preserved V10 plan hash mismatch.')
    if any(_hash(root / name) != digest for name, digest in v10_plan['dependencies'].items()):
        raise ValueError('A frozen V10 scientific dependency changed.')
    if _hash(source / 'B_oof_scores.npz') != v10_report['score_artifacts_sha256']['B_oof_scores.npz']:
        raise ValueError('Saved V10 B186 OOF source hash mismatch.')
    for name, digest in v10_report['score_artifacts_sha256'].items():
        if _hash(source / name) != digest:
            raise ValueError('A preserved V10 score/mining artifact changed.')
    if _hash(source / 'candidate_audits.json') != v10_report['candidate_audits_sha256']:
        raise ValueError('The preserved negative V10 candidate report changed.')
    paths = list(v10_plan['dependencies']) + [PROTOCOL, 'signalwatch/ml_trend_backbones.py',
            'tests/test_ml_trend_backbones.py', 'scripts/study_trend_backbones.py',
            'signalwatch/ml_trends.py', 'artifacts/trend-development/plan.json',
            'artifacts/trend-development/report.json', 'artifacts/trend-development/B_oof_scores.npz']
    paths += [str(path.relative_to(root)).replace('\\', '/') for path in sorted(source.iterdir()) if path.is_file()]
    return {'study': 'trend_backbones_v11', 'schema_version': 1, 'source_snapshot': SNAPSHOT,
            'protocol_path': PROTOCOL, 'protocol_sha256': _hash(root / PROTOCOL),
            'dependencies': {name: _hash(root / name) for name in dict.fromkeys(paths)},
            'development_ids': v10_plan['development_ids'], 'source_hashes': v10_plan['source_hashes'],
            'folds': v10_plan['folds'], 'normal_train_anchor': v10_plan['normal_train_anchor'],
            'feature_count': 186, 'feature_names': feature_names(), 'learners': CONFIGS,
            'development_fits': 6, 'conditional_final_fits': 1, 'maximum_total_fits': 7,
            'threshold_quantiles': v10_plan['threshold_quantiles'], 'additional_threshold': v10_plan['additional_threshold'],
            'maximum_scan_points': 204, 'constraints': v10_plan['constraints'],
            'progressive_event_ids': PROGRESSIVE_EVENTS,
            'selection': 'Mostevents,fewerfalse,lessnormalalertseconds,smallermedian/mean delay,HGB31beforeExtraTrees256,thresholdorder.',
            'controls': ['A120 preserved classifier OOF', 'V10B186 preserved classifier OOF', 'V5 preserved fullfusion OOF'],
            'inference': 'Immediate learned raw classifier probability; no temporal confirmation, innovationOR, smoothing or calibration.',
            'old_test_access': False, 'teaser_access': False, 'activation': False}


def freeze_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/trend-backbones-development'; target.mkdir(parents=True, exist_ok=True)
    plan = study_plan(root); path = target / 'plan.json'
    if path.exists():
        if _read(path) != plan:
            raise ValueError('Frozen backbone study plan or dependencies changed.')
    else:
        _write(path, plan)
    return plan


def run_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/trend-backbones-development'; started = time.perf_counter()
    plan = freeze_study(root)
    if (target / 'execution_started.json').exists() or (target / 'report.json').exists():
        raise ValueError('The one fixed backbone comparison already started; refusing another execution.')
    active = {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}
    _write(target / 'execution_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
            'dependencies': plan['dependencies'], 'active_hashes': active,
            'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__}})
    records, anchor = _load_development(root, plan); by_id = {record['id']: record for record in records}
    source = root / 'artifacts/v5/development'; reference_report = _read(source / 'report.json')
    fusion = _arrays(source / 'selected_oof_scores.npz'); classifier_A = _arrays(source / 'classifier_oof_scores.npz')
    control_B = _arrays(root / 'artifacts/trend-development/B_oof_scores.npz')
    v10_report = _read(root / 'artifacts/trend-development/report.json')
    expected = set(plan['development_ids'])
    if any(set(values) != expected for values in [fusion, classifier_A, control_B]):
        raise ValueError('Control source arrays do not contain the exact frozen development recordings.')
    for record in records:
        _, indices, _ = causal_features(record); eligible = np.zeros(len(record['values']), bool); eligible[indices] = True
        if any(values[record['id']].shape != eligible.shape or not np.array_equal(np.isfinite(values[record['id']]), eligible)
               for values in [fusion, classifier_A, control_B]):
            raise ValueError('Saved source controls differ from the frozen trend eligibility.')
    scores = {learner: {} for learner in LEARNERS}; counter = {'fits': 0}; folds = []
    for fold in plan['folds']:
        if set(fold['train_ids']) & set(fold['heldout_ids']) or set(fold['train_ids']) | set(fold['heldout_ids']) != expected:
            raise ValueError('Invalid recording-disjoint outer partition.')
        fitting = [anchor] + [by_id[key] for key in fold['train_ids']]
        old_fit = next(item for item in v10_report['folds'] if item['fold'] == fold['fold'])['B_fit']
        evidence = {}
        for learner in LEARNERS:
            bundle, detail = fit_model(fitting, learner, forbidden_ids=fold['heldout_ids'], fit_counter=counter)
            np.testing.assert_array_equal(bundle['feature_center'], old_fit['feature_center'])
            np.testing.assert_array_equal(bundle['feature_scale'], old_fit['feature_scale'])
            evidence[learner] = detail
            for key in fold['heldout_ids']:
                scores[learner][key] = classifier_scores(bundle, by_id[key])['scores']
                if not np.array_equal(np.isfinite(scores[learner][key]), np.isfinite(fusion[key])):
                    raise ValueError('New learner changed scored finite exposure.')
            print(json.dumps({'outer_fold': fold['fold'], 'learner_complete': learner, 'actual_fits': counter['fits']}), flush=True)
        folds.append({'fold': fold['fold'], 'outer_train_ids': fold['train_ids'],
                      'outer_heldout_ids': fold['heldout_ids'], 'fits': evidence})
    if counter['fits'] != 6:
        raise ValueError('Development fit count differs from the fixed six-fit budget.')
    for learner, values in scores.items():
        np.savez_compressed(target / f'{learner}_oof_scores.npz', **values)
    evaluated = evaluate_candidates(records, scores, fusion, classifier_A, reference_report['selected']['classifier_threshold'])
    if evaluated['reference_fusion']['metrics'] != reference_report['metrics']:
        raise ValueError('V5 original fusion control failed metric reproduction.')
    evaluated['reference_trend_B186_probability'] = probability_diagnostics(records, control_B)
    np.testing.assert_allclose(evaluated['reference_trend_B186_probability']['pooled']['brier_loss'],
                               v10_report['probability_diagnostics']['B']['pooled']['brier_loss'], atol=0, rtol=0)
    points = evaluated.pop('candidate_audits'); _write(target / 'candidate_audits.json', points)
    selected = evaluated['selected']; selection_path = target / 'selection_before_final_fit.json'
    _write(selection_path, {'selected': selected, 'plan_sha256': _hash(target / 'plan.json'),
                            'candidate_audits_sha256': _hash(target / 'candidate_audits.json'),
                            'oof_hashes': {learner: _hash(target / f'{learner}_oof_scores.npz') for learner in LEARNERS},
                            'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'fits_before_final': counter['fits']})
    report = {'schema_version': 1, 'scope': 'trend_backbone_grouped_oof_development_selection',
              'status': 'selected_development_candidate' if selected else 'no_feasible_candidate',
              'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
              'selection_before_final_fit_sha256': _hash(selection_path),
              'candidate_audits_sha256': _hash(target / 'candidate_audits.json'), 'folds': folds,
              'development_fits': 6, 'feature_count': 186, 'control_source_ids': plan['development_ids'],
              'old_test_scored': False, 'teaser_scored': False, 'activation': False,
              'limits': ['Repeated development learner comparison, not fresh independent validation.',
                         'HGB31 changes leaf capacity; ExtraTrees256 changes learner bias; cutoffs are selected on development.',
                         'No hard-normal mining, score calibration or temporal alarm policy.',
                         'Parallel forest probability summation can differ at float64 roundoff; prefix/label score checks useatol1e-12, finite masks and features remain exact.',
                         'Only a sole frozen final candidate may enter reused old-audit/teaser diagnostics.'], **evaluated}
    if selected is not None:
        model, evidence = fit_model([anchor] + records, selected['arm'], fit_counter=counter)
        model.update(threshold=selected['threshold'], plan_sha256=report['plan_sha256'],
                     selection_before_final_fit_sha256=report['selection_before_final_fit_sha256'])
        joblib.dump(model, target / 'selected_bundle.joblib', compress=3)
        _write(target / 'final_fit.json', {'learner': selected['arm'], 'fit': evidence,
                                         'selected_cutoff': selected['threshold'], 'total_fits': counter['fits']})
        report.update(model_sha256=_hash(target / 'selected_bundle.joblib'), final_fit=evidence,
                      final_fit_sha256=_hash(target / 'final_fit.json'))
    else:
        report['final_fit'] = None
    if counter['fits'] != (7 if selected else 6):
        raise ValueError('Whole study exceeded or differed from its selected six-plus-one fit budget.')
    if active != {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}:
        raise ValueError('Protected active V5 artifacts changed.')
    if any(_hash(root / name) != digest for name, digest in plan['dependencies'].items()):
        raise ValueError('A frozen scientific dependency changed during learner fitting.')
    report.update(actual_fits=counter['fits'], active_files_unchanged=True, dependencies_unchanged=True,
                  score_artifacts_sha256={path.name: _hash(path) for path in target.glob('*.npz')},
                  study_seconds=round(time.perf_counter() - started, 3))
    _write(target / 'report.json', report)
    return report
