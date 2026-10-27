"""Learned elapsed-time trends and record-disjoint hard-normal weighting."""
from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from signalwatch.dataset import SNAPSHOT, SENSORS, load_builtin, manifest, parse_csv_bytes
from signalwatch.evaluation import aggregate_reports, inspect_recording
from signalwatch.ml_ablation import probability_diagnostics
from signalwatch.ml_supervised import causal_features as base_features, feature_names as base_feature_names
from signalwatch.ml_supervised_weighted import sample_weights

HORIZONS = [30., 120.]
PARAMETERS = {'max_iter': 150, 'max_leaf_nodes': 15, 'min_samples_leaf': 20,
              'learning_rate': .05, 'l2_regularization': 1., 'early_stopping': False}
SEED = 20261023
ANCHOR = 'normal_train_anchor'
PROTOCOL = 'docs/ML_V10_TREND_MODEL_PLAN.md'
PROGRESSIVE_EVENTS = ['other-6-event-0', 'other-9-event-0']
QUANTILES = np.linspace(.90, 1., 101).tolist()
ACTIVE_FILES = ['model.joblib', 'metadata.json', 'frozen_evaluation.json',
                'selection_manifest.json', 'split_manifest.json', 'release_manifest.json']
SOURCE_NAMES = ['plan.json', 'report.json', 'classifier_oof_scores.npz', 'selected_oof_scores.npz']


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def feature_names() -> list[str]:
    keys = [sensor['key'] for sensor in SENSORS]
    names = base_feature_names()
    names += [f'trend_{statistic}{int(window)}s:{key}' for window in HORIZONS
              for statistic in ['slope', 'net', 'direction', 'r2'] for key in keys]
    return names + [f'trend_coverage{int(window)}s' for window in HORIZONS]


def causal_features(record: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    base, indices, reference = base_features(record)
    elapsed = np.asarray(record['elapsed'], float)
    values = np.asarray(record['values'], float)
    gaps = np.asarray(record['gap_before'], bool)
    if elapsed.shape != (len(values),) or gaps.shape != elapsed.shape:
        raise ValueError('Trend timestamps and gap flags must match observations.')
    if not np.isfinite(elapsed).all() or np.any(np.diff(elapsed) <= 0):
        raise ValueError('Trend timestamps must be finite and strictly increasing.')
    extra = np.zeros((len(indices), 66), float)
    segment_starts = np.maximum.accumulate(np.where(gaps, np.arange(len(values)), 0))
    for row, index in enumerate(indices):
        segment_start = int(segment_starts[index])
        for hindex, window in enumerate(HORIZONS):
            lower = max(segment_start, int(np.searchsorted(elapsed, elapsed[index] - window, side='left')))
            past = values[lower:index]
            if len(past) < 2:
                continue
            times = elapsed[lower:index] - elapsed[lower]
            centered_time = times - times.mean()
            centered_values = past - past.mean(axis=0)
            time_ss = float(centered_time @ centered_time)
            covariance = centered_time @ centered_values
            slope = covariance / time_ss
            value_ss = np.sum(centered_values * centered_values, axis=0)
            r2 = np.divide(covariance * covariance, time_ss * value_ss,
                           out=np.zeros(8), where=value_ss > 0)
            r2 = np.clip(r2, 0., 1.)
            constant = np.ptp(past, axis=0) == 0.
            slope[constant] = 0.; r2[constant] = 0.
            statistics = np.concatenate([slope, past[-1] - past[0],
                                          np.sign(np.diff(past, axis=0)).mean(axis=0), r2])
            extra[row, hindex * 32:(hindex + 1) * 32] = statistics
            extra[row, 64 + hindex] = min(float(times[-1]) / window, 1.)
    output = np.column_stack([base, extra])
    if output.shape[1] != 186 or not np.isfinite(output).all():
        raise ValueError('Trend features exceed the finite numerical range.')
    return output, indices, reference


def inner_splits(ids: list[str]) -> list[dict]:
    ordered = sorted(ids)
    if len(ordered) != len(set(ordered)) or len(ordered) < 3 or ANCHOR in ordered:
        raise ValueError('Mining requires at least three unique disturbed recording IDs.')
    return [{'fold': fold,
             'train_ids': [key for index, key in enumerate(ordered) if index % 3 != fold],
             'heldout_ids': [key for index, key in enumerate(ordered) if index % 3 == fold]}
            for fold in range(3)]


def choose_hard_normals(record: dict, scores: np.ndarray, indices: np.ndarray) -> np.ndarray:
    labels = np.asarray(record['labels'])
    scores = np.asarray(scores, float)
    if scores.shape != labels.shape:
        raise ValueError('Mining score and recording label shapes differ.')
    normal = np.asarray(indices, int)[labels[indices] == 0]
    if not len(normal):
        return np.zeros(0, int)
    if not np.isfinite(scores[normal]).all():
        raise ValueError('Every eligible normal mining row requires an out-of-fold score.')
    order = np.lexsort((normal, -scores[normal]))
    return np.sort(normal[order[:math.ceil(.10 * len(normal))]])


def hard_normal_weights(records: list[dict], indices_by_record: list[np.ndarray], hard_indices: dict) -> tuple[np.ndarray, dict]:
    if ANCHOR in hard_indices:
        raise ValueError('The always-fit anchor must not be mined or upweighted.')
    ids = {record['id'] for record in records}
    if set(hard_indices) != ids - {ANCHOR}:
        raise ValueError('Hard-normal indices must cover exactly the disturbed training recordings.')
    original, base_evidence = sample_weights(records, indices_by_record, 'record_equal_class_balanced')
    output = original.copy(); offset = 0; evidence = {}
    for record, indices in zip(records, indices_by_record):
        part = output[offset:offset + len(indices)]
        base_part = original[offset:offset + len(indices)]
        labels = np.asarray(record['labels'])[indices]
        raw_selected = np.asarray(hard_indices.get(record['id'], []))
        selected = raw_selected.astype(int)
        if not np.array_equal(raw_selected, selected) or len(selected) != len(np.unique(selected)):
            raise ValueError('Hard-normal indices must be unique integer row positions.')
        retained = np.isin(indices, selected)
        if int(retained.sum()) != len(selected) or np.any(labels[retained] != 0):
            raise ValueError('Only eligible normal rows may receive hard-normal weights.')
        normal = labels == 0
        normal_mass = float(base_part[normal].sum())
        if len(selected):
            part[retained] *= 3.
            part[normal] *= normal_mass / float(part[normal].sum())
        evidence[record['id']] = {'hard_indices': selected.tolist(), 'hard_normal_rows': len(selected),
            'eligible_normal_rows': int(normal.sum()), 'base_normal_mass': normal_mass,
            'updated_normal_mass': float(part[normal].sum()),
            'base_anomalous_mass': float(base_part[~normal].sum()),
            'updated_anomalous_mass': float(part[~normal].sum()),
            'base_record_mass': float(base_part.sum()), 'updated_record_mass': float(part.sum())}
        offset += len(indices)
    if not np.isfinite(output).all() or np.any(output <= 0):
        raise ValueError('Hard-normal weighting produced invalid fit weights.')
    return output, {'scheme': 'inner_oof_top10percent_normal_x3_record_class_mass_preserved',
                    'anchor_mined': False, 'base': base_evidence,
                    'per_recording': evidence, 'weight_sum': float(output.sum())}


def fit_model(records: list[dict], arm: str = 'B', hard_indices: dict | None = None,
              forbidden_ids: list[str] | None = None, fit_counter: dict | None = None) -> tuple[dict, dict]:
    ids = [record['id'] for record in records]
    if len(ids) != len(set(ids)) or set(ids) & set(forbidden_ids or []):
        raise ValueError('Duplicate or forbidden recording entered model fitting.')
    if arm not in ['B', 'C']:
        raise ValueError('Model arm is outside the frozen learned study.')
    designs = []; targets = []; indices_by_record = []; raw_values = []; counts = {}
    for record in records:
        if record.get('labels') is None or not np.isin(record['labels'], [0, 1]).all():
            raise ValueError('Model fitting requires explicit binary training labels.')
        X, indices, _ = causal_features(record)
        designs.append(X); targets.append(np.asarray(record['labels'], int)[indices])
        indices_by_record.append(indices); raw_values.append(record['values'][indices]); counts[record['id']] = len(indices)
    X = np.concatenate(designs); y = np.concatenate(targets)
    if len(np.unique(y)) != 2:
        raise ValueError('Both pooled training classes must be present.')
    if arm == 'C':
        if hard_indices is None:
            raise ValueError('Arm C requires record-disjoint mined normal indices.')
        weights, weighting = hard_normal_weights(records, indices_by_record, hard_indices)
    else:
        if hard_indices is not None:
            raise ValueError('Arm B does not apply hard-normal weighting.')
        weights, weighting = sample_weights(records, indices_by_record, 'record_equal_class_balanced')
    center = np.median(X, axis=0)
    scale = np.maximum(np.quantile(X, .75, axis=0) - np.quantile(X, .25, axis=0),
                       np.maximum(X.std(axis=0) * .05, 1e-6))
    sensors = np.concatenate(raw_values)
    sensor_scale = np.maximum(np.quantile(sensors, .75, axis=0) - np.quantile(sensors, .25, axis=0),
                              np.maximum(sensors.std(axis=0) * .05, 1e-6))
    model = HistGradientBoostingClassifier(**PARAMETERS, random_state=SEED)
    if fit_counter is not None:
        fit_counter['fits'] += 1
    with threadpool_limits(limits=2):
        model.fit((X - center) / scale, y, sample_weight=weights)
    bundle = {'model': model, 'arm': arm, 'feature_center': center, 'feature_scale': scale,
              'sensor_scale': sensor_scale, 'feature_names': feature_names(), 'feature_count': 186,
              'threshold': .5, 'window': 2, 'version': 'learned-trends-10.0.0',
              'algorithm': 'HistGradientBoostingClassifier', 'objective': 'supervised_known_anomaly_classification'}
    evidence = {'arm': arm, 'training_ids': ids, 'forbidden_ids': sorted(forbidden_ids or []),
                'training_rows_by_recording': counts, 'training_rows': len(y),
                'normal_rows': int((y == 0).sum()), 'anomalous_rows': int((y == 1).sum()),
                'feature_count': 186, 'sample_weights': weighting,
                'feature_center': center.tolist(), 'feature_scale': scale.tolist(),
                'hgb_fits': 1}
    return bundle, evidence


def classifier_scores(bundle: dict, record: dict) -> dict:
    X, indices, reference = causal_features(record)
    scores = np.full(len(record['values']), np.nan)
    contributions = np.full((len(scores), 8), np.nan)
    if len(indices):
        with threadpool_limits(limits=2):
            probabilities = bundle['model'].predict_proba((X - bundle['feature_center']) / bundle['feature_scale'])
        positive = np.flatnonzero(bundle['model'].classes_ == 1)
        if len(positive) != 1:
            raise ValueError('The fitted model requires a unique positive class.')
        scores[indices] = probabilities[:, positive[0]]
        contributions[indices] = np.abs(reference) / bundle['sensor_scale']
    return {'scores': scores, 'contributions': contributions, 'warmup_samples': 2,
            'threshold': float(bundle['threshold'])}


def mine_hard_normals(records: list[dict], anchor: dict, forbidden_ids: list[str] | None = None,
                      fit_counter: dict | None = None) -> tuple[dict, dict, dict]:
    by_id = {record['id']: record for record in records}
    forbidden = set(forbidden_ids or [])
    if len(by_id) != len(records) or ANCHOR in by_id or anchor['id'] != ANCHOR or forbidden & (set(by_id) | {ANCHOR}):
        raise ValueError('Forbidden or duplicate record entered inner hard-normal mining.')
    scores = {}; folds = []
    for fold in inner_splits(list(by_id)):
        fitting = [anchor] + [by_id[key] for key in fold['train_ids']]
        bundle, evidence = fit_model(fitting, forbidden_ids=sorted(forbidden | set(fold['heldout_ids'])), fit_counter=fit_counter)
        for key in fold['heldout_ids']:
            scores[key] = classifier_scores(bundle, by_id[key])['scores']
        folds.append(dict(fold, fit=evidence))
    hard = {key: choose_hard_normals(record, scores[key], causal_features(record)[1]) for key, record in by_id.items()}
    return hard, {'inner_folds': folds, 'mined_record_ids': sorted(by_id),
                  'outer_forbidden_ids': sorted(forbidden), 'anchor_mined': False,
                  'fraction': .10, 'multiplier_before_mass_normalization': 3.,
                  'hard_indices': {key: value.tolist() for key, value in hard.items()}, 'hgb_fits': 3}, scores


def screen_candidate(metric: dict, reference: dict, lost_progressive_ids: list[str]) -> list[str]:
    if metric['observed_normal_seconds'] != reference['observed_normal_seconds']:
        raise ValueError('Candidate changed scored normal exposure.')
    reasons = []
    if metric['detected_events'] < 26: reasons.append('development_recall_floor')
    if metric['false_episodes'] >= 3: reasons.append('no_strict_false_count_improvement')
    if metric['false_episodes_per_hour'] is None or metric['false_episodes_per_hour'] > 2.: reasons.append('false_onset_budget')
    if metric['false_alert_seconds'] > 37.: reasons.append('reference_normal_alert_time')
    if lost_progressive_ids: reasons.append('progressive_subset_retention')
    return reasons


def select_candidate(points: list[dict]) -> dict | None:
    feasible = [point for point in points if point['feasible']]
    def rank(point):
        metric = point['metrics']
        return (-metric['detected_events'], metric['false_episodes'], metric['false_alert_seconds'],
                metric['median_delay_seconds'] if metric['median_delay_seconds'] is not None else float('inf'),
                metric['mean_delay_seconds'] if metric['mean_delay_seconds'] is not None else float('inf'),
                ['B', 'C'].index(point['arm']), point['threshold_order'])
    return min(feasible, key=rank) if feasible else None


def _trace(score: np.ndarray, threshold: float) -> dict:
    return {'scores': {'learned': score, 'robust': np.full(len(score), np.nan),
                       'persistence': np.full(len(score), np.nan)},
            'thresholds': {'learned': threshold, 'robust': 1., 'persistence': 1.}}


def _reports(records: list[dict], scores: dict, threshold: float) -> tuple[dict, list[dict]]:
    reports = [inspect_recording(record, _trace(scores[record['id']], threshold)) for record in records]
    return aggregate_reports(reports)['learned'], reports


def _events(reports: list[dict]) -> dict:
    return {event['id']: event for report in reports for event in report['events']
            if event['status'] in ['detected', 'missed']}


def _groups(records: list[dict], reports: list[dict]) -> dict:
    return {group: aggregate_reports([report for record, report in zip(records, reports)
                                     if record['group'] == group])['learned']
            for group in sorted({record['group'] for record in records})}


def evaluate_grid(records: list[dict], candidate_scores: dict, reference_scores: dict,
                  control_classifier: dict, classifier_threshold: float) -> dict:
    reference_metric, reference_reports = _reports(records, reference_scores, 1.)
    reference_events = _events(reference_reports)
    reference_hits = {key for key, event in reference_events.items() if event['status'] == 'detected'}
    if not set(PROGRESSIVE_EVENTS) <= reference_hits:
        raise ValueError('Frozen progressive reference events are not detected by V5.')
    classifier_metric, classifier_reports = _reports(records, control_classifier, classifier_threshold)
    points = []
    for arm in ['B', 'C']:
        scores = candidate_scores[arm]
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
            progressive = [row for row in pairs if row['id'] in PROGRESSIVE_EVENTS]
            points.append({'arm': arm, 'threshold': float(cutoff), 'threshold_order': order,
                           'metrics': metric, 'feasible': not reasons, 'rejection_reasons': reasons,
                           'lost_event_ids': lost, 'gained_event_ids': gained,
                           'progressive_subset': progressive, 'lost_progressive_event_ids': lost_progressive,
                           'paired_outcomes': pairs, 'group_slices': _groups(records, reports),
                           'per_recording': [{'id': record['id'], 'group': record['group'],
                                              'metrics': report['metrics']['learned'],
                                              'false_onset_seconds': [event['start_seconds'] for event in report['events'] if event['status'] == 'false_alert']}
                                             for record, report in zip(records, reports)]})
    if len(points) > 204:
        raise ValueError('The threshold scan exceeds its predeclared candidate budget.')
    return {'candidate_audits': points, 'selected': select_candidate(points),
            'reference_fusion': {'metrics': reference_metric, 'group_slices': _groups(records, reference_reports),
                                 'detected_event_ids': sorted(reference_hits)},
            'reference_classifier_A': {'metrics': classifier_metric, 'threshold': classifier_threshold,
                                      'group_slices': _groups(records, classifier_reports)},
            'probability_diagnostics': {arm: probability_diagnostics(records, scores)
                                       for arm, scores in {'A': control_classifier, **candidate_scores}.items()}}


def study_plan(root: Path) -> dict:
    root = Path(root); archive = root / 'artifacts/v5'
    preservation = _read(archive / 'preservation_manifest.json')
    for name in SOURCE_NAMES:
        if _hash(archive / 'development' / name) != preservation['files']['development/' + name]:
            raise ValueError('Preserved V5 source hash mismatch.')
    source = _read(archive / 'development/plan.json')
    paths = [PROTOCOL, 'signalwatch/ml_trends.py', 'tests/test_ml_trends.py', 'scripts/study_trends.py',
             'signalwatch/dataset.py', 'signalwatch/evaluation.py', 'signalwatch/ml_supervised.py',
             'signalwatch/ml_supervised_weighted.py', 'signalwatch/ml_ablation.py',
             'signalwatch/ml_hybrid.py', 'signalwatch/ml_v2.py',
             'artifacts/v5/preservation_manifest.json', 'data/SKAB-DATA-README.md']
    paths += ['artifacts/v5/development/' + name for name in SOURCE_NAMES]
    return {'study': 'learned_trends_hard_normals_v10', 'schema_version': 1, 'source_snapshot': SNAPSHOT,
            'protocol_path': PROTOCOL, 'protocol_sha256': _hash(root / PROTOCOL),
            'dependencies': {name: _hash(root / name) for name in paths},
            'development_ids': source['development_ids'], 'source_hashes': source['source_hashes'],
            'folds': source['folds'], 'normal_train_anchor': source['normal_train_anchor'],
            'feature_names': feature_names(), 'feature_count': 186, 'horizons_seconds': HORIZONS,
            'features': 'Original120 plus past-only30/120s slope/net/sign-consistency/R2 perchannel and two coverage values; same2row/gap eligibility.',
            'parameters': PARAMETERS, 'random_state': SEED, 'max_cpu_threads': 2,
            'arms': {'A': 'Preserved120feature classifier probability OOF reference, no fit.',
                     'B': 'HGB186features with original equalrecord/class masses.',
                     'C': 'Same186features/HGB with nestedOOF top10percentnormal x3 and restored record/class masses.'},
            'mining': {'fraction': .10, 'multiplier': 3., 'ties': 'score descending, raw row ascending',
                       'inner_folds': 3, 'anchor_mined': False, 'base_inner_arm': 'B'},
            'development_hgb_fits': 15, 'final_fit_if_selected': {'B': 1, 'C': 4}, 'maximum_total_hgb_fits': 19,
            'threshold_quantiles': QUANTILES, 'additional_threshold': 'nextafter(max normal,+inf)',
            'maximum_scan_points': 204, 'progressive_event_ids': PROGRESSIVE_EVENTS,
            'constraints': {'minimum_events': 26, 'false_starts_strictly_less_than': 3,
                            'max_false_starts_per_normal_hour': 2., 'max_normal_alert_seconds': 37.,
                            'normal_exposure_seconds': 9179., 'progressive_retention': 'all2originalV5hits'},
            'selection': 'Mostevents, fewerfalse, lessnormalalerttime, smaller median/mean delay, BbeforeC, thresholdorder.',
            'inference': 'Raw learned classifier probability, immediate cutoff; no temporal confirmation/smoothing/innovationOR.',
            'old_test_access': False, 'teaser_access': False, 'activation': False,
            'final_selection_freeze': 'Write selected identity/cutoff and OOF hashes before finalB1 orC4fits.'}


def freeze_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/trend-development'; target.mkdir(parents=True, exist_ok=True)
    plan = study_plan(root); path = target / 'plan.json'
    if path.exists():
        if _read(path) != plan:
            raise ValueError('Frozen trend study protocol or dependencies changed.')
    else:
        _write(path, plan)
    return plan


def _load_development(root: Path, plan: dict) -> tuple[list[dict], dict]:
    records = [load_builtin(root, key) for key in plan['development_ids']]
    if any(record['split'] != 'validation' or record['sha256'] != plan['source_hashes'][record['id']] for record in records):
        raise ValueError('Only hash-matched original development recordings may enter this study.')
    source = next(entry for entry in manifest(root)['recordings'] if entry['id'] == 'anomaly-free')
    raw = (root / source['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan['source_hashes']['anomaly-free']:
        raise ValueError('Normal training anchor source hash mismatch.')
    normal = parse_csv_bytes(raw, ANCHOR, 'Fixed normal training anchor')
    start, end = plan['normal_train_anchor']
    anchor = dict(normal)
    for name in ['values', 'timestamps', 'elapsed', 'gap_before']:
        anchor[name] = normal[name][start:end].copy() if isinstance(normal[name], np.ndarray) else normal[name][start:end]
    anchor['elapsed'] -= anchor['elapsed'][0]; anchor['gap_before'][0] = False
    anchor['labels'] = np.zeros(end - start, bool); anchor['changepoints'] = None
    return records, anchor


def _mining_boundary_description(records: list[dict], evidence: list[dict]) -> dict:
    by_id = {record['id']: record for record in records}; result = []
    for outer in evidence:
        for key, indices in outer['mining']['hard_indices'].items():
            record = by_id[key]
            boundary_rows = np.flatnonzero(np.r_[False, record['labels'][1:] != record['labels'][:-1]])
            distances = [float(np.min(np.abs(record['elapsed'][boundary_rows] - record['elapsed'][index])))
                         if len(boundary_rows) else None for index in indices]
            result.append({'outer_fold': outer['fold'], 'recording_id': key, 'mined_rows': len(indices),
                           'within1s_annotation_boundary': sum(value is not None and value <= 1. for value in distances),
                           'within5s_annotation_boundary': sum(value is not None and value <= 5. for value in distances)})
    return {'scope': 'Post-selection descriptive proximity to label transitions, never an exclusion or inference feature.', 'rows': result}


def run_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/trend-development'; started = time.perf_counter()
    plan = freeze_study(root)
    if (target / 'execution_started.json').exists() or (target / 'report.json').exists():
        raise ValueError('The one frozen learned study already started; refusing another run.')
    active = {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}
    _write(target / 'execution_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
            'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
            'dependencies': plan['dependencies'], 'active_hashes': active,
            'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__, 'numpy': np.__version__},
            'authorization': 'One15fitdevelopment study plus conditionally one selected finalB1/C4fit bank; no audit or teaser inference.'})
    records, anchor = _load_development(root, plan); by_id = {record['id']: record for record in records}
    archive = root / 'artifacts/v5/development'
    control_classifier = _arrays(archive / 'classifier_oof_scores.npz'); fusion = _arrays(archive / 'selected_oof_scores.npz')
    expected = set(plan['development_ids'])
    if any(set(values) != expected for values in [control_classifier, fusion]):
        raise ValueError('Control OOF recording IDs differ from the frozen development set.')
    for record in records:
        key = record['id']; _, indices, _ = causal_features(record)
        eligible = np.zeros(len(record['values']), bool); eligible[indices] = True
        if any(values[key].shape != eligible.shape or not np.array_equal(np.isfinite(values[key]), eligible)
               for values in [control_classifier, fusion]):
            raise ValueError('Control source finite eligibility differs from trend features.')
    original, _ = _reports(records, fusion, 1.)
    source_report = _read(archive / 'report.json')
    if original != source_report['metrics']:
        raise ValueError('Saved V5 original operating point failed to reproduce.')
    counter = {'fits': 0}; oof = {'B': {}, 'C': {}}; folds = []; mining_arrays = {}
    for fold in plan['folds']:
        fitting = [by_id[key] for key in fold['train_ids']]; heldout = fold['heldout_ids']
        if set(heldout) & set(fold['train_ids']) or set(heldout) | set(fold['train_ids']) != expected:
            raise ValueError('Invalid whole-recording outer fold.')
        model_b, evidence_b = fit_model([anchor] + fitting, forbidden_ids=heldout, fit_counter=counter)
        hard, mining, inner_oof = mine_hard_normals(fitting, anchor, forbidden_ids=heldout, fit_counter=counter)
        model_c, evidence_c = fit_model([anchor] + fitting, arm='C', hard_indices=hard,
                                         forbidden_ids=heldout, fit_counter=counter)
        np.testing.assert_array_equal(model_b['feature_center'], model_c['feature_center'])
        np.testing.assert_array_equal(model_b['feature_scale'], model_c['feature_scale'])
        for key in heldout:
            for arm, model in [('B', model_b), ('C', model_c)]:
                oof[arm][key] = classifier_scores(model, by_id[key])['scores']
                if not np.array_equal(np.isfinite(oof[arm][key]), np.isfinite(fusion[key])):
                    raise ValueError('A new classifier changed finite scored exposure.')
        for key, values in inner_oof.items():
            mining_arrays[f'outer{fold["fold"]}__{key}'] = values
        folds.append({'fold': fold['fold'], 'outer_train_ids': fold['train_ids'], 'outer_heldout_ids': heldout,
                      'B_fit': evidence_b, 'C_fit': evidence_c, 'mining': mining})
        print(json.dumps({'outer_fold_complete': fold['fold'], 'actual_hgb_fits': counter['fits']}), flush=True)
    if counter['fits'] != 15:
        raise ValueError('Development fit count differs from the frozen15fit budget.')
    for arm, values in oof.items():
        np.savez_compressed(target / f'{arm}_oof_scores.npz', **values)
    np.savez_compressed(target / 'inner_mining_oof_scores.npz', **mining_arrays)
    evaluated = evaluate_grid(records, oof, fusion, control_classifier, source_report['selected']['classifier_threshold'])
    points = evaluated.pop('candidate_audits'); _write(target / 'candidate_audits.json', points)
    selected = evaluated['selected']; selection_path = target / 'selection_before_final_fit.json'
    _write(selection_path, {'selected': selected, 'plan_sha256': _hash(target / 'plan.json'),
                            'candidate_audits_sha256': _hash(target / 'candidate_audits.json'),
                            'oof_hashes': {arm: _hash(target / f'{arm}_oof_scores.npz') for arm in ['B', 'C']},
                            'frozen_at_utc': datetime.now(timezone.utc).isoformat(), 'fits_before_final': counter['fits']})
    report = {'schema_version': 1, 'scope': 'learned_trend_grouped_oof_development_selection',
              'status': 'selected_development_candidate' if selected else 'no_feasible_candidate',
              'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
              'selection_before_final_fit_sha256': _hash(selection_path),
              'candidate_audits_sha256': _hash(target / 'candidate_audits.json'),
              'folds': folds, 'development_hgb_fits': 15, 'feature_count': 186,
              'control_source_ids': plan['development_ids'],
              'mined_normal_boundary_proximity': _mining_boundary_description(records, folds),
              'old_test_scored': False, 'teaser_scored': False, 'activation': False,
              'limits': ['Repeated development selection, not fresh independent validation.',
                         'B/A changes representation; B or C versus fusion also removes innovationOR.',
                         'Hard normals are high-score annotated normal rows, not necessarily misclassifications.',
                         'No delay/smoothing/confirmation rules or target relabelling.',
                         'Previously inspected audit/teaser must remain retrospective diagnostics.'], **evaluated}
    if selected is not None:
        arm = selected['arm']; final_mining = None
        if arm == 'C':
            hard, final_mining, final_scores = mine_hard_normals(records, anchor, fit_counter=counter)
            np.savez_compressed(target / 'final_mining_oof_scores.npz', **final_scores)
        else:
            hard = None
        final_model, final_evidence = fit_model([anchor] + records, arm=arm, hard_indices=hard, fit_counter=counter)
        final_model.update(threshold=selected['threshold'], plan_sha256=report['plan_sha256'],
                           selection_before_final_fit_sha256=report['selection_before_final_fit_sha256'])
        joblib.dump(final_model, target / 'selected_bundle.joblib', compress=3)
        _write(target / 'final_fit.json', {'arm': arm, 'fit': final_evidence, 'mining': final_mining,
                                         'selected_cutoff': selected['threshold'], 'total_hgb_fits': counter['fits']})
        report.update(model_sha256=_hash(target / 'selected_bundle.joblib'), final_fit=final_evidence,
                      final_fit_sha256=_hash(target / 'final_fit.json'))
    else:
        report['final_fit'] = None
    allowed_fits = 15 if selected is None else 16 if selected['arm'] == 'B' else 19
    if counter['fits'] != allowed_fits:
        raise ValueError('Whole-study fit count differs from the frozen selected-arm budget.')
    if active != {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}:
        raise ValueError('Protected active V5 files changed during training.')
    if any(_hash(root / name) != digest for name, digest in plan['dependencies'].items()):
        raise ValueError('Frozen scientific dependency changed during the study.')
    report.update(actual_hgb_fits=counter['fits'], active_files_unchanged=True, dependencies_unchanged=True,
                  score_artifacts_sha256={path.name: _hash(path) for path in target.glob('*.npz')},
                  study_seconds=round(time.perf_counter() - started, 3))
    _write(target / 'report.json', report)
    return report
