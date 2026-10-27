"""One frozen branch-selective alarm-policy study over preserved V5 ML scores."""
from __future__ import annotations

import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from signalwatch.alarm_policy import apply_confirmation
from signalwatch.dataset import SNAPSHOT, load_builtin
from signalwatch.evaluation import aggregate_reports, inspect_recording

CONFIRMATION_SECONDS = 1.
PROTOCOL = 'docs/ML_V9_BRANCH_POLICY_PLAN.md'
ACTIVE_FILES = ['model.joblib', 'metadata.json', 'frozen_evaluation.json',
                'selection_manifest.json', 'split_manifest.json', 'release_manifest.json']
SOURCE_NAMES = ['plan.json', 'report.json', 'classifier_oof_scores.npz',
                'residual_oof_channels.npz', 'selected_oof_scores.npz']


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _arrays(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as stored:
        return {key: stored[key].copy() for key in stored.files}


def apply_branch_policy(classifier_ratio: np.ndarray, residual_ratio: np.ndarray,
                        elapsed: np.ndarray, gaps: np.ndarray) -> dict:
    """Confirm only residual support; keep the classifier immediate and both-finite exposure."""
    classifier = np.asarray(classifier_ratio, float)
    residual = np.asarray(residual_ratio, float)
    if classifier.ndim != 1 or residual.shape != classifier.shape:
        raise ValueError('Branch ratios must be matching one-dimensional vectors.')
    eligible = np.isfinite(classifier) & np.isfinite(residual)
    visible_residual = np.where(eligible, residual, np.nan)
    policy = apply_confirmation(visible_residual, elapsed, gaps, CONFIRMATION_SECONDS)
    raw = np.full(len(classifier), np.nan)
    decision = np.full(len(classifier), np.nan)
    raw[eligible] = np.maximum(classifier[eligible], residual[eligible])
    decision[eligible] = np.maximum(classifier[eligible], policy['decision_score'][eligible])
    return {'raw_score': raw, 'decision_score': decision,
            'residual_raw_score': visible_residual, 'residual_decision_score': policy['decision_score'],
            'pending': policy['pending'], 'confirmed': policy['confirmed'],
            'raw_episode_start': policy['raw_episode_start'],
            'confirmation_seconds': CONFIRMATION_SECONDS}


def apply_to_scored(raw_scored: dict, record: dict) -> dict:
    branches = raw_scored['normalized_branch_scores']
    policy = apply_branch_policy(branches['classifier'], branches['innovation'],
                                 record['elapsed'], record['gap_before'])
    np.testing.assert_allclose(policy['raw_score'], raw_scored['scores']['learned'],
                               atol=0, rtol=0, equal_nan=True)
    output = dict(raw_scored)
    output['scores'] = dict(raw_scored['scores'], learned=policy['decision_score'])
    output['thresholds'] = dict(raw_scored['thresholds'], learned=1.)
    output['raw_scores'] = {'learned': policy['raw_score']}
    output['alarm_policy'] = {'kind': 'branch_selective_confirmation',
                              'classifier': 'immediate', 'innovation_confirmation_seconds': CONFIRMATION_SECONDS,
                              'confirmation_seconds': CONFIRMATION_SECONDS,
                              'finite_eligibility': 'Both branches finite; either non-finite resets residual pending.'}
    output['policy_state'] = {key: policy[key] for key in ['pending', 'confirmed', 'raw_episode_start']}
    output['residual_decision_score'] = policy['residual_decision_score']
    output['learned_version'] = 'branch-policy-research-9.0.0'
    return output


def screen_candidate(metric: dict, reference: dict) -> list[str]:
    if metric['observed_normal_seconds'] != reference['observed_normal_seconds']:
        raise ValueError('Policy changed observed normal exposure.')
    reasons = []
    if metric['detected_events'] < 26:
        reasons.append('development_recall_floor')
    if metric['false_episodes'] >= reference['false_episodes']:
        reasons.append('no_strict_false_count_improvement')
    if metric['false_episodes_per_hour'] is None or metric['false_episodes_per_hour'] > 2.:
        reasons.append('false_onset_budget')
    if metric['false_alert_seconds'] > reference['false_alert_seconds']:
        reasons.append('reference_normal_alert_time')
    return reasons


def _trace(score: np.ndarray, classifier: np.ndarray, residual: np.ndarray) -> dict:
    return {'scores': {'learned': score, 'robust': np.full(len(score), np.nan),
                       'persistence': np.full(len(score), np.nan)},
            'thresholds': dict.fromkeys(['learned', 'robust', 'persistence'], 1.),
            'normalized_branch_scores': {'classifier': classifier, 'innovation': residual}}


def _event_map(reports: list[dict]) -> dict:
    return {event['id']: event for report in reports for event in report['events']
            if event['status'] in ['detected', 'missed']}


def _groups(records: list[dict], reports: list[dict]) -> dict:
    return {group: aggregate_reports([report for record, report in zip(records, reports)
                                     if record['group'] == group])['learned']
            for group in sorted({record['group'] for record in records})}


def study_plan(root: Path) -> dict:
    root = Path(root); archive = root / 'artifacts/v5'
    preservation = _read(archive / 'preservation_manifest.json')
    for name in SOURCE_NAMES:
        if _hash(archive / 'development' / name) != preservation['files']['development/' + name]:
            raise ValueError('Preserved V5 development source hash mismatch.')
    source = _read(archive / 'development/plan.json')
    selected = _read(archive / 'development/report.json')['selected']
    dependencies = [PROTOCOL, 'signalwatch/ml_branch_policy.py', 'tests/test_ml_branch_policy.py',
                    'scripts/study_branch_policy.py', 'signalwatch/alarm_policy.py',
                    'signalwatch/dataset.py', 'signalwatch/evaluation.py',
                    'artifacts/v5/preservation_manifest.json',
                    'outputs/v9-diagnosis/diagnostics.json', 'work/v9-diagnosis/diagnose.py']
    dependencies += ['artifacts/v5/development/' + name for name in SOURCE_NAMES]
    return {'study': 'branch-selective-confirmation-v9', 'schema_version': 1,
            'source_snapshot': SNAPSHOT, 'protocol_path': PROTOCOL,
            'protocol_sha256': _hash(root / PROTOCOL),
            'dependencies': {name: _hash(root / name) for name in dependencies},
            'development_ids': source['development_ids'], 'source_hashes': source['source_hashes'],
            'folds': source['folds'], 'classifier_threshold': selected['classifier_threshold'],
            'residual_threshold': selected['residual_threshold'], 'aggregation': 'max',
            'candidates': [{'confirmation_seconds': CONFIRMATION_SECONDS, 'classifier': 'immediate'}],
            'finite_mask': 'Both raw branches finite; either non-finite resets residual confirmation.',
            'constraints': {'min_detected_events': 26, 'strictly_fewer_false_than_reference': True,
                            'max_false_onsets_per_normal_hour': 2., 'max_normal_alert_seconds': 37.,
                            'observed_normal_seconds': 9179.},
            'known_sensitivity_cost': 'Diagnosis predicts loss of other-12-event-16; retain and report it.',
            'selection_scope': 'One fixed policy over preserved development OOF, no fitting or parameter grid.',
            'v6_difference': 'V6 confirmed entire fusion and required retaining all28 hits; this new protocol uses the established26/29 project proxy.',
            'old_test_access': False, 'model_fit': False, 'deployment_fit': False, 'activation': False}


def freeze_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/branch-policy-development'
    target.mkdir(parents=True, exist_ok=True)
    plan = study_plan(root); path = target / 'plan.json'
    if path.exists():
        if _read(path) != plan:
            raise ValueError('Frozen branch-policy protocol or dependencies changed.')
    else:
        _write(path, plan)
    return plan


def run_study(root: Path) -> dict:
    root = Path(root); target = root / 'artifacts/branch-policy-development'
    started = time.perf_counter(); plan = freeze_study(root)
    if (target / 'execution_started.json').exists() or (target / 'report.json').exists():
        raise ValueError('The one frozen policy execution already started; refusing another candidate run.')
    active = {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}
    _write(target / 'execution_started.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
           'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
           'environment': {'python': platform.python_version(), 'numpy': np.__version__},
           'active_hashes': active, 'dependencies': plan['dependencies'],
           'model_fits': 0, 'authorization': 'One fixed development policy execution after preflight QA.'})
    archive = root / 'artifacts/v5/development'
    source_report = _read(archive / 'report.json')
    classifier = _arrays(archive / 'classifier_oof_scores.npz')
    channels = _arrays(archive / 'residual_oof_channels.npz')
    saved_raw = _arrays(archive / 'selected_oof_scores.npz')
    ids = plan['development_ids']; expected = set(ids)
    if any(set(values) != expected for values in [classifier, channels, saved_raw]):
        raise ValueError('Source OOF arrays differ from exact development recording IDs.')
    records = [load_builtin(root, key) for key in ids]
    raw_reports = []; policy_reports = []; policies = {}; raw_scores = {}; decisions = {}
    normalized_classifier = {}; normalized_residual = {}; states = {}
    for record in records:
        key = record['id']; n = len(record['values'])
        if record['split'] != 'validation' or record['sha256'] != plan['source_hashes'][key]:
            raise ValueError('Only hash-matched development recordings may be evaluated.')
        if classifier[key].shape != (n,) or channels[key].shape != (n, 8) or saved_raw[key].shape != (n,):
            raise ValueError('Preserved OOF source shape mismatch.')
        c = classifier[key] / plan['classifier_threshold']
        r = channels[key].max(axis=1) / plan['residual_threshold']
        raw = np.maximum(c, r)
        np.testing.assert_allclose(raw, saved_raw[key], atol=0, rtol=0, equal_nan=True)
        base = _trace(raw, c, r)
        raw_reports.append(inspect_recording(record, base))
        policy = apply_to_scored(base, record); policies[key] = policy
        if not np.array_equal(np.isfinite(raw), np.isfinite(policy['scores']['learned'])):
            raise ValueError('Branch policy changed finite exposure.')
        if np.any((policy['scores']['learned'] > 1.) & ~(raw > 1.)):
            raise ValueError('Policy fires outside a raw positive observation.')
        policy_reports.append(inspect_recording(record, policy))
        raw_scores[key] = raw; decisions[key] = policy['scores']['learned']
        normalized_classifier[key] = c; normalized_residual[key] = r
        for field in ['pending', 'confirmed', 'raw_episode_start']:
            states[key + '__' + field] = policy['policy_state'][field]
    raw_metric = aggregate_reports(raw_reports)['learned']
    if raw_metric != source_report['metrics']:
        raise ValueError('Preserved V5 development metrics do not reproduce.')
    metric = aggregate_reports(policy_reports)['learned']
    reasons = screen_candidate(metric, raw_metric)
    raw_events = _event_map(raw_reports); policy_events = _event_map(policy_reports)
    raw_hits = {key for key, event in raw_events.items() if event['status'] == 'detected'}
    new_hits = {key for key, event in policy_events.items() if event['status'] == 'detected'}
    pairs = [{'id': key, 'raw_status': raw_events[key]['status'], 'policy_status': policy_events[key]['status'],
              'raw_first_alert_seconds': raw_events[key]['first_alert_seconds'],
              'policy_first_alert_seconds': policy_events[key]['first_alert_seconds'],
              'raw_delay_seconds': raw_events[key]['delay_seconds'],
              'policy_delay_seconds': policy_events[key]['delay_seconds']} for key in raw_events]
    point = {'confirmation_seconds': CONFIRMATION_SECONDS, 'classifier': 'immediate', 'metrics': metric,
             'feasible': not reasons, 'rejection_reasons': reasons,
             'lost_event_ids': sorted(raw_hits - new_hits), 'gained_event_ids': sorted(new_hits - raw_hits),
             'paired_outcomes': pairs, 'group_slices': _groups(records, policy_reports),
             'per_recording': [{'id': record['id'], 'group': record['group'],
                                'metrics': report['metrics']['learned'], 'events': report['events']}
                               for record, report in zip(records, policy_reports)]}
    arrays = {'raw_oof_scores.npz': raw_scores, 'decision_oof_scores.npz': decisions,
              'classifier_ratio_oof_scores.npz': normalized_classifier,
              'residual_ratio_oof_scores.npz': normalized_residual, 'policy_state.npz': states}
    for name, values in arrays.items():
        np.savez_compressed(target / name, **values)
    _write(target / 'candidate_audits.json', [point])
    if active != {name: _hash(root / 'artifacts' / name) for name in ACTIVE_FILES}:
        raise ValueError('Protected active V5 artifacts changed.')
    if any(_hash(root / name) != digest for name, digest in plan['dependencies'].items()):
        raise ValueError('Frozen source dependency changed during policy execution.')
    report = {'schema_version': 1, 'scope': 'development_only_fixed_branch_policy',
              'status': 'development_candidate_only' if not reasons else 'no_feasible_candidate',
              'selected': point if not reasons else None, 'candidate': point,
              'raw_metrics': raw_metric, 'raw_group_slices': _groups(records, raw_reports),
              'raw_per_recording': [{'id': record['id'], 'group': record['group'],
                                    'metrics': report['metrics']['learned'], 'events': report['events']}
                                   for record, report in zip(records, raw_reports)],
              'plan_sha256': _hash(target / 'plan.json'), 'protocol_sha256': plan['protocol_sha256'],
              'candidate_audits_sha256': _hash(target / 'candidate_audits.json'),
              'score_artifacts_sha256': {name: _hash(target / name) for name in arrays},
              'active_files_unchanged': True, 'dependencies_unchanged': True,
              'model_fits': 0, 'old_test_scored': False, 'deployment_fit': False, 'activation': False,
              'study_seconds': round(time.perf_counter() - started, 3),
              'limits': ['Repeated development-derived rule; no independent generalization estimate.',
                         'Alarm policy on unchanged trained models, not new ML training.',
                         'Immediate classifier differs from V6 global confirmation; the26/29 target proxy permits a documented loss.',
                         'Decision-score AP is not unchanged raw model discrimination.',
                         'Separate recording comparison does not authorize activation.']}
    _write(target / 'report.json', report)
    return report
