"""Ordinary point scores and literal alert/event overlap; no point adjustment."""
from __future__ import annotations
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from signalwatch.dataset import SENSORS

METHODS = ('learned', 'robust', 'persistence')


def _top_sensors(contributions: np.ndarray) -> list[str]:
    if not contributions.size or not np.isfinite(contributions).any():
        return []
    finite=np.isfinite(contributions)
    means=np.sum(np.where(finite,contributions,0.),axis=0)/np.maximum(finite.sum(axis=0),1)
    return [SENSORS[i]['key'] for i in np.argsort(means)[-3:][::-1] if means[i]>0]


def intervals(mask: np.ndarray, gaps: np.ndarray) -> list[tuple[int, int]]:
    result, start = [], None
    for i, active in enumerate(mask):
        if start is not None and (not active or gaps[i]):
            result.append((start, i - 1))
            start = None
        if active and start is None:
            start = i
    if start is not None:
        result.append((start, len(mask) - 1))
    return result


def durations(record: dict) -> np.ndarray:
    elapsed = record['elapsed']
    if not len(elapsed):
        return np.zeros(0)
    dt = np.r_[np.diff(elapsed), 0.]
    dt[np.r_[record['gap_before'][1:], False]] = 0.
    return dt


def inspect_recording(recording: dict, scored: dict, protocol: str = 'onset_v2') -> dict:
    if protocol not in ['onset_v2','legacy_overlap_v1']:
        raise ValueError('Unknown evaluation protocol.')
    n = len(recording['values'])
    elapsed = recording['elapsed']
    gaps = recording['gap_before']
    labels = recording['labels']
    annotated = intervals(labels, gaps) if labels is not None else []
    annotations = [{'id': f'annotation-{j}', 'start_index': a, 'end_index': b,
                    'start_seconds': float(elapsed[a]), 'end_seconds': float(elapsed[b])}
                   for j, (a, b) in enumerate(annotated)]
    all_episodes, metrics, events = {}, {}, []
    dt = durations(recording)
    contributions = scored.get('contributions', np.zeros((n, 8)))
    for method in METHODS:
        scores = np.asarray(scored['scores'][method])
        finite = np.isfinite(scores)
        alerts = finite & (scores > scored['thresholds'][method])
        segments = intervals(alerts, gaps)
        onsets=np.array([a for a,b in segments],dtype=int)
        episodes = []
        for j, (a, b) in enumerate(segments):
            episode = {'id': f'{method}-{j}', 'start_index': a, 'end_index': b,
                       'start_seconds': float(elapsed[a]), 'end_seconds': float(elapsed[b]),
                       'ongoing': b == n - 1, 'alert_samples': b - a + 1}
            episodes.append(episode)
        all_episodes[method] = episodes
        if labels is not None:
            if protocol=='legacy_overlap_v1':
                delays = [float(elapsed[a + np.flatnonzero(alerts[a:b+1])[0]] - elapsed[a])
                          for a,b in annotated if alerts[a:b+1].any()]
                false_count = sum(not labels[a:b+1].any() for a,b in segments)
            else:
                delays=[float(elapsed[onsets[(onsets>=a)&(onsets<=b)][0]]-elapsed[a])
                        for a,b in annotated if np.any((onsets>=a)&(onsets<=b))]
                false_count=sum(not labels[a] for a,b in segments)
            normal_seconds = float(dt[~labels & finite].sum())
            eligible_seconds = float(dt[finite].sum())
            quality = {'total_events': len(annotated), 'detected_events': len(delays),
                       'missed_events': len(annotated)-len(delays), 'false_episodes': false_count,
                       'observed_normal_seconds': normal_seconds,
                       'total_normal_seconds': float(dt[~labels].sum()),
                       'observed_seconds': eligible_seconds,
                       'false_episodes_per_hour': false_count * 3600 / normal_seconds if normal_seconds else None,
                       'mean_delay_seconds': float(np.mean(delays)) if delays else None,
                       'median_delay_seconds': float(np.median(delays)) if delays else None,
                       'delay_seconds': delays,
                       'alert_seconds': float(dt[alerts].sum()),
                       'false_alert_seconds': float(dt[alerts & ~labels].sum()),
                       'normal_alert_fraction': float(dt[alerts & ~labels].sum()/normal_seconds) if normal_seconds else None,
                       'true_positive_samples': int(np.sum(alerts & labels)),
                       'false_positive_samples': int(np.sum(alerts & ~labels)),
                       'point_precision': float(np.sum(alerts & labels)/np.sum(alerts)) if alerts.any() else None,
                       'point_recall': float(np.sum(alerts & labels)/np.sum(finite & labels)) if np.any(finite & labels) else None,
                       'alert_fraction': float(dt[alerts].sum()/eligible_seconds) if eligible_seconds else None,
                       'scored_samples': int(finite.sum()),
                       'average_precision': float(average_precision_score(labels[finite],scores[finite]))
                           if finite.any() and labels[finite].any() and (~labels[finite]).any() else None}
            metrics[method] = quality
        if method != 'learned':
            continue
        for j,(a,b) in enumerate(annotated):
            found = np.flatnonzero(alerts[a:b+1]) if protocol=='legacy_overlap_v1' else onsets[(onsets>=a)&(onsets<=b)]
            first = (a+int(found[0]) if protocol=='legacy_overlap_v1' else int(found[0])) if len(found) else None
            events.append({'id': f'{recording["id"]}-event-{j}', 'title': f'Annotated interval {j+1}',
                           'status': 'detected' if first is not None else 'missed',
                           'start_index':a,'end_index':b,'start_seconds':float(elapsed[a]),'end_seconds':float(elapsed[b]),
                           'first_alert_seconds':float(elapsed[first]) if first is not None else None,
                           'delay_seconds':float(elapsed[first]-elapsed[a]) if first is not None else None,
                           'method':method,'contributing_sensors':_top_sensors(contributions[a:b+1]),
                           'baseline_overlap':{key:bool(np.any(np.isfinite(scored['scores'][key][a:b+1]) & (scored['scores'][key][a:b+1]>scored['thresholds'][key]))) for key in METHODS[1:]},
                           'evidence':scored.get('contributions_note','Largest normalized prediction residuals; not a diagnosis.')})
        for j,(a,b) in enumerate(segments):
            if labels is not None and (labels[a:b+1].any() if protocol=='legacy_overlap_v1' else labels[a]):
                continue
            events.append({'id':f'{recording["id"]}-alert-{j}','title':f'Alert episode {j+1}',
                           'status':'false_alert' if labels is not None else 'unlabelled',
                           'start_index':a,'end_index':b,'start_seconds':float(elapsed[a]),'end_seconds':float(elapsed[b]),
                           'first_alert_seconds':float(elapsed[a]),'delay_seconds':None,'method':method,
                           'contributing_sensors':_top_sensors(contributions[a:b+1]), 'ongoing':b==n-1,
                           'baseline_overlap':{key:bool(np.any(np.isfinite(scored['scores'][key][a:b+1]) & (scored['scores'][key][a:b+1]>scored['thresholds'][key]))) for key in METHODS[1:]},
                           'evidence':'Alert episode started outside a known annotation; later overlap does not turn it into a fresh detection.' if labels is not None else 'No labels supplied; correctness is unknown.'})
    for event in events:
        event['contributions_kind']=scored.get('contributions_kind','prediction_residual')
        if scored.get('contributions_note'):
            event['contributions_note']=scored['contributions_note']
    events.sort(key=lambda item:(item['start_index'],item['id']))
    return {'annotations':annotations,'episodes':all_episodes,'metrics':metrics if labels is not None else None,'events':events,'evaluation_protocol':protocol}


def aggregate_reports(reports: list[dict]) -> dict:
    output={}
    for method in METHODS:
        rows=[report['metrics'][method] for report in reports if report['metrics'] is not None]
        sums={key:sum(row[key] for row in rows) for key in ['total_events','detected_events','missed_events','false_episodes','observed_normal_seconds','total_normal_seconds','observed_seconds','alert_seconds','false_alert_seconds','true_positive_samples','false_positive_samples','scored_samples']}
        delays=[delay for row in rows for delay in row['delay_seconds']]
        aps=[row['average_precision'] for row in rows if row['average_precision'] is not None]
        sums.update(false_episodes_per_hour=sums['false_episodes']*3600/sums['observed_normal_seconds'] if sums['observed_normal_seconds'] else None,
                    mean_delay_seconds=float(np.mean(delays)) if delays else None,
                    median_delay_seconds=float(np.median(delays)) if delays else None,
                    alert_fraction=sums['alert_seconds']/sums['observed_seconds'] if sums['observed_seconds'] else None,
                    normal_alert_fraction=sums['false_alert_seconds']/sums['observed_normal_seconds'] if sums['observed_normal_seconds'] else None,
                    point_precision=sums['true_positive_samples']/(sums['true_positive_samples']+sums['false_positive_samples']) if sums['true_positive_samples']+sums['false_positive_samples'] else None,
                    average_precision=float(np.mean(aps)) if aps else None,
                    average_precision_aggregation='Unweighted mean of per-recording ordinary average precision; single-class recordings excluded.',
                    event_recall=sums['detected_events']/sums['total_events'] if sums['total_events'] else None)
        output[method]=sums
    return output


def compare_validation(root: Path, bundle: dict, budget: float) -> dict:
    if not .25 <= budget <= 10:
        raise ValueError('Alert budget must be between 0.25 and 10 per hour.')
    if bundle.get('study_version') in (3, 4):
        from signalwatch.release import study_module
        return study_module(bundle['study_version']).compare_development_budget(root, bundle['supervised_bundle'], budget)
    if bundle.get('study_version') == 2:
        from signalwatch.ml_v2 import compare_development_budget
        return compare_development_budget(root, bundle['innovation_bundle'], budget)
    from signalwatch.ml import calibration_record, calibrate, score_recording, evaluate_split
    normal=calibration_record(root, bundle['split_manifest'])
    thresholds, calibration = calibrate(normal,score_recording(bundle,normal),budget)
    trial=dict(bundle,thresholds=thresholds)
    return {'scope':'validation_only','budget':budget,'thresholds':thresholds,'calibration':calibration,
            'methods':evaluate_split(root,trial,'validation')['methods'],
            'note':'Exploratory validation results. Frozen final test artifact is unchanged.'}
