"""Development-only selection of an elapsed-time policy over fixed V5 ML."""
from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

from signalwatch.alarm_policy import apply_confirmation
from signalwatch.dataset import SNAPSHOT
from signalwatch.evaluation import aggregate_reports
from signalwatch.ml import score_recording_v1
from signalwatch.ml_hybrid import (
    CLASSIFIER_CONFIG, CLASSIFIER_THRESHOLD, INNOVATION_CONFIG,
    score_recording as score_raw_hybrid,
)
from signalwatch.ml_supervised import CONTRIBUTIONS_NOTE
from signalwatch.ml_supervised_weighted import fit_model as fit_classifier
from signalwatch.ml_v2 import _fit as fit_innovation, _load_development, _point, _write_json

CONFIRMATION_SECONDS=[0.,1.,2.,5.,10.]
SOURCE_WHITELIST=['model.joblib','metadata.json','development/plan.json','development/report.json',
                  'development/selected_oof_scores.npz','development/selected_bundle.joblib']


def apply_to_scored(raw_scored: dict,record: dict,confirmation_seconds: float) -> dict:
    policy=apply_confirmation(raw_scored['scores']['learned'],record['elapsed'],record['gap_before'],confirmation_seconds)
    result=dict(raw_scored)
    result['scores']=dict(raw_scored['scores'],learned=policy['decision_score'])
    result['thresholds']=dict(raw_scored['thresholds'],learned=1.)
    result['raw_scores']={'learned':policy['raw_score']}
    result['alarm_policy']={'kind':'continuous_elapsed_confirmation','confirmation_seconds':float(confirmation_seconds)}
    result['policy_state']={key:policy[key] for key in ['pending','confirmed','raw_episode_start']}
    result['learned_version']='confirmed-6.0.0'
    return result


def score_recording(bundle: dict,record: dict,base_scored: dict) -> dict:
    raw=score_raw_hybrid(bundle,record,base_scored)
    return apply_to_scored(raw,record,bundle['alarm_policy']['confirmation_seconds'])


def _rank_policy(point: dict) -> tuple:
    metric=point['metrics']
    median=metric['median_delay_seconds'];mean=metric['mean_delay_seconds']
    return (metric['false_episodes'],metric['normal_alert_fraction'],
            median if median is not None else float('inf'),mean if mean is not None else float('inf'),point['index'])


def choose_policy(points: list[dict]) -> dict | None:
    feasible=[point for point in points if point['feasible']]
    return min(feasible,key=_rank_policy) if feasible else None


def _verify_archive(root: Path) -> dict:
    archive=Path(root)/'artifacts/v5'
    preservation=json.loads((archive/'preservation_manifest.json').read_text(encoding='utf-8'))
    # Only training/template/development provenance is read. Retrospective files
    # are deliberately absent from this whitelist, including frozen_evaluation.
    for name in SOURCE_WHITELIST:
        expected=preservation['files'].get(name)
        if expected is None or hashlib.sha256((archive/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Preserved V5 training/development source hash mismatch.')
    return preservation


def _load_source(root: Path) -> tuple[dict,dict,dict]:
    _verify_archive(root)
    archive=Path(root)/'artifacts/v5'
    report=json.loads((archive/'development/report.json').read_text(encoding='utf-8'))
    score_path=archive/'development/selected_oof_scores.npz'
    if hashlib.sha256(score_path.read_bytes()).hexdigest()!=report['oof_scores_sha256']:
        raise ValueError('Raw V5 OOF score hash differs from its development report.')
    wrapper=joblib.load(archive/'model.joblib')
    template=copy.deepcopy(wrapper['supervised_bundle'])
    if template['candidate']['classifier']!=CLASSIFIER_CONFIG or template['candidate']['innovation']!=INNOVATION_CONFIG:
        raise ValueError('V5 fitted branch configurations differ from the fixed study.')
    if template['classifier_threshold']!=CLASSIFIER_THRESHOLD or template['threshold']!=1.:
        raise ValueError('V5 raw thresholds differ from the fixed study.')
    with np.load(score_path,allow_pickle=False) as stored:
        scores={key:stored[key].copy() for key in stored.files}
    return template,report,scores


def _dependencies(root: Path) -> dict:
    archive_paths=['artifacts/v5/preservation_manifest.json']+[f'artifacts/v5/{name}' for name in SOURCE_WHITELIST]
    prior=[f'artifacts/v1/{name}' for name in ['preservation_manifest.json','model.joblib','metadata.json','split_manifest.json']]
    science=['signalwatch/alarm_policy.py','signalwatch/ml_confirmed.py','signalwatch/ml_hybrid.py',
             'signalwatch/ml_v2.py','signalwatch/ml_supervised.py','signalwatch/ml_supervised_weighted.py',
             'signalwatch/dataset.py','signalwatch/evaluation.py']
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in archive_paths+prior+science}


def study_plan(root: Path) -> dict:
    root=Path(root);_verify_archive(root)
    source=json.loads((root/'artifacts/v5/development/plan.json').read_text(encoding='utf-8'))
    return {'study':'confirmed-v6','schema_version':1,'source_snapshot':SNAPSHOT,
            'dependencies':_dependencies(root),'development_ids':source['development_ids'],
            'source_hashes':source['source_hashes'],'folds':source['folds'],
            'confirmation_seconds':CONFIRMATION_SECONDS,'raw_threshold':1.,
            'policy':'Consecutive OBSERVED finite raw scores strictly above1 spanning D elapsed seconds. This does not assert the analog signal stayed positive between observations.',
            'activation':'Emit only at the current observed positive row where elapsed-current minus elapsed-pending-start >=D. Never backfill.',
            'recovery':'Raw score<=1, nonfinite score or timestamp gap resets pending/confirmed state. No cooldown, delayed recovery, periodic rearming or label input.',
            'scores':'Decision score is min(raw,1) while pending, otherwise actual raw score. Preserve every nonfinite value and finite exposure; D=0 is exact identity.',
            'provenance':'raw_episode_start is the first row index of the uninterrupted observed positive raw episode, while pending or confirmed; otherwise -1. Confirmed episode onset is its actual emission row.',
            'feasibility':{'retain':'Every annotation event ID detected by raw OOF must also be detected after confirmation.',
                           'false_onsets_per_normal_hour_max':2.,'normal_alert_fraction_max':.05,
                           'median_delay_increase_max_seconds':10.,'scored_normal_exposure':'EXACT equality to raw control'},
            'ranking':'Fewer false onsets, lower normal-alert fraction, lower median delay, lower mean delay, then declared candidate order.',
            'promotion':'Promote only a feasible nonidentity point with strictly fewer false onsets or lower normal-alert time than raw identity. Otherwise retain V5 and skip retrospective scoring.',
            'event_protocol':'onset_v2 unchanged. Normal pending start -> anomalous confirmation can be a hit; anomalous pending start -> normal confirmation is false. Never inherit raw event status.',
            'raw_metric_separation':'methods refer to decision scores. raw_methods describe original V5 scores. Capped decision-score AP is not unchanged model discrimination.',
            'classifier_config':CLASSIFIER_CONFIG,'innovation_config':INNOVATION_CONFIG,
            'branch_thresholds':'Keep preserved V5 classifier and residual thresholds and aggregation exactly.',
            'final_refit':'After policy selection, fit both fixed branches on original development+normal train anchor. Policy is not selected using these in-sample scores. Compare refit raw scores with archived V5 template on development only for deterministic reproduction.',
            'report_scope':'Repeated OOF DEVELOPMENT decision-policy selection; not independent validation or a blind test.',
            'forbidden_access':['artifacts/v5/audit','artifacts/v5/frozen_evaluation.json','old test recordings'],
            'old_test_access':False}


def _event_map(reports: list[dict]) -> dict:
    return {event['id']:event for report in reports for event in report['events'] if event['status'] in ['detected','missed']}


def verify_development_sources(plan: dict,records: list[dict],anchor: dict,calibration: dict) -> None:
    if {record['id'] for record in records}!=set(plan['development_ids']):
        raise ValueError('Loaded development IDs differ from the frozen source set.')
    for record in records:
        if record['sha256']!=plan['source_hashes'][record['id']]:
            raise ValueError('Loaded development source hash differs from the frozen protocol.')
    expected_normal=plan['source_hashes']['anomaly-free']
    if anchor['sha256']!=expected_normal or calibration['sha256']!=expected_normal:
        raise ValueError('Loaded normal source hash differs from the frozen protocol.')


def _make_raw_trace(base: dict,score: np.ndarray) -> dict:
    output=dict(base,scores=dict(base['scores'],learned=score),thresholds=dict(base['thresholds'],learned=1.))
    output['contributions']=np.full((len(score),8),np.nan)
    output['contributions_kind']='unavailable_in_saved_oof'
    output['contributions_note']='Saved OOF model score; no fold-specific channel attribution is claimed.'
    return output


def evaluate_policies(records: list[dict],raw_scores: dict,baseline: dict,budget: float=2.) -> tuple[dict | None,list[dict],dict]:
    if not .25<=budget<=10.:
        raise ValueError('Exploratory budget must be between0.25 and10 false onsets per normal hour.')
    raw_traces={record['id']:_make_raw_trace(baseline[record['id']],raw_scores[record['id']]) for record in records}
    raw_metric,raw_reports=_point(records,raw_traces,1.)
    raw_events=_event_map(raw_reports)
    raw_hits={key for key,event in raw_events.items() if event['status']=='detected'}
    candidates=[];details={}
    for index,duration in enumerate(CONFIRMATION_SECONDS):
        traces={record['id']:apply_to_scored(raw_traces[record['id']],record,duration) for record in records}
        for record in records:
            key=record['id']
            if not np.array_equal(np.isfinite(traces[key]['scores']['learned']),np.isfinite(raw_scores[key])):
                raise ValueError('Policy changed finite scored exposure.')
            if np.any(traces[key]['policy_state']['confirmed'] & ~(np.isfinite(raw_scores[key])&(raw_scores[key]>1.))):
                raise ValueError('Confirmed alarm is not a subset of actual raw positive scores.')
        metrics,reports=_point(records,traces,1.)
        events=_event_map(reports);hits={key for key,event in events.items() if event['status']=='detected'}
        lost=sorted(raw_hits-hits);gained=sorted(hits-raw_hits)
        exposure_equal=metrics['observed_normal_seconds']==raw_metric['observed_normal_seconds']
        subset_time=metrics['false_alert_seconds']<=raw_metric['false_alert_seconds']
        if not exposure_equal or not subset_time:
            raise ValueError('Policy violated unchanged normal exposure or alert-time subset invariant.')
        median=metrics['median_delay_seconds'];raw_median=raw_metric['median_delay_seconds']
        delay_ok=median is not None and raw_median is not None and median<=raw_median+10.
        reasons=[]
        if lost:reasons.append('lost_raw_detected_event_ids')
        if metrics['false_episodes_per_hour'] is None or metrics['false_episodes_per_hour']>budget:reasons.append('false_onset_budget')
        if metrics['normal_alert_fraction'] is None or metrics['normal_alert_fraction']>.05:reasons.append('normal_alert_time_budget')
        if not delay_ok:reasons.append('median_delay_budget')
        pairs=[]
        for key,raw_event in raw_events.items():
            event=events[key]
            onset=event['first_alert_seconds'];start_index=None;pending_seconds=None
            if onset is not None:
                recording_id=next(record['id'] for record in records if key.startswith(record['id']+'-event-'))
                record=next(record for record in records if record['id']==recording_id)
                emitted=int(np.flatnonzero(record['elapsed']==onset)[0])
                start_index=int(traces[recording_id]['policy_state']['raw_episode_start'][emitted])
                pending_seconds=float(record['elapsed'][start_index]) if start_index>=0 else None
            pairs.append({'id':key,'raw_status':raw_event['status'],'confirmed_status':event['status'],
                          'raw_first_alert_seconds':raw_event['first_alert_seconds'],'confirmed_first_alert_seconds':onset,
                          'raw_delay_seconds':raw_event['delay_seconds'],'confirmed_delay_seconds':event['delay_seconds'],
                          'raw_episode_start_index_at_emission':start_index,'pending_start_seconds_at_emission':pending_seconds})
        point={'index':index,'threshold':1.,'confirmation_seconds':duration,'metrics':metrics,
               'feasible':not reasons,'rejection_reasons':reasons,'raw_detected_event_ids':sorted(raw_hits),
               'confirmed_detected_event_ids':sorted(hits),'lost_event_ids':lost,'gained_event_ids':gained,
               'normal_exposure_equal':exposure_equal,'paired_outcomes':pairs}
        candidates.append(point);details[index]={'traces':traces,'reports':reports}
    control={'metrics':raw_metric,'methods':aggregate_reports(raw_reports),'reports':raw_reports,
             'traces':raw_traces,'raw_detected_event_ids':sorted(raw_hits)}
    return choose_policy(candidates),candidates,{'control':control,'candidates':details}


def run_study(root: Path) -> dict:
    root=Path(root);target=root/'artifacts/confirmed-development';target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();plan=study_plan(root);path=target/'plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:
            raise ValueError('Frozen confirmation protocol or dependencies changed.')
    else:
        _write_json(path,plan)
    template,source_report,raw_scores=_load_source(root)
    legacy,records,anchor,calibration=_load_development(root)
    verify_development_sources(plan,records,anchor,calibration)
    if set(raw_scores)!={record['id'] for record in records}:
        raise ValueError('Saved raw OOF scores differ from the development recording set.')
    baseline={record['id']:score_recording_v1(legacy,record) for record in records}
    selected,candidates,details=evaluate_policies(records,raw_scores,baseline)
    control=details['control'];raw=control['metrics']
    if raw!=source_report['metrics']:
        raise ValueError('Preserved V5 OOF metrics do not reproduce the source development report.')
    for point in candidates:
        reports=details['candidates'][point['index']]['reports']
        point['group_slices']={group:aggregate_reports([result for record,result in zip(records,reports) if record['group']==group])
                               for group in sorted({record['group'] for record in records})}
        point['fold_slices']={str(fold['fold']):aggregate_reports([result for record,result in zip(records,reports) if record['id'] in fold['heldout_ids']])
                              for fold in plan['folds']}
        point['per_recording']=[{'id':record['id'],'group':record['group'],'metrics':result['metrics'],'events':result['events']}
                                for record,result in zip(records,reports)]
        point['methods']=aggregate_reports(reports)
    _write_json(target/'candidate_audits.json',candidates)
    report={'schema_version':1,'scope':'confirmed_policy_oof_development_selection','selected':selected,
            'source_snapshot':SNAPSHOT,'old_test_scored':False,'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_raw_model_sha256':source_report['model_sha256'],'source_raw_oof_sha256':source_report['oof_scores_sha256'],
            'candidate_audits_sha256':hashlib.sha256((target/'candidate_audits.json').read_bytes()).hexdigest(),
            'source_bundle':{'archive':'artifacts/v5','model_sha256':plan['dependencies']['artifacts/v5/model.joblib'],
                             'development_model_sha256':source_report['model_sha256'],
                             'oof_scores_sha256':source_report['oof_scores_sha256'],
                             'training_fixtures':{'development_ids':plan['development_ids'],
                                                  'source_hashes':plan['source_hashes'],
                                                  'anchor':'original normal train block',
                                                  'residual_scale':'original separate normal calibration block'}},
            'raw_methods':control['methods'],'raw_detected_event_ids':control['raw_detected_event_ids'],
            'limitations':['Confirmation is a separately selected temporal decision policy, not a new learned ML architecture.',
                          'Consecutive observed positives do not establish a continuously positive analog signal between observations.',
                          'Repeated OOF policy selection is not independent validation; old retrospective results are not used for parameter selection.',
                          'Decision-score AP after capping pending rows must not be presented as unchanged raw-model discrimination.',
                          'Confirmed onset delay and raw pending start may straddle annotations; actual emission determines event status.']}
    promoted=selected is not None and selected['index']!=0 and (
        selected['metrics']['false_episodes']<raw['false_episodes'] or
        selected['metrics']['normal_alert_fraction']<raw['normal_alert_fraction'])
    report['promote']=bool(promoted)
    if selected is not None:
        chosen=details['candidates'][selected['index']]
        report.update(metrics=selected['metrics'],methods=aggregate_reports(chosen['reports']),
                      group_slices=selected['group_slices'],fold_slices=selected['fold_slices'],
                      per_recording=selected['per_recording'],paired_outcomes=selected['paired_outcomes'])
        np.savez_compressed(target/'selected_oof_scores.npz',**{key:trace['scores']['learned'] for key,trace in chosen['traces'].items()})
        np.savez_compressed(target/'selected_policy_state.npz',**{f'{key}__{name}':trace['policy_state'][name]
                            for key,trace in chosen['traces'].items() for name in ['pending','confirmed','raw_episode_start']})
        report['oof_scores_sha256']=hashlib.sha256((target/'selected_oof_scores.npz').read_bytes()).hexdigest()
        report['policy_state_sha256']=hashlib.sha256((target/'selected_policy_state.npz').read_bytes()).hexdigest()
    (target/'raw_oof_scores.npz').write_bytes((root/'artifacts/v5/development/selected_oof_scores.npz').read_bytes())
    report['raw_oof_scores_sha256']=hashlib.sha256((target/'raw_oof_scores.npz').read_bytes()).hexdigest()
    _write_json(target/'selection_before_refit.json',report)
    if not promoted:
        report['status']='retain_v5' if selected is not None else 'no_feasible_policy'
    else:
        classifier,class_evidence=fit_classifier([anchor]+records,CLASSIFIER_CONFIG)
        innovation,innovation_evidence=fit_innovation([anchor]+records,calibration,INNOVATION_CONFIG)
        fitted=copy.deepcopy(template);fitted.pop('study_report',None)
        fitted['classifier_bundle']=classifier;fitted['innovation_bundle']=innovation
        fitted['alarm_policy']={'kind':'continuous_elapsed_confirmation','confirmation_seconds':selected['confirmation_seconds']}
        fitted['candidate']=dict(template['candidate'],alarm_policy=fitted['alarm_policy'])
        fitted['version']='confirmed-6.0.0';fitted['plan']=plan
        fitted['training_evidence']={'classifier':class_evidence,'innovation':innovation_evidence}
        fitted['parameters']=dict(fitted.get('parameters',{}),alarm_policy=fitted['alarm_policy'])
        fitted['source_bundle']=report['source_bundle']
        fitted['score_note']='Confirmed decision score over the unchanged raw fused ML ratio; pending scores are capped at1. Raw scores remain available separately.'
        checks=[]
        for record in records:
            expected=score_raw_hybrid(template,record,baseline[record['id']])['scores']['learned']
            actual=score_raw_hybrid(fitted,record,baseline[record['id']])['scores']['learned']
            np.testing.assert_allclose(actual,expected,atol=1e-8,rtol=0,equal_nan=True)
            checks.append(record['id'])
        joblib.dump(fitted,target/'selected_bundle.joblib',compress=3)
        report['model_sha256']=hashlib.sha256((target/'selected_bundle.joblib').read_bytes()).hexdigest()
        report['final_fit']={'training_rows':class_evidence['training_rows'],'classifier_training_rows':class_evidence['training_rows'],
                             'innovation_training_rows':innovation_evidence['training_rows'],'feature_count':len(fitted['feature_names']),
                             'feature_names':fitted['feature_names'],
                             'classifier':class_evidence,'innovation':innovation_evidence,
                             'raw_refit_matches_archived_template_on_development':checks}
        report['status']='selected_confirmed_policy'
    report['study_seconds']=round(time.perf_counter()-started,3)
    _write_json(target/'report.json',report)
    return report


def load_development_bundle(root: Path) -> dict:
    target=Path(root)/'artifacts/confirmed-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    if not report['promote']:
        raise ValueError('The confirmation study retained V5; no new confirmed model is promoted.')
    path=target/'selected_bundle.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['model_sha256']:
        raise ValueError('Confirmed model hash mismatch.')
    model=joblib.load(path);model['study_report']=report
    return model


def compare_development_budget(root: Path,bundle: dict,budget: float) -> dict:
    if not .25<=budget<=10.:
        raise ValueError('Exploratory budget must be between0.25 and10 false onsets per normal hour.')
    target=Path(root)/'artifacts/confirmed-development'
    report=bundle['study_report']
    path=target/'candidate_audits.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['candidate_audits_sha256']:
        raise ValueError('Saved confirmation candidate hash mismatch.')
    points=json.loads(path.read_text(encoding='utf-8'))
    for point in points:
        reasons=[reason for reason in point['rejection_reasons'] if reason!='false_onset_budget']
        rate=point['metrics']['false_episodes_per_hour']
        if rate is None or rate>budget:reasons.append('false_onset_budget')
        point['rejection_reasons']=reasons;point['feasible']=not reasons
    selected=choose_policy(points)
    if selected is None:
        return {'scope':report['scope'],'budget':budget,'selected':None,'methods':None,
                'raw_methods':report['raw_methods'],'feasible':False,
                'note':'No frozen confirmation policy meets the exploratory budget and retained-hit constraints.'}
    return {'scope':report['scope'],'budget':budget,'feasible':True,'selected':selected,
            'methods':selected['methods'],'raw_methods':report['raw_methods'],
            'calibration':{'learned':selected},'note':'OOF decision-policy exploration only; raw ML and active policy remain unchanged.'}
