"""Nested recording-disjoint sigmoid-calibrated classifier ensembles."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from threadpoolctl import threadpool_limits

from signalwatch.dataset import SNAPSHOT
from signalwatch.evaluation import aggregate_reports
from signalwatch.ml import score_recording_v1
from signalwatch.ml_hybrid import (
    AGGREGATIONS, CLASSIFIER_CONFIG, INNOVATION_CONFIG, _hybrid_feature_names,
    aggregate_residuals, fuse_scores,
)
from signalwatch.ml_supervised import CONTRIBUTIONS_NOTE, causal_features, classifier_scores
from signalwatch.ml_supervised_weighted import fit_model as fit_classifier
from signalwatch.ml_v2 import _fit as fit_innovation, _load_development, _point, _write_json, innovation_scores

QUANTILES=[.95,.975,.99,.995,.999,1.]
SOURCE_NAMES=['development/plan.json','development/report.json','development/residual_oof_channels.npz',
              'development/selected_oof_scores.npz']


def inner_splits(ids: list[str]) -> list[dict]:
    ordered=sorted(ids)
    if len(ordered)!=len(set(ordered)) or len(ordered)<3:
        raise ValueError('Inner calibration requires at least three unique recording IDs.')
    return [{'member':member,'fit_ids':[key for index,key in enumerate(ordered) if index%3!=member],
             'calibration_ids':[key for index,key in enumerate(ordered) if index%3==member]}
            for member in range(3)]


def calibration_weights(labels_by_record: list[np.ndarray]) -> np.ndarray:
    counts=[len(values) for values in labels_by_record]
    if not counts or min(counts)<=0:
        raise ValueError('Calibration needs eligible rows in each calibration recording.')
    total=sum(counts)
    return np.concatenate([np.full(count,total/(len(counts)*count)) for count in counts])


def fit_ensemble(records: list[dict],anchor: dict,forbidden_ids: list[str] | None=None) -> tuple[list[dict],dict]:
    by_id={record['id']:record for record in records}
    if len(by_id)!=len(records) or anchor['id'] in by_id:
        raise ValueError('Recording IDs and the fit-only anchor must be unique.')
    forbidden=set(forbidden_ids or [])
    if forbidden&(set(by_id)|{anchor['id']}):
        raise ValueError('Outer-held recording entered nested fitting or calibration.')
    members=[];evidence=[]
    for split in inner_splits(list(by_id)):
        fitting=[anchor]+[by_id[key] for key in split['fit_ids']]
        calibration=[by_id[key] for key in split['calibration_ids']]
        fitted,fit_evidence=fit_classifier(fitting,CLASSIFIER_CONFIG)
        designs=[];labels=[];counts={}
        for record in calibration:
            if record.get('labels') is None:
                raise ValueError('Explicit calibration labels are required.')
            X,indices,_=causal_features(record)
            y=np.asarray(record['labels'],int)[indices]
            if not np.isin(y,[0,1]).all():
                raise ValueError('Calibration labels must be binary.')
            designs.append(X);labels.append(y);counts[record['id']]=len(indices)
        X=np.concatenate(designs);y=np.concatenate(labels)
        if len(np.unique(y))!=2:
            raise ValueError('Both classes must occur in each calibration partition.')
        weights=calibration_weights(labels)
        transformed=(X-fitted['feature_center'])/fitted['feature_scale']
        calibrator=CalibratedClassifierCV(FrozenEstimator(fitted['model']),method='sigmoid')
        with threadpool_limits(limits=2):
            before=fitted['model'].predict_proba(transformed).copy()
            calibrator.fit(transformed,y,sample_weight=weights)
            np.testing.assert_array_equal(fitted['model'].predict_proba(transformed),before)
        member=dict(fitted,model=calibrator,version='calibrated-pair-7.0.0')
        members.append(member)
        evidence.append({'member':split['member'],'fit_ids':fit_evidence['training_ids'],
                         'calibration_ids':split['calibration_ids'],'outer_forbidden_ids':sorted(forbidden),
                         'fit':fit_evidence,'calibration_rows':len(y),'calibration_rows_by_recording':counts,
                         'calibration_normal_rows':int((y==0).sum()),'calibration_anomalous_rows':int((y==1).sum()),
                         'calibration_weight_sum':float(weights.sum()),
                         'calibration_weighted_positive_fraction':float(np.sum(weights*y)/weights.sum()),
                         'frozen_hgb_predictions_unchanged':True,
                         'calibration_weighting':'equal_record_without_class_balancing'})
    unique_counts={record['id']:len(causal_features(record)[1]) for record in [anchor]+records}
    return members,{'members':evidence,'training_ids':list(unique_counts),'training_rows_by_recording':unique_counts,
                    'training_rows':sum(unique_counts.values()),
                    'unique_calibration_rows':sum(unique_counts[record['id']] for record in records),
                    'counting_note':'Unique eligible observations, never the sum of overlapping ensemble-member fits.'}


def ensemble_scores(members: list[dict],record: dict) -> dict:
    if len(members)!=3:
        raise ValueError('The frozen ensemble contains exactly three classifier/calibrator pairs.')
    outputs=[classifier_scores(member,record) for member in members]
    return {'scores':np.mean(np.stack([output['scores'] for output in outputs]),axis=0),
            'contributions':np.mean(np.stack([output['contributions'] for output in outputs]),axis=0)}


def score_recording(bundle: dict,record: dict,base_scored: dict) -> dict:
    classifier=ensemble_scores(bundle['classifier_ensemble'],record)
    innovation=innovation_scores(bundle['innovation_bundle'],record)
    residual=aggregate_residuals(innovation['contributions'],bundle['aggregation'])
    learned=fuse_scores(classifier['scores'],residual,bundle['classifier_threshold'],bundle['residual_threshold'])
    output=dict(base_scored,scores=dict(base_scored['scores'],learned=learned),
                thresholds=dict(base_scored['thresholds'],learned=1.))
    output.update(contributions=classifier['contributions'],warmup_samples=2,learned_version='calibrated-7.0.0',
                  contributions_kind='reference_deviation_not_model_attribution',contributions_note=CONTRIBUTIONS_NOTE,
                  branch_scores={'classifier':classifier['scores'],'innovation':residual},
                  branch_thresholds={'classifier':bundle['classifier_threshold'],'innovation':bundle['residual_threshold']},
                  normalized_branch_scores={'classifier':classifier['scores']/bundle['classifier_threshold'],
                                            'innovation':residual/bundle['residual_threshold']})
    return output


def verify_development_sources(plan: dict,records: list[dict],anchor: dict,calibration: dict) -> None:
    if {record['id'] for record in records}!=set(plan['development_ids']):
        raise ValueError('Loaded development IDs differ from the frozen source set.')
    if any(record['sha256']!=plan['source_hashes'][record['id']] for record in records):
        raise ValueError('Loaded development source hash differs from the frozen protocol.')
    if any(record['sha256']!=plan['source_hashes']['anomaly-free'] for record in [anchor,calibration]):
        raise ValueError('Loaded normal source hash differs from the frozen protocol.')


def _verify_source(root: Path) -> dict:
    archive=root/'artifacts/v5'
    preservation=json.loads((archive/'preservation_manifest.json').read_text(encoding='utf-8'))
    for name in SOURCE_NAMES:
        if hashlib.sha256((archive/name).read_bytes()).hexdigest()!=preservation['files'][name]:
            raise ValueError('Preserved V5 development source hash mismatch.')
    report=json.loads((archive/'development/report.json').read_text(encoding='utf-8'))
    for name,key in [('residual_oof_channels.npz','channel_oof_sha256'),('selected_oof_scores.npz','oof_scores_sha256')]:
        if hashlib.sha256((archive/'development'/name).read_bytes()).hexdigest()!=report[key]:
            raise ValueError('Preserved V5 development array hash mismatch.')
    return report


def study_plan(root: Path) -> dict:
    root=Path(root);_verify_source(root)
    source=json.loads((root/'artifacts/v5/development/plan.json').read_text(encoding='utf-8'))
    paths=['artifacts/v5/preservation_manifest.json']+['artifacts/v5/'+name for name in SOURCE_NAMES]
    paths += ['artifacts/v1/'+name for name in ['preservation_manifest.json','model.joblib','metadata.json','split_manifest.json']]
    paths += ['signalwatch/'+name for name in ['ml_calibrated.py','ml_hybrid.py','ml_v2.py','ml_supervised.py',
                                              'ml_supervised_weighted.py','dataset.py','evaluation.py']]
    nested=[{'fold':fold['fold'],'outer_train_ids':fold['train_ids'],'outer_heldout_ids':fold['heldout_ids'],
             'inner_splits':inner_splits(fold['train_ids'])} for fold in source['folds']]
    return {'study':'nested-calibrated-v7','schema_version':1,'source_snapshot':SNAPSHOT,
            'development_ids':source['development_ids'],'source_hashes':source['source_hashes'],'folds':source['folds'],
            'nested_folds':nested,'final_inner_splits':inner_splits(source['development_ids']),
            'dependencies':{name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths},
            'classifier_config':CLASSIFIER_CONFIG,'innovation_config':INNOVATION_CONFIG,
            'classifier':'Three HGB+sigmoid pairs. Each fits preprocessing/HGB on innerfit+normal train anchor; FrozenEstimator calibrator fits ONLY innercal recording rows. Average three outputs; never full-data classifier refit after calibration.',
            'calibration_weights':'For calibration record r with n_r eligible rows, R records and total N rows: w_i=N/(R*n_r). Equal mass per recording, no class balancing. Require both pooled calibration classes.',
            'inner_split_rule':'Sort recording IDs lexicographically; innercal member j contains positions index%3==j, all other IDs fit. Anchor always fit-only. Outer-held IDs forbidden from both.',
            'predictor_OOF':'Reuse hash-verified V5 residual_oof_channels with identical outer folds; verify exact recording IDs, lengths,8channels and finite eligibility against causal classifier features.',
            'aggregations':AGGREGATIONS,'threshold_grid':{'normal_quantiles':QUANTILES,'additional':'nextafter(max normal,+inf)',
                                                        'filter':'Unique finite strictly positive thresholds; classifier and residual separately; at most49pairs per aggregation.'},
            'fusion':'max(mean calibrated classifier/classifier threshold, aggregated residual/residual threshold), require BOTH finite; alarm strictly >1. No temporal wait/smoothing/cooldown/backfill.',
            'constraints':{'minimum_detected_events':26,'false_onsets_per_normal_hour_max':2.,'normal_alert_fraction_max':.05,
                           'strictly_fewer_false_than_V5':True,'normal_alert_seconds_no_more_than_V5':True},
            'ranking':'Among qualifying NEW models: maximum detected events, fewer false onsets, lower normal-alert fraction, lower median/mean delay, aggregation order, classifier threshold order, residual threshold order.',
            'fallback':'V5 control is NOT ranked against qualifying models. Retain V5 ONLY when no new point qualifies. No old-test access or alternative final audit.',
            'final_fit':'Fit3deployment pairs using all12development inner splits plus fit-only normal anchor, and genuinely refit fixed normal predictor. Report unique eligible observations once,152distinct feature names, member fit/calibration counts separately.',
            'score_scope':'Repeated nested grouped OOF development selection. Calibrated known-state scores are not guaranteed field or physical-failure probabilities.',
            'source_reference':'V5 preserved OOF report/selectedscores are reference_methods; not raw_methods or confirmation semantics.',
            'old_test_access':False}


def _rank(point: dict) -> tuple:
    metric=point['metrics']
    return (-metric['detected_events'],metric['false_episodes'],metric['normal_alert_fraction'],
            metric['median_delay_seconds'] if metric['median_delay_seconds'] is not None else float('inf'),
            metric['mean_delay_seconds'] if metric['mean_delay_seconds'] is not None else float('inf'),
            point['index'],point['threshold_order'])


def select_candidate(points: list[dict]) -> dict | None:
    feasible=[point for point in points if point['feasible']]
    return min(feasible,key=_rank) if feasible else None


def _trace(base: dict,score: np.ndarray) -> dict:
    output=dict(base,scores=dict(base['scores'],learned=score),thresholds=dict(base['thresholds'],learned=1.))
    output.update(contributions=np.full((len(score),8),np.nan),contributions_kind='unavailable_in_saved_oof',
                  contributions_note='Saved nested OOF score; no fold-specific attribution is claimed.')
    return output


def _thresholds(normal: np.ndarray) -> np.ndarray:
    if not len(normal) or not np.isfinite(normal).all():
        raise ValueError('Threshold grid requires finite normal OOF observations.')
    values=np.unique(np.r_[np.quantile(normal,QUANTILES),np.nextafter(normal.max(),np.inf)])
    return values[np.isfinite(values)&(values>0)]


def _selected_traces(records: list[dict],class_scores: dict,channels: dict,baseline: dict,point: dict) -> dict:
    return {record['id']:_trace(baseline[record['id']],fuse_scores(class_scores[record['id']],
            aggregate_residuals(channels[record['id']],point['aggregation']),
            point['classifier_threshold'],point['residual_threshold'])) for record in records}


def _reasons(metric: dict,reference: dict,budget: float) -> list[str]:
    reasons=[]
    if metric['detected_events']<26:reasons.append('development_recall_floor')
    if metric['false_episodes_per_hour'] is None or metric['false_episodes_per_hour']>budget:reasons.append('false_onset_budget')
    if metric['normal_alert_fraction'] is None or metric['normal_alert_fraction']>.05:reasons.append('normal_alert_time_budget')
    if metric['false_episodes']>=reference['false_episodes']:reasons.append('no_strict_false_count_improvement')
    if metric['false_alert_seconds']>reference['false_alert_seconds']:reasons.append('reference_normal_alert_seconds')
    if metric['observed_normal_seconds']!=reference['observed_normal_seconds']:reasons.append('changed_normal_exposure')
    return reasons


def select_thresholds(records: list[dict],class_scores: dict,channels: dict,baseline: dict,reference: dict) -> tuple[dict | None,list[dict]]:
    class_normal=np.concatenate([class_scores[record['id']][~record['labels']&np.isfinite(class_scores[record['id']])] for record in records])
    class_grid=_thresholds(class_normal);audits=[]
    for index,kind in enumerate(AGGREGATIONS,1):
        residual={record['id']:aggregate_residuals(channels[record['id']],kind) for record in records}
        normal=np.concatenate([residual[record['id']][~record['labels']&np.isfinite(residual[record['id']])] for record in records])
        residual_grid=_thresholds(normal)
        for ci,ct in enumerate(class_grid):
            for ri,rt in enumerate(residual_grid):
                point={'index':index,'threshold':1.,'classifier_threshold':float(ct),'residual_threshold':float(rt),
                       'aggregation':kind,'classifier_threshold_index':ci,'residual_threshold_index':ri,
                       'threshold_order':ci*len(residual_grid)+ri}
                traces=_selected_traces(records,class_scores,channels,baseline,point)
                metric,reports=_point(records,traces,1.);reasons=_reasons(metric,reference,2.)
                point.update(metrics=metric,feasible=not reasons,rejection_reasons=reasons,methods=aggregate_reports(reports))
                audits.append(point)
    return select_candidate(audits),audits


def _event_map(reports: list[dict]) -> dict:
    return {event['id']:event for result in reports for event in result['events'] if event['status'] in ['detected','missed']}


def run_study(root: Path) -> dict:
    root=Path(root);target=root/'artifacts/calibrated-development';target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();plan=study_plan(root);path=target/'plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:raise ValueError('Frozen calibration plan/dependencies changed.')
    else:_write_json(path,plan)
    source=_verify_source(root);archive=root/'artifacts/v5/development'
    legacy,records,anchor,calibration=_load_development(root)
    verify_development_sources(plan,records,anchor,calibration)
    by_id={record['id']:record for record in records};baseline={key:score_recording_v1(legacy,record) for key,record in by_id.items()}
    with np.load(archive/'residual_oof_channels.npz',allow_pickle=False) as saved:
        channels={key:saved[key].copy() for key in saved.files}
    with np.load(archive/'selected_oof_scores.npz',allow_pickle=False) as saved:
        old_scores={key:saved[key].copy() for key in saved.files}
    if set(channels)!=set(by_id) or set(old_scores)!=set(by_id):raise ValueError('Source OOF recording IDs do not align.')
    for key,record in by_id.items():
        _,indices,_=causal_features(record);eligible=np.zeros(len(record['values']),bool);eligible[indices]=True
        if channels[key].shape!=(len(eligible),8) or old_scores[key].shape!=(len(eligible),):raise ValueError('Source OOF shape mismatch.')
        if not np.array_equal(np.isfinite(channels[key]).all(axis=1),eligible) or not np.array_equal(np.isfinite(old_scores[key]),eligible):
            raise ValueError('Source predictor/classifier eligibility does not align.')
    reference_traces={key:_trace(baseline[key],old_scores[key]) for key in by_id}
    reference,reference_reports=_point(records,reference_traces,1.)
    if reference!=source['metrics']:raise ValueError('V5 reference OOF metrics do not reproduce.')
    class_scores={};fold_evidence=[]
    for fold in plan['folds']:
        members,evidence=fit_ensemble([by_id[key] for key in fold['train_ids']],anchor,fold['heldout_ids'])
        evidence.update(fold=fold['fold'],outer_heldout_ids=fold['heldout_ids'])
        fold_evidence.append(evidence)
        for key in fold['heldout_ids']:
            class_scores[key]=ensemble_scores(members,by_id[key])['scores']
            if not np.array_equal(np.isfinite(class_scores[key]),np.isfinite(channels[key]).all(axis=1)):
                raise ValueError('Nested classifier/predictor eligibility mismatch.')
    np.savez_compressed(target/'classifier_oof_scores.npz',**class_scores)
    (target/'residual_oof_channels.npz').write_bytes((archive/'residual_oof_channels.npz').read_bytes())
    selected,audits=select_thresholds(records,class_scores,channels,baseline,reference)
    _write_json(target/'candidate_audits.json',audits)
    report={'schema_version':1,'scope':'nested_calibrated_grouped_oof_development_selection','selected':selected,
            'source_snapshot':SNAPSHOT,'old_test_scored':False,'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'nested_folds':fold_evidence,'reference_methods':aggregate_reports(reference_reports),'promote':selected is not None,
            'candidate_audits_sha256':hashlib.sha256((target/'candidate_audits.json').read_bytes()).hexdigest(),
            'classifier_oof_sha256':hashlib.sha256((target/'classifier_oof_scores.npz').read_bytes()).hexdigest(),
            'channel_oof_sha256':hashlib.sha256((target/'residual_oof_channels.npz').read_bytes()).hexdigest(),
            'source_bundle':{'archive':'artifacts/v5/development','source_model_sha256':source['model_sha256'],
                             'reference_oof_sha256':source['oof_scores_sha256'],'channel_oof_sha256':source['channel_oof_sha256'],
                             'training_fixtures':{'development_ids':plan['development_ids'],'source_hashes':plan['source_hashes']}},
            'limitations':['Repeated nested OOF selection is not independent validation or a blind test.',
                          'Sigmoid-calibrated known-state class scores are not guaranteed deployment or physical-failure probabilities.',
                          'Equal-record calibration changes the recording distribution; its weighted prior is reported explicitly.',
                          'OOF-to-final-ensemble threshold transfer and unseen-regime false-alert control remain unproven.',
                          'Sensor evidence is observational reference deviation, not model attribution or physical cause.']}
    if selected is None:
        report['status']='retain_v5'
    else:
        traces=_selected_traces(records,class_scores,channels,baseline,selected)
        metrics,reports=_point(records,traces,1.);old_events=_event_map(reference_reports);new_events=_event_map(reports)
        paired=[{'id':key,'reference_status':old['status'],'new_status':new_events[key]['status'],
                 'reference_delay_seconds':old['delay_seconds'],'new_delay_seconds':new_events[key]['delay_seconds']}
                for key,old in old_events.items()]
        report.update(metrics=metrics,methods=aggregate_reports(reports),paired_outcomes=paired,
                      lost_event_ids=[row['id'] for row in paired if row['reference_status']=='detected' and row['new_status']=='missed'],
                      gained_event_ids=[row['id'] for row in paired if row['reference_status']=='missed' and row['new_status']=='detected'],
                      group_slices={group:aggregate_reports([result for record,result in zip(records,reports) if record['group']==group])
                                    for group in sorted({record['group'] for record in records})},
                      fold_slices={str(fold['fold']):aggregate_reports([result for record,result in zip(records,reports) if record['id'] in fold['heldout_ids']]) for fold in plan['folds']},
                      per_recording=[{'id':record['id'],'group':record['group'],'metrics':result['metrics'],'events':result['events']}
                                     for record,result in zip(records,reports)])
        np.savez_compressed(target/'selected_oof_scores.npz',**{key:trace['scores']['learned'] for key,trace in traces.items()})
        report['oof_scores_sha256']=hashlib.sha256((target/'selected_oof_scores.npz').read_bytes()).hexdigest()
        _write_json(target/'selection_before_refit.json',report)
        members,class_evidence=fit_ensemble(records,anchor)
        innovation,innovation_evidence=fit_innovation([anchor]+records,calibration,INNOVATION_CONFIG)
        names=_hybrid_feature_names()
        candidate={'algorithm':'NestedSigmoidCalibratedHGBInnovationOR','aggregation':selected['aggregation'],
                   'classifier':CLASSIFIER_CONFIG,'innovation':INNOVATION_CONFIG,'ensemble_members':3,'calibration':'sigmoid'}
        bundle={'classifier_ensemble':members,'innovation_bundle':innovation,'classifier_threshold':selected['classifier_threshold'],
                'residual_threshold':selected['residual_threshold'],'aggregation':selected['aggregation'],
                'window':2,'threshold':1.,'feature_names':names,'candidate':candidate,'algorithm':candidate['algorithm'],
                'version':'calibrated-7.0.0','objective':'calibrated_known_state_and_normal_innovation_detection',
                'training_scope':'Nested recording-disjoint HGB/sigmoid ensemble plus fixed normal-only innovation predictor.',
                'score_note':'Maximum of relative alert scores; calibrated known-state ensemble probabilities do not guarantee field calibration.',
                'plan':plan,'training_evidence':{'classifier':class_evidence,'innovation':innovation_evidence},
                'source_bundle':report['source_bundle']}
        joblib.dump(bundle,target/'selected_bundle.joblib',compress=3)
        report['model_sha256']=hashlib.sha256((target/'selected_bundle.joblib').read_bytes()).hexdigest()
        report['final_fit']={'training_rows':class_evidence['training_rows'],'classifier_training_rows':class_evidence['training_rows'],
                             'innovation_training_rows':innovation_evidence['training_rows'],'feature_count':len(names),'feature_names':names,
                             'classifier':class_evidence,'innovation':innovation_evidence,
                             'ensemble_member_training_calibration':class_evidence['members'],
                             'unique_calibration_rows':class_evidence['unique_calibration_rows']}
        report['status']='selected_calibrated_development_candidate'
    report['study_seconds']=round(time.perf_counter()-started,3);_write_json(target/'report.json',report)
    return report


def load_development_bundle(root: Path) -> dict:
    target=Path(root)/'artifacts/calibrated-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    if not report['promote']:raise ValueError('Calibration study retained V5; no new model is promoted.')
    path=target/'selected_bundle.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['model_sha256']:raise ValueError('Calibrated model hash mismatch.')
    model=joblib.load(path);model['study_report']=report
    return model


def compare_development_budget(root: Path,bundle: dict,budget: float) -> dict:
    if not .25<=budget<=10.:raise ValueError('Budget must be between0.25 and10 false onsets per normal hour.')
    report=bundle['study_report'];path=Path(root)/'artifacts/calibrated-development/candidate_audits.json'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['candidate_audits_sha256']:raise ValueError('Saved calibrated candidate hash mismatch.')
    points=json.loads(path.read_text(encoding='utf-8'))
    for point in points:
        reasons=[reason for reason in point['rejection_reasons'] if reason!='false_onset_budget']
        rate=point['metrics']['false_episodes_per_hour']
        if rate is None or rate>budget:reasons.append('false_onset_budget')
        point.update(feasible=not reasons,rejection_reasons=reasons)
    selected=select_candidate(points)
    return {'scope':report['scope'],'budget':budget,'feasible':selected is not None,'selected':selected,
            'methods':selected['methods'] if selected else None,'reference_methods':report['reference_methods'],
            'calibration':{'learned':selected},'note':'Cached nested-development exploration only; active ensemble and thresholds remain unchanged.'}
