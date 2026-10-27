"""Explicit supervised known-anomaly study with held-out development recordings."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from signalwatch.dataset import SNAPSHOT, SENSORS, manifest
from signalwatch.evaluation import aggregate_reports
from signalwatch.ml import score_recording_v1
from signalwatch.ml_v2 import (
    _availability, _load_development, _point, _write_json,
    development_plan, select_candidate, threshold_selection,
)

WINDOWS=[2,5,15,30]
CANDIDATES=[{'algorithm':'ExtraTreesClassifier','n_estimators':128,'min_samples_leaf':leaf}
            for leaf in [4,12]]+[
    {'algorithm':'HistGradientBoostingClassifier','max_iter':150,'max_leaf_nodes':leaves,
     'min_samples_leaf':20,'learning_rate':.05,'l2_regularization':1.,'early_stopping':False}
    for leaves in [15,31]]
CONTRIBUTIONS_NOTE='Observed deviation from the initial two-sample reference; not model attribution or a physical cause.'


def feature_names() -> list[str]:
    keys=[sensor['key'] for sensor in SENSORS]
    output=[f'current:{key}' for key in keys]+[f'delta1:{key}' for key in keys]
    for window in WINDOWS:
        for kind in ['past_mean','past_std','current_minus_past_mean']:
            output.extend(f'{kind}{window}:{key}' for key in keys)
    return output+[f'current_minus_initial2:{key}' for key in keys]


def study_plan(ids: list[str],hashes: dict) -> dict:
    folds=development_plan(ids,hashes)['folds']
    return {'study':'supervised-known-anomaly-v1','schema_version':1,'dataset_snapshot':SNAPSHOT,
            'development_ids':ids,'source_hashes':hashes,'folds':folds,
            'normal_train_anchor':[0,5172],
            'target':'Known annotated abnormal state classification; this is not unknown-fault detection.',
            'training_rule':'Use all causally eligible normal and anomalous rows of fold TRAIN recordings plus the normal train anchor. Held-out recording labels enter threshold/candidate selection only.',
            'preprocessing':'Training-fold feature median and IQR, floored by0.05 times training feature standard deviation and1e-6; no held-out fit.',
            'feature_names':feature_names(),'windows':WINDOWS,'warmup_samples':2,
            'reference_rule':'Each recording/gap segment reference is the mean of its first two observed sensor rows. It remains fixed until the next gap; those rows are not assumed normal.',
            'rolling_rule':'At index t, each historical window uses min(window,t-segment_start) available PAST observations and excludes the present row. Mean and population standard deviation are causal.',
            'candidates':CANDIDATES,'random_state':20261023,'max_cpu_threads':2,
            'class_weight':None,'smoothing_samples':1,'score':'Uncalibrated predict_proba class1 score; not a physical-failure probability.',
            'threshold_grid':{'normal_quantiles':np.linspace(.90,1.,101).tolist(),'additional':'nextafter(max normal score,+inf)'},
            'constraints':{'normal_alert_fraction_max':.05,'false_onsets_per_normal_hour_max':2.},
            'selection':'Evaluate all thresholds. Maximize onset_v2 recall among feasible points, then lower normal alert fraction, false onsets/hour, declared candidate order and threshold order.',
            'no_feasible_behavior':'Save all candidate audits and a no-feasible report; do not fit/promote a selected model or inspect old test recordings.',
            'event_protocol':'onset_v2 unchanged, actual new alert episode must start inside an annotation; no point adjustment, smoothing, cooldown or periodic pulsing.',
            'report_scope':'Grouped OOF DEVELOPMENT threshold and candidate selection; not unbiased cross-validation or a blind test.',
            'availability':'Report common legacy30 eligibility point/time metrics and separately the number of observable annotated segments. Never manufacture onsets by masking alerts.',
            'contributions_note':CONTRIBUTIONS_NOTE,'legacy_test_access':False}


def causal_features(record: dict) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    values=np.asarray(record['values'],dtype=float)
    if values.ndim!=2 or values.shape[1]!=8 or not np.isfinite(values).all():
        raise ValueError('Classifier requires eight finite sensor values per row.')
    n=len(values)
    gaps=np.asarray(record['gap_before'],bool)
    starts=np.unique(np.r_[0,np.flatnonzero(gaps)]) if n else np.zeros(0,dtype=int)
    designs=[];all_indices=[];references=[]
    for start,end in zip(starts,np.r_[starts[1:],n]):
        segment=values[start:end]
        if len(segment)<=2:
            continue
        local=np.arange(2,len(segment))
        current=segment[local]
        fields=[current,current-segment[local-1]]
        cumulative=np.vstack([np.zeros((1,8)),np.cumsum(segment,axis=0)])
        squared=np.vstack([np.zeros((1,8)),np.cumsum(segment*segment,axis=0)])
        for window in WINDOWS:
            lower=np.maximum(0,local-window)
            counts=(local-lower)[:,None]
            mean=(cumulative[local]-cumulative[lower])/counts
            variance=(squared[local]-squared[lower])/counts-mean*mean
            fields.extend([mean,np.sqrt(np.maximum(variance,0.)),current-mean])
        initial=segment[:2].mean(axis=0)
        reference=current-initial
        fields.append(reference)
        designs.append(np.concatenate(fields,axis=1));all_indices.append(local+start);references.append(reference)
    if not designs:
        return np.zeros((0,120)),np.zeros(0,dtype=int),np.zeros((0,8))
    X=np.concatenate(designs)
    if not np.isfinite(X).all():
        raise ValueError('Sensor magnitude exceeds the supported numerical feature range.')
    return X,np.concatenate(all_indices),np.concatenate(references)


def classifier_scores(bundle: dict,record: dict) -> dict:
    n=len(record['values']);score=np.full(n,np.nan);contributions=np.full((n,8),np.nan)
    X,indices,reference=causal_features(record)
    if len(indices):
        transformed=(X-bundle['feature_center'])/bundle['feature_scale']
        with threadpool_limits(limits=2):
            probabilities=bundle['model'].predict_proba(transformed)
        positive=np.flatnonzero(bundle['model'].classes_==1)
        if len(positive)!=1:
            raise ValueError('Supervised artifact must contain both known classes.')
        score[indices]=np.clip(probabilities[:,positive[0]],0.,1.)
        contributions[indices]=np.abs(reference)/bundle['sensor_scale']
    return {'scores':score,'contributions':contributions,'warmup_samples':2,'threshold':float(bundle.get('threshold',.5))}


def score_recording(bundle: dict,record: dict,base_scored: dict) -> dict:
    learned=classifier_scores(bundle,record)
    output=dict(base_scored)
    output['scores']=dict(base_scored['scores'],learned=learned['scores'])
    output['thresholds']=dict(base_scored['thresholds'],learned=learned['threshold'])
    output.update(contributions=learned['contributions'],warmup_samples=2,
                  learned_version='supervised-1.0.0',contributions_kind='reference_deviation_not_model_attribution',
                  contributions_note=CONTRIBUTIONS_NOTE)
    return output


def fit_model(records: list[dict],candidate: dict) -> tuple[dict,dict]:
    designs=[];targets=[];counts={};sensor_values=[]
    for record in records:
        if record.get('labels') is None:
            raise ValueError('Supervised training requires explicit development labels.')
        X,indices,_=causal_features(record)
        labels=np.asarray(record['labels'],dtype=int)[indices]
        designs.append(X);targets.append(labels);counts[record['id']]=len(indices)
        sensor_values.append(record['values'][indices])
    X=np.concatenate(designs);y=np.concatenate(targets)
    if len(np.unique(y))!=2:
        raise ValueError('Both normal and anomalous development classes are required.')
    center=np.median(X,axis=0)
    scale=np.quantile(X,.75,axis=0)-np.quantile(X,.25,axis=0)
    scale=np.maximum(scale,np.maximum(np.std(X,axis=0)*.05,1e-6))
    raw=np.concatenate(sensor_values)
    sensor_scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),
                            np.maximum(np.std(raw,axis=0)*.05,1e-6))
    parameters={key:value for key,value in candidate.items() if key!='algorithm'}
    if candidate['algorithm']=='ExtraTreesClassifier':
        model=ExtraTreesClassifier(**parameters,random_state=20261023,n_jobs=2,class_weight=None)
    elif candidate['algorithm']=='HistGradientBoostingClassifier':
        model=HistGradientBoostingClassifier(**parameters,random_state=20261023)
    else:
        raise ValueError('Candidate algorithm is not in the frozen study.')
    with threadpool_limits(limits=2):
        model.fit((X-center)/scale,y)
    bundle={'model':model,'feature_center':center,'feature_scale':scale,'sensor_scale':sensor_scale,
            'threshold':.5,'window':2,'version':'supervised-1.0.0','candidate':candidate,
            'algorithm':candidate['algorithm'],'feature_names':feature_names(),
            'objective':'supervised_known_anomaly_classification'}
    evidence={'training_ids':[record['id'] for record in records],'training_rows_by_recording':counts,
              'training_rows':len(y),'normal_rows':int(np.sum(y==0)),'anomalous_rows':int(np.sum(y==1)),
              'feature_count':120,'feature_center':center.tolist(),'feature_scale':scale.tolist(),
              'sensor_scale':sensor_scale.tolist()}
    return bundle,evidence


def run_study(root: Path) -> dict:
    root=Path(root);target=root/'artifacts/supervised-development';target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    old_split=json.loads((root/'artifacts/v1/split_manifest.json').read_text(encoding='utf-8'))
    ids=old_split['validation_ids']
    hashes={entry['id']:entry['sha256'] for entry in manifest(root)['recordings'] if entry['id'] in ids or entry['id']=='anomaly-free'}
    plan=study_plan(ids,hashes);plan_path=target/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
            raise ValueError('Supervised development plan changed; a new versioned study is required.')
    else:
        _write_json(plan_path,plan)
    legacy,records,anchor,_=_load_development(root)
    by_id={record['id']:record for record in records}
    baseline={record['id']:score_recording_v1(legacy,record) for record in records}
    audits=[];all_traces={};finalists=[]
    for index,candidate in enumerate(CANDIDATES):
        traces={};folds=[]
        for fold in plan['folds']:
            model,evidence=fit_model([anchor]+[by_id[key] for key in fold['train_ids']],candidate)
            evidence.update(fold=fold['fold'],heldout_ids=fold['heldout_ids'])
            if set(evidence['training_ids'])&set(fold['heldout_ids']):
                raise ValueError('Held-out supervised recording entered fitting.')
            folds.append(evidence)
            for key in fold['heldout_ids']:
                traces[key]=score_recording(model,by_id[key],baseline[key])
        selected,threshold_audit=threshold_selection(records,traces,2.,index)
        audits.append({'index':index,'candidate':candidate,'folds':folds,'selected_point':selected,
                       'threshold_audit':threshold_audit,'feasible':selected is not None})
        all_traces[index]=traces
        if selected is not None:
            finalists.append(dict(selected,candidate=candidate))
        print(json.dumps({'candidate':index,'parameters':candidate,'selected':selected}),flush=True)
    selected=select_candidate(finalists)
    _write_json(target/'candidate_audits.json',audits)
    report={'schema_version':1,'scope':'supervised_grouped_oof_development_selection','selected':selected,
            'plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),'old_test_scored':False,
            'source_snapshot':SNAPSHOT,'objective':'supervised_known_anomaly_classification',
            'legacy_baseline_model_sha256':legacy['metadata']['model_sha256'],
            'limitations':['Known annotated abnormal states were used for training; unknown-fault detection is not established.',
                          'OOF development scores select thresholds and candidates; these are not unbiased test estimates.',
                          'The first two observations after a gap define a reference and are not assumed normal.',
                          'Classifier scores are uncalibrated; channel deviations are observations, not model attribution.',
                          'The25/28 old-test aspiration is not guaranteed and old test recordings were not accessed by this study.']}
    if selected is None:
        report['status']='no_feasible_candidate'
    else:
        traces=all_traces[selected['index']]
        for trace in traces.values():
            trace['thresholds']['learned']=selected['threshold']
        metrics,reports=_point(records,traces,selected['threshold'])
        availability=_availability(records,traces)
        availability['segments_with_any_classifier_score']=availability.pop('segments_with_any_v2_score')
        report.update(status='selected_development_candidate',metrics=metrics,methods=aggregate_reports(reports),
                      availability=availability,
                      group_slices={group:aggregate_reports([result for record,result in zip(records,reports) if record['group']==group])
                                    for group in sorted({record['group'] for record in records})},
                      per_recording=[{'id':record['id'],'group':record['group'],'metrics':result['metrics'],'events':result['events']}
                                     for record,result in zip(records,reports)])
        _write_json(target/'selection_before_refit.json',report)
        model,evidence=fit_model([anchor]+records,selected['candidate'])
        model['threshold']=selected['threshold'];model['plan']=plan;model['training_evidence']=evidence
        model['threshold_selection_scope']=report['scope']
        joblib.dump(model,target/'selected_bundle.joblib',compress=3)
        report['model_sha256']=hashlib.sha256((target/'selected_bundle.joblib').read_bytes()).hexdigest()
        report['final_fit']=evidence
        np.savez_compressed(target/'selected_oof_scores.npz',**{key:traces[key]['scores']['learned'] for key in ids})
        report['oof_scores_sha256']=hashlib.sha256((target/'selected_oof_scores.npz').read_bytes()).hexdigest()
    report['study_seconds']=round(time.perf_counter()-started,3)
    _write_json(target/'report.json',report)
    return report


def load_development_bundle(root: Path) -> dict:
    target=Path(root)/'artifacts/supervised-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    path=target/'selected_bundle.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report.get('model_sha256'):
        raise ValueError('Supervised development artifact hash mismatch.')
    model=joblib.load(path);model['study_report']=report
    return model


def compare_development_budget(root: Path,bundle: dict,budget: float) -> dict:
    target=Path(root)/'artifacts/supervised-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    path=target/'selected_oof_scores.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['oof_scores_sha256']:
        raise ValueError('Supervised OOF scores hash mismatch.')
    legacy,records,_,_=_load_development(Path(root))
    traces={}
    with np.load(path,allow_pickle=False) as stored:
        for record in records:
            trace=score_recording_v1(legacy,record)
            trace['scores']['learned']=stored[record['id']]
            traces[record['id']]=trace
    selected,_=threshold_selection(records,traces,budget,report['selected']['index'])
    if selected is None:
        return {'scope':report['scope'],'budget':budget,'feasible':False,'note':'No feasible threshold in the fixed development grid.'}
    _,reports=_point(records,traces,selected['threshold'])
    return {'scope':report['scope'],'budget':budget,'feasible':True,
            'thresholds':dict(traces[records[0]['id']]['thresholds'],learned=selected['threshold']),
            'methods':aggregate_reports(reports),'calibration':{'learned':selected},
            'note':'Supervised OOF development threshold exploration. No legacy test or active artifact is changed.'}
