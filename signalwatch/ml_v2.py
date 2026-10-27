"""Versioned normal-only innovation study; never opens legacy test recordings."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import average_precision_score

from signalwatch.dataset import SNAPSHOT, load_builtin, manifest
from signalwatch.evaluation import aggregate_reports, durations, inspect_recording, intervals
from signalwatch.ml import _normal, _slice, calibration_record
from signalwatch.ml import score_recording_v1 as score_legacy

CANDIDATES=[{'window':window,'min_samples_leaf':leaf,'n_estimators':64}
            for window in [2,5,10] for leaf in [4,12]]


def _write_json(path: Path, data: dict | list):
    path.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')


def development_plan(ids: list[str], source_hashes: dict) -> dict:
    return {'study':'normal-only-innovation-v2','schema_version':1,'dataset_snapshot':SNAPSHOT,
            'development_ids':list(ids),'source_hashes':source_hashes,
            'folds':[{'fold':fold,'heldout_ids':[key for i,key in enumerate(ids) if i%3==fold],
                      'train_ids':[key for i,key in enumerate(ids) if i%3!=fold]} for fold in range(3)],
            'normal_train_anchor':[0,5172],'normal_scale_calibration_anchor':[5202,7524],
            'anchors_note':'Chronological blocks of the original normal recording. Calibration is reused development material, not an independent audit.',
            'normal_training_rule':'Target and every historical sample must have anomaly=0; no window may cross a gap. Held-out recordings never enter model/scaler fitting.',
            'target':'Present eight-channel observation minus preceding-window mean, divided by fold-training innovation scale.',
            'features':'Centered historical observations flattened, historical channel standard deviations and endpoint changes; no absolute levels, labels, timestamps or recording IDs.',
            'candidates':CANDIDATES,'random_state':20261023,'n_jobs':1,'smoothing_samples':1,
            'residual_scale':'Per-channel median absolute forecast residual on the separate normal calibration anchor, floored at0.03 standardized units. No in-sample residual normalization.',
            'threshold_grid':{'normal_quantiles':np.linspace(.90,1.,101).tolist(),'additional':'nextafter(max normal score,+inf)'},
            'constraints':{'normal_alert_fraction_max':.05,'false_onsets_per_normal_hour_max':2.},
            'selection':'Evaluate every predefined threshold. Maximize onset_v2 recall among feasible points, then minimize normal-alert fraction, false onsets/hour, candidate order and threshold order.',
            'no_feasible_behavior':'Save candidate audits and a no-feasible report; do not fit/promote a final model or inspect legacy test recordings.',
            'protocol':'onset_v2 unchanged; literal consecutive alert episodes; no cooldown, pulsing, point adjustment or gap history carryover.',
            'report_scope':'Grouped out-of-fold DEVELOPMENT selection, not unbiased validation or a new blind test.',
            'availability':'Report per-window event observability plus point metrics on common legacy30-sample eligibility; do not manufacture new onsets by masking episodes.',
            'legacy_test_access':False}


def _design(record: dict, window: int, scale: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    values=np.asarray(record['values'],dtype=float)
    if values.ndim!=2 or values.shape[1]!=8 or not np.isfinite(values).all():
        raise ValueError('Inference requires eight finite sensor values per row.')
    if len(values)<=window:
        return np.zeros((0,window*8+16)),np.zeros((0,8)),np.zeros(0,dtype=int)
    history=sliding_window_view(values,(window,8))[:,0][:-1]
    means=history.mean(axis=1)
    centered=(history-means[:,None,:])/scale
    X=np.concatenate([centered.reshape(len(history),-1),history.std(axis=1)/scale,
                      (history[:,-1]-history[:,0])/scale],axis=1)
    y=(values[window:]-means)/scale
    indices=np.arange(window,len(values))
    good=np.ones(len(indices),bool)
    for index in np.flatnonzero(record['gap_before']):
        good[max(0,index-window):min(len(good),index)]=False
    return X[good],y[good],indices[good]


def normal_training_arrays(record: dict,window: int,scale: np.ndarray) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    if record.get('labels') is None:
        raise ValueError('Trusted normal-window training requires explicit dataset annotations.')
    X,y,indices=_design(record,window,scale)
    labels=np.asarray(record['labels'],bool)
    anomalous=np.r_[0,np.cumsum(labels.astype(int))]
    valid=(anomalous[indices+1]-anomalous[indices-window])==0
    return X[valid],y[valid],indices[valid]


def innovation_scores(bundle: dict,record: dict) -> dict:
    n=len(record['values']);window=bundle['window']
    scores=np.full(n,np.nan);contributions=np.full((n,8),np.nan)
    X,y,indices=_design(record,window,bundle['scale'])
    if len(indices):
        residual=np.abs(y-bundle['model'].predict(X))/bundle['residual_scale']
        scores[indices]=residual.max(axis=1)
        contributions[indices]=residual
    return {'scores':scores,'contributions':contributions,'warmup_samples':window,
            'threshold':float(bundle.get('threshold',1.))}


def score_recording(bundle: dict,record: dict,base_scored: dict) -> dict:
    learned=innovation_scores(bundle,record)
    output=dict(base_scored)
    output['scores']=dict(base_scored['scores'],learned=learned['scores'])
    output['thresholds']=dict(base_scored['thresholds'],learned=learned['threshold'])
    output.update(contributions=learned['contributions'],warmup_samples=bundle['window'],learned_version='2.0.0')
    return output


def _rank(point: dict) -> tuple:
    metric=point['metrics']
    return (-(metric.get('event_recall') or 0.),metric.get('normal_alert_fraction') or 0.,
            metric.get('false_episodes_per_hour') or 0.,point.get('index',0),point.get('threshold_index',0))


def select_candidate(candidates: list[dict]) -> dict | None:
    eligible=[candidate for candidate in candidates if candidate['feasible']]
    return min(eligible,key=_rank) if eligible else None


def _fit(records: list[dict],calibration: dict,candidate: dict) -> tuple[dict,dict]:
    window=candidate['window']
    unscaled=[normal_training_arrays(record,window,np.ones(8)) for record in records]
    raw=np.concatenate([target for _,target,_ in unscaled])
    center=np.median(raw,axis=0)
    scale=1.4826*np.median(np.abs(raw-center),axis=0)
    scale=np.maximum(scale,np.maximum(np.std(raw,axis=0)*.05,1e-6))
    designs=[normal_training_arrays(record,window,scale) for record in records]
    X=np.concatenate([item[0] for item in designs]);y=np.concatenate([item[1] for item in designs])
    model=ExtraTreesRegressor(n_estimators=candidate['n_estimators'],min_samples_leaf=candidate['min_samples_leaf'],
                             random_state=20261023,n_jobs=1,max_features=1.)
    model.fit(X,y)
    cal_X,cal_y,cal_indices=_design(calibration,window,scale)
    calibration_errors=np.abs(cal_y-model.predict(cal_X))
    residual_scale=np.maximum(np.median(calibration_errors,axis=0),.03)
    bundle={'model':model,'window':window,'scale':scale,'residual_scale':residual_scale,'threshold':1.,
            'version':'2.0.0','algorithm':'ExtraTreesRegressor causal innovation','candidate':candidate}
    evidence={'training_ids':[record['id'] for record in records],
              'normal_training_windows':{record['id']:len(design[0]) for record,design in zip(records,designs)},
              'training_rows':len(X),'feature_count':X.shape[1],'calibration_rows':len(cal_indices),
              'scale':scale.tolist(),'residual_scale':residual_scale.tolist()}
    return bundle,evidence


def _point(records: list[dict],traces: dict[str,dict],threshold: float) -> tuple[dict,list[dict]]:
    reports=[]
    for record in records:
        trace=traces[record['id']]
        output=dict(trace,thresholds=dict(trace['thresholds'],learned=threshold))
        reports.append(inspect_recording(record,output))
    return aggregate_reports(reports)['learned'],reports


def threshold_selection(records: list[dict],traces: dict[str,dict],budget: float=2.,index: int=0) -> tuple[dict | None,list[dict]]:
    if not .25<=budget<=10.:
        raise ValueError('Budget must be between0.25 and10 episodes per normal hour.')
    normal=np.concatenate([traces[record['id']]['scores']['learned'][
        ~record['labels'] & np.isfinite(traces[record['id']]['scores']['learned'])] for record in records])
    if not len(normal):
        return None,[]
    thresholds=np.unique(np.r_[np.quantile(normal,np.linspace(.90,1.,101)),np.nextafter(normal.max(),np.inf)])
    audits=[]
    for threshold_index,threshold in enumerate(thresholds):
        metric,_=_point(records,traces,float(threshold))
        fraction=metric['normal_alert_fraction'];rate=metric['false_episodes_per_hour']
        feasible=fraction is not None and rate is not None and fraction<=.05 and rate<=budget
        audits.append({'index':index,'threshold_index':threshold_index,'threshold':float(threshold),
                       'feasible':bool(feasible),'metrics':metric})
    return select_candidate(audits),audits


def _availability(records: list[dict],traces: dict[str,dict]) -> dict:
    total=observable=legacy_observable=common_normal_seconds=common_false_seconds=0
    true_samples=false_samples=0
    for record in records:
        trace=traces[record['id']]
        finite=np.isfinite(trace['scores']['learned'])
        legacy=np.isfinite(trace['scores']['robust'])
        for a,b in intervals(record['labels'],record['gap_before']):
            total+=1;observable+=int(finite[a:b+1].any());legacy_observable+=int(legacy[a:b+1].any())
        common=finite&legacy
        alerts=trace['scores']['learned']>trace['thresholds']['learned']
        dt=durations(record)
        common_normal_seconds+=float(dt[common&~record['labels']].sum())
        common_false_seconds+=float(dt[common&~record['labels']&alerts].sum())
        true_samples+=int(np.sum(common&record['labels']&alerts))
        false_samples+=int(np.sum(common&~record['labels']&alerts))
    return {'annotated_segments':total,'segments_with_any_v2_score':observable,
            'segments_with_any_legacy30_score':legacy_observable,
            'common_legacy30_scored_normal_seconds':common_normal_seconds,
            'common_legacy30_normal_alert_fraction':common_false_seconds/common_normal_seconds if common_normal_seconds else None,
            'common_legacy30_point_precision':true_samples/(true_samples+false_samples) if true_samples+false_samples else None,
            'note':'Availability is separate from event correctness. Common eligibility is used only for point/time metrics, never to create artificial event onsets.'}


def _load_development(root: Path) -> tuple[dict,list[dict],dict,dict]:
    archive=Path(root)/'artifacts/v1'
    preservation=json.loads((archive/'preservation_manifest.json').read_text(encoding='utf-8'))
    for name in ['metadata.json','model.joblib','split_manifest.json']:
        if hashlib.sha256((archive/name).read_bytes()).hexdigest()!=preservation['files'][name]:
            raise ValueError('Archived version1 artifact hash mismatch.')
    metadata=json.loads((archive/'metadata.json').read_text(encoding='utf-8'))
    if metadata['model_sha256']!=preservation['model_sha256']:
        raise ValueError('Archived version1 metadata does not identify the preserved model.')
    legacy=joblib.load(archive/'model.joblib')
    legacy['metadata']=metadata
    split=legacy['split_manifest']
    normal=_normal(root)
    train_anchor=_slice(normal,*split['normal_blocks']['train'],'normal_train_anchor')
    calibration=calibration_record(root,split)
    records=[load_builtin(root,key) for key in split['validation_ids']]
    return legacy,records,train_anchor,calibration


def run_study(root: Path) -> dict:
    root=Path(root);target=root/'artifacts/v2-development';target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    # Freeze before loading/scoring development records; only manifest metadata is read here.
    old_split=json.loads((root/'artifacts/v1/split_manifest.json').read_text(encoding='utf-8'))
    ids=old_split['validation_ids']
    hashes={entry['id']:entry['sha256'] for entry in manifest(root)['recordings'] if entry['id'] in ids or entry['id']=='anomaly-free'}
    plan=development_plan(ids,hashes)
    plan_path=target/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding='utf-8'))!=plan:
            raise ValueError('The frozen development plan differs; create an explicitly versioned study.')
    else:
        _write_json(plan_path,plan)
    legacy,records,anchor,calibration=_load_development(root)
    by_id={record['id']:record for record in records}
    baseline={record['id']:score_legacy(legacy,record) for record in records}
    candidate_audits=[];candidate_traces={};finalists=[]
    for index,candidate in enumerate(CANDIDATES):
        traces={};fold_evidence=[]
        for fold in plan['folds']:
            training=[anchor]+[by_id[key] for key in fold['train_ids']]
            bundle,evidence=_fit(training,calibration,candidate)
            evidence.update(fold=fold['fold'],heldout_ids=fold['heldout_ids'])
            if set(evidence['training_ids'])&set(fold['heldout_ids']):
                raise ValueError('A held-out recording entered model fitting.')
            fold_evidence.append(evidence)
            for key in fold['heldout_ids']:
                traces[key]=score_recording(bundle,by_id[key],baseline[key])
        selected,threshold_audit=threshold_selection(records,traces,2.,index)
        audit={'index':index,'candidate':candidate,'folds':fold_evidence,'selected_point':selected,
               'threshold_audit':threshold_audit,'feasible':selected is not None}
        candidate_audits.append(audit)
        candidate_traces[index]=traces
        if selected is not None:
            finalists.append(dict(selected,candidate=candidate))
        print(json.dumps({'candidate':index,'parameters':candidate,'selected':selected}),flush=True)
    selected=select_candidate(finalists)
    _write_json(target/'candidate_audits.json',candidate_audits)
    report={'schema_version':1,'scope':'grouped_oof_development_selection','plan_sha256':hashlib.sha256(plan_path.read_bytes()).hexdigest(),
            'selected':selected,'old_test_scored':False,'source_snapshot':SNAPSHOT,
            'legacy_baseline_model_sha256':legacy['metadata']['model_sha256'],
            'limitations':['The same development OOF scores select thresholds and candidates; these are selection results, not unbiased test estimates.',
                          'The original calibration anchor is reused development material, not an independent final audit.',
                          'No production reliability, physical diagnosis or guaranteed25/28 outcome is claimed.']}
    if selected is None:
        report['status']='no_feasible_candidate'
    else:
        index=selected['index'];traces=candidate_traces[index]
        for trace in traces.values():
            trace['thresholds']['learned']=selected['threshold']
        selected_metrics,reports=_point(records,traces,selected['threshold'])
        report.update(status='selected_development_candidate',metrics=selected_metrics,
                      methods=aggregate_reports(reports),availability=_availability(records,traces),
                      group_slices={group:aggregate_reports([result for record,result in zip(records,reports) if record['group']==group])
                                    for group in sorted({record['group'] for record in records})},
                      per_recording=[{'id':record['id'],'group':record['group'],'metrics':result['metrics'],'events':result['events']}
                                     for record,result in zip(records,reports)])
        # Freeze selection report before fitting a single final development model.
        _write_json(target/'selection_before_refit.json',report)
        final_bundle,evidence=_fit([anchor]+records,calibration,selected['candidate'])
        final_bundle['threshold']=selected['threshold'];final_bundle['plan']=plan
        final_bundle['training_evidence']=evidence
        final_bundle['threshold_selection_scope']='grouped_oof_development_selection'
        joblib.dump(final_bundle,target/'selected_bundle.joblib',compress=3)
        report['model_sha256']=hashlib.sha256((target/'selected_bundle.joblib').read_bytes()).hexdigest()
        report['final_fit']=evidence
        arrays={key:traces[key]['scores']['learned'] for key in ids}
        np.savez_compressed(target/'selected_oof_scores.npz',**arrays)
        report['oof_scores_sha256']=hashlib.sha256((target/'selected_oof_scores.npz').read_bytes()).hexdigest()
    report['study_seconds']=round(time.perf_counter()-started,3)
    _write_json(target/'report.json',report)
    return report


def load_development_bundle(root: Path) -> dict:
    target=Path(root)/'artifacts/v2-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    path=target/'selected_bundle.joblib'
    if report.get('model_sha256')!=hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError('Version2 development model hash mismatch.')
    bundle=joblib.load(path)
    bundle['study_report']=report
    return bundle


def compare_development_budget(root: Path,bundle: dict,budget: float) -> dict:
    target=Path(root)/'artifacts/v2-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    path=target/'selected_oof_scores.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['oof_scores_sha256']:
        raise ValueError('Version2 development scores hash mismatch.')
    legacy,records,anchor,calibration=_load_development(Path(root))
    traces={}
    with np.load(path,allow_pickle=False) as stored:
        for record in records:
            trace=score_legacy(legacy,record)
            trace['scores']['learned']=stored[record['id']]
            traces[record['id']]=trace
    selected,audit=threshold_selection(records,traces,budget,report['selected']['index'])
    if selected is None:
        return {'scope':'grouped_oof_development_selection','budget':budget,'feasible':False,
                'note':'No feasible threshold in the predefined grid. Active artifacts are unchanged.'}
    for trace in traces.values():
        trace['thresholds']['learned']=selected['threshold']
    metrics,reports=_point(records,traces,selected['threshold'])
    return {'scope':'grouped_oof_development_selection','budget':budget,'feasible':True,
            'thresholds':dict(traces[records[0]['id']]['thresholds']),
            'methods':aggregate_reports(reports),'calibration':{'learned':selected},
            'note':'Exploratory OOF development selection, not an independent audit. Legacy test and active artifacts are unchanged.'}
