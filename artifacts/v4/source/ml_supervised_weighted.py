"""Predeclared sample-weight comparison on development recordings only."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from signalwatch.dataset import SNAPSHOT, manifest
from signalwatch.evaluation import aggregate_reports, intervals
from signalwatch.ml import score_recording_v1
from signalwatch.ml_supervised import (
    CONTRIBUTIONS_NOTE, causal_features, classifier_scores, feature_names,
    score_recording as supervised_score_recording, study_plan as supervised_plan,
)
from signalwatch.ml_v2 import (
    _availability, _load_development, _point, _write_json,
    select_candidate, threshold_selection,
)

SCHEMES=['record_equal_class_balanced','segment_equal_class_balanced']
BASE_MODELS=[{'algorithm':'HistGradientBoostingClassifier','max_iter':150,'max_leaf_nodes':15,
              'min_samples_leaf':20,'learning_rate':.05,'l2_regularization':1.,'early_stopping':False},
             {'algorithm':'ExtraTreesClassifier','n_estimators':128,'min_samples_leaf':4}]
CANDIDATES=[dict(model,weighting=scheme) for model in BASE_MODELS for scheme in SCHEMES]


def study_plan(ids: list[str],hashes: dict) -> dict:
    plan=supervised_plan(ids,hashes)
    plan.update(study='supervised-known-anomaly-weighted-v2',candidates=CANDIDATES,
                weighting_definitions={
                    'N':'Number of all eligible training rows across all nonempty training recordings.',
                    'R':'Number of training recordings having at least one eligible row.',
                    'K_r':'Number of label classes represented among eligible rows of recording r; either1 or2.',
                    'n_rc':'Eligible training rows of class c in recording r.',
                    'S_rc':'Nonempty eligible label segments of class c in recording r. Segments are contiguous raw labels split at every gap; segments without eligible rows are excluded.',
                    'n_rsc':'Eligible rows in label segment s, recording r, class c.',
                    'record_equal_class_balanced':'w_i=N/(R*K_r*n_rc). Every recording has total mass N/R; each represented class has mass N/(R*K_r).',
                    'segment_equal_class_balanced':'w_i=N/(R*K_r*S_rc*n_rsc). Each represented class has mass N/(R*K_r), shared equally among its nonempty segments.',
                    'normalization':'Construct positive raw weights summing to1 and multiply by N; numerical final normalization enforces sum(weights)=N.',
                    'preprocessing':'Unweighted fold-training median/IQR preprocessing is unchanged; only estimator.fit sample_weight changes. class_weight remains None.'},
                statistical_bootstrap={'performed':False,'note':'This is provenance metadata, not a statistical bootstrap or an independent confidence interval.'},
                report_scope='Weighted supervised grouped OOF DEVELOPMENT selection, not an unbiased or blind evaluation.',
                release_rule='Compare the selected weighted point with the original supervised point using max recall, lower normal-alert fraction, lower false-onset rate. Exact ties retain the original supervised release; legacy test results never select the release.')
    return plan


def sample_weights(records: list[dict],indices_by_record: list[np.ndarray],scheme: str) -> tuple[np.ndarray,dict]:
    if scheme not in SCHEMES:
        raise ValueError('Unknown frozen sample weighting scheme.')
    if len(records)!=len(indices_by_record):
        raise ValueError('Training record and eligibility lengths differ.')
    count=sum(len(indices) for indices in indices_by_record)
    nonempty=sum(bool(len(indices)) for indices in indices_by_record)
    if not count or not nonempty:
        raise ValueError('Sample weighting requires eligible training observations.')
    pieces=[];summaries={}
    for record,indices in zip(records,indices_by_record):
        if record.get('labels') is None:
            raise ValueError('Explicit training labels are required for weighting.')
        labels=np.asarray(record['labels'],dtype=int)
        if not np.isin(labels,[0,1]).all():
            raise ValueError('Training labels must be binary.')
        current=labels[indices]
        classes=np.unique(current)
        weights=np.zeros(len(indices),dtype=float)
        class_summary={}
        for label in classes:
            members=current==label
            segments=[]
            for a,b in intervals(labels==label,record['gap_before']):
                retained=(indices>=a)&(indices<=b)
                if retained.any():
                    segments.append(retained)
            if scheme=='record_equal_class_balanced':
                weights[members]=1./(nonempty*len(classes)*int(members.sum()))
            else:
                for retained in segments:
                    weights[retained]=1./(nonempty*len(classes)*len(segments)*int(retained.sum()))
            class_summary[str(int(label))]={'eligible_rows':int(members.sum()),'nonempty_segments':len(segments)}
        pieces.append(weights)
        summaries[record['id']]={'eligible_rows':len(indices),'represented_classes':len(classes),'classes':class_summary}
    combined=np.concatenate(pieces)
    if not np.isfinite(combined).all() or np.any(combined<=0):
        raise ValueError('Every eligible row must receive a finite positive weight.')
    combined*=count/combined.sum()
    offset=0
    for record,indices in zip(records,indices_by_record):
        part=combined[offset:offset+len(indices)];offset+=len(indices)
        summary=summaries[record['id']]
        summary['weight_sum']=float(part.sum())
        labels=np.asarray(record['labels'],int)[indices]
        for label,values in summary['classes'].items():
            values['weight_sum']=float(part[labels==int(label)].sum())
    return combined,{'scheme':scheme,'nonempty_record_count':nonempty,'eligible_rows':count,
                     'weight_sum':float(combined.sum()),'min_weight':float(combined.min()),
                     'max_weight':float(combined.max()),'per_recording':summaries}


def fit_model(records: list[dict],candidate: dict) -> tuple[dict,dict]:
    designs=[];targets=[];indices_by_record=[];counts={};sensor_values=[]
    for record in records:
        if record.get('labels') is None:
            raise ValueError('Weighted supervised training requires explicit development labels.')
        X,indices,_=causal_features(record)
        designs.append(X);targets.append(np.asarray(record['labels'],dtype=int)[indices])
        indices_by_record.append(indices);counts[record['id']]=len(indices)
        sensor_values.append(record['values'][indices])
    X=np.concatenate(designs);y=np.concatenate(targets)
    if len(np.unique(y))!=2:
        raise ValueError('Both known classes are required.')
    weights,weight_evidence=sample_weights(records,indices_by_record,candidate['weighting'])
    center=np.median(X,axis=0)
    scale=np.quantile(X,.75,axis=0)-np.quantile(X,.25,axis=0)
    scale=np.maximum(scale,np.maximum(np.std(X,axis=0)*.05,1e-6))
    raw=np.concatenate(sensor_values)
    sensor_scale=np.maximum(np.quantile(raw,.75,axis=0)-np.quantile(raw,.25,axis=0),
                            np.maximum(np.std(raw,axis=0)*.05,1e-6))
    parameters={key:value for key,value in candidate.items() if key not in ['algorithm','weighting']}
    if candidate['algorithm']=='ExtraTreesClassifier':
        estimator=ExtraTreesClassifier(**parameters,random_state=20261023,n_jobs=2,class_weight=None)
    elif candidate['algorithm']=='HistGradientBoostingClassifier':
        estimator=HistGradientBoostingClassifier(**parameters,random_state=20261023)
    else:
        raise ValueError('Algorithm is outside the frozen weighted study.')
    with threadpool_limits(limits=2):
        estimator.fit((X-center)/scale,y,sample_weight=weights)
    bundle={'model':estimator,'feature_center':center,'feature_scale':scale,'sensor_scale':sensor_scale,
            'threshold':.5,'window':2,'version':'supervised-weighted-2.0.0','candidate':candidate,
            'algorithm':candidate['algorithm'],'feature_names':feature_names(),
            'objective':'supervised_known_anomaly_classification'}
    evidence={'training_ids':[record['id'] for record in records],'training_rows_by_recording':counts,
              'training_rows':len(y),'normal_rows':int(np.sum(y==0)),'anomalous_rows':int(np.sum(y==1)),
              'feature_count':120,'feature_center':center.tolist(),'feature_scale':scale.tolist(),
              'sensor_scale':sensor_scale.tolist(),'sample_weights':weight_evidence,'weight_sum':float(weights.sum())}
    return bundle,evidence


def score_recording(bundle: dict,record: dict,base_scored: dict) -> dict:
    output=supervised_score_recording(bundle,record,base_scored)
    output['learned_version']='supervised-weighted-2.0.0'
    return output


def run_study(root: Path) -> dict:
    root=Path(root);target=root/'artifacts/supervised-weighted-development';target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    ids=json.loads((root/'artifacts/v1/split_manifest.json').read_text(encoding='utf-8'))['validation_ids']
    hashes={entry['id']:entry['sha256'] for entry in manifest(root)['recordings'] if entry['id'] in ids or entry['id']=='anomaly-free'}
    plan=study_plan(ids,hashes);path=target/'plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:
            raise ValueError('Frozen weighted study plan differs; create a new versioned study.')
    else:
        _write_json(path,plan)
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
                raise ValueError('Held-out weighted recording entered fitting.')
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
    report={'schema_version':1,'scope':'supervised_weighted_grouped_oof_development_selection',
            'selected':selected,'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'old_test_scored':False,'source_snapshot':SNAPSHOT,'objective':'supervised_known_anomaly_classification',
            'legacy_baseline_model_sha256':legacy['metadata']['model_sha256'],
            'statistical_bootstrap':plan['statistical_bootstrap'],
            'limitations':['Known-state labels train and select the classifier; unknown-fault detection is not established.',
                          'This is another bounded development study. Selection results are not an unbiased or blind test.',
                          'Training weights do not alter event labels, annotation segments, raw metrics or false-alert constraints.',
                          'Group-level alert rates may exceed aggregate constraints and are reported explicitly.',
                          'Old test results were not read by this study and cannot determine which release is selected.']}
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
    target=Path(root)/'artifacts/supervised-weighted-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    path=target/'selected_bundle.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report.get('model_sha256'):
        raise ValueError('Weighted supervised model hash mismatch.')
    bundle=joblib.load(path);bundle['study_report']=report
    return bundle


def compare_development_budget(root: Path,bundle: dict,budget: float) -> dict:
    target=Path(root)/'artifacts/supervised-weighted-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    path=target/'selected_oof_scores.npz'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['oof_scores_sha256']:
        raise ValueError('Weighted supervised OOF score hash mismatch.')
    legacy,records,_,_=_load_development(Path(root))
    traces={}
    with np.load(path,allow_pickle=False) as stored:
        for record in records:
            trace=score_recording_v1(legacy,record)
            trace['scores']['learned']=stored[record['id']]
            traces[record['id']]=trace
    selected,_=threshold_selection(records,traces,budget,report['selected']['index'])
    if selected is None:
        return {'scope':report['scope'],'budget':budget,'feasible':False,'note':'No feasible threshold in the frozen weighted development grid.'}
    _,reports=_point(records,traces,selected['threshold'])
    return {'scope':report['scope'],'budget':budget,'feasible':True,
            'thresholds':dict(traces[records[0]['id']]['thresholds'],learned=selected['threshold']),
            'methods':aggregate_reports(reports),'calibration':{'learned':selected},
            'note':'Weighted supervised OOF development threshold exploration. Active artifacts and legacy test remain unchanged.'}
