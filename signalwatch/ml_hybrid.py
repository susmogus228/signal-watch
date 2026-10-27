"""Frozen complementary-branch development study; no legacy test access."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np

from signalwatch.dataset import SENSORS, SNAPSHOT, manifest
from signalwatch.evaluation import aggregate_reports
from signalwatch.ml import score_recording_v1
from signalwatch.ml_supervised import CONTRIBUTIONS_NOTE, classifier_scores, feature_names
from signalwatch.ml_supervised_weighted import fit_model as fit_classifier
from signalwatch.ml_v2 import (
    _availability, _fit as fit_innovation, _load_development, _point, _rank,
    _write_json, development_plan, innovation_scores, select_candidate,
)

CLASSIFIER_THRESHOLD=0.9954052245087474
CLASSIFIER_CONFIG={'algorithm':'HistGradientBoostingClassifier','max_iter':150,'max_leaf_nodes':15,
                   'min_samples_leaf':20,'learning_rate':.05,'l2_regularization':1.,
                   'early_stopping':False,'weighting':'record_equal_class_balanced'}
INNOVATION_CONFIG={'window':2,'min_samples_leaf':12,'n_estimators':64}
AGGREGATIONS=['max','top2_mean','all8_mean']


def aggregate_residuals(channels: np.ndarray,kind: str) -> np.ndarray:
    values=np.asarray(channels,dtype=float)
    if values.ndim!=2 or values.shape[1]!=8:
        raise ValueError('Residual aggregation requires exactly eight channels.')
    if kind not in AGGREGATIONS:
        raise ValueError('Unknown frozen residual aggregation.')
    output=np.full(len(values),np.nan)
    eligible=np.isfinite(values).all(axis=1)
    if np.any(values[eligible]<0):
        raise ValueError('Normalized absolute residuals cannot be negative.')
    if eligible.any():
        current=values[eligible]
        if kind=='max':
            output[eligible]=current.max(axis=1)
        elif kind=='top2_mean':
            output[eligible]=np.partition(current,-2,axis=1)[:,-2:].mean(axis=1)
        else:
            output[eligible]=current.mean(axis=1)
    return output


def fuse_scores(classifier: np.ndarray,residual: np.ndarray | None,classifier_threshold: float,
                residual_threshold: float | None=None) -> np.ndarray:
    if not np.isfinite(classifier_threshold) or classifier_threshold<=0:
        raise ValueError('Classifier threshold must be finite and positive.')
    classifier=np.asarray(classifier,dtype=float)
    if residual is None:
        return classifier/classifier_threshold
    if residual_threshold is None or not np.isfinite(residual_threshold) or residual_threshold<=0:
        raise ValueError('Residual threshold must be finite and positive.')
    residual=np.asarray(residual,dtype=float)
    if residual.shape!=classifier.shape:
        raise ValueError('Fusion branch shapes differ.')
    output=np.full(len(classifier),np.nan)
    valid=np.isfinite(classifier)&np.isfinite(residual)
    output[valid]=np.maximum(classifier[valid]/classifier_threshold,residual[valid]/residual_threshold)
    return output


def select_hybrid_candidate(points: list[dict]) -> dict | None:
    return select_candidate(points)


def score_recording(bundle: dict,record: dict,base_scored: dict) -> dict:
    classifier=classifier_scores(bundle['classifier_bundle'],record)
    innovation=innovation_scores(bundle['innovation_bundle'],record)
    residual=aggregate_residuals(innovation['contributions'],bundle['aggregation'])
    learned=fuse_scores(classifier['scores'],residual,bundle['classifier_threshold'],bundle['residual_threshold'])
    output=dict(base_scored)
    output['scores']=dict(base_scored['scores'],learned=learned)
    output['thresholds']=dict(base_scored['thresholds'],learned=1.)
    output.update(contributions=classifier['contributions'],warmup_samples=2,learned_version='hybrid-5.0.0',
                  contributions_kind='reference_deviation_not_model_attribution',contributions_note=CONTRIBUTIONS_NOTE,
                  branch_scores={'classifier':classifier['scores'],'innovation':residual},
                  branch_thresholds={'classifier':bundle['classifier_threshold'],'innovation':bundle['residual_threshold']},
                  normalized_branch_scores={'classifier':classifier['scores']/bundle['classifier_threshold'],
                                            'innovation':residual/bundle['residual_threshold']})
    return output


def _dependencies(root: Path) -> dict:
    paths=[]
    for directory in ['artifacts/v4/development','artifacts/v2-development']:
        paths.extend(f'{directory}/{name}' for name in ['plan.json','report.json','selected_oof_scores.npz','selected_bundle.joblib'])
    paths.extend(f'artifacts/v1/{name}' for name in ['preservation_manifest.json','model.joblib','metadata.json','split_manifest.json'])
    paths.extend(['signalwatch/ml_hybrid.py','signalwatch/ml_v2.py','signalwatch/ml_supervised.py',
                  'signalwatch/ml_supervised_weighted.py','signalwatch/dataset.py','signalwatch/evaluation.py'])
    return {path:hashlib.sha256((root/path).read_bytes()).hexdigest() for path in paths}


def study_plan(root: Path) -> dict:
    root=Path(root)
    ids=json.loads((root/'artifacts/v1/split_manifest.json').read_text(encoding='utf-8'))['validation_ids']
    source_hashes={entry['id']:entry['sha256'] for entry in manifest(root)['recordings'] if entry['id'] in ids or entry['id']=='anomaly-free'}
    return {'study':'complementary-hybrid-v5','schema_version':1,'dataset_snapshot':SNAPSHOT,
            'development_ids':ids,'source_hashes':source_hashes,'folds':development_plan(ids,source_hashes)['folds'],
            'dependencies':_dependencies(root),'classifier_config':CLASSIFIER_CONFIG,'innovation_config':INNOVATION_CONFIG,
            'classifier_threshold':CLASSIFIER_THRESHOLD,'normal_train_anchor':[0,5172],
            'normal_residual_calibration_anchor':[5202,7524],
            'candidates':[{'index':0,'aggregation':None,'kind':'classifier_only_fallback'}]+
                         [{'index':i+1,'aggregation':kind,'kind':'classifier_OR_innovation'} for i,kind in enumerate(AGGREGATIONS)],
            'aggregation_rules':{'max':'Maximum of eight normalized absolute forecast residuals.',
                                 'top2_mean':'Arithmetic mean of the largest two channel residuals at the same observation.',
                                 'all8_mean':'Arithmetic mean of all eight channel residuals at the same observation.'},
            'channel_reconstruction':'Refit the three fixed innovation fold models and normalizers; score only each held-out recording; assert channel maxima match saved V2 OOF scores with atol1e-8, rtol0, equal_nan=True.',
            'classifier_OOF':'Reuse hash-verified saved V4 OOF classifier scores from matching recording-held-out folds. Do not apply the final all-development model to construct OOF scores.',
            'threshold_grid':{'normal_quantiles':np.linspace(.90,1.,101).tolist(),
                              'additional':'nextafter(max normal,+inf)','retain':'finite positive thresholds only'},
            'fusion':'max(classifier_score/classifier_threshold, aggregated_residual/residual_threshold)>1; require both finite. Classifier-only disables the residual branch.',
            'constraints':{'normal_alert_fraction_max':.05,'false_onsets_per_normal_hour_max':2.},
            'ranking':'Maximum onset_v2 recall, lower normal-alert fraction, lower false-onset rate, candidate order, threshold order. Classifier-only index0 wins exact ties.',
            'promotion':'Refit selected branches only when the selected hybrid point strictly outranks classifier-only. Otherwise retain V4 and do not audit another candidate.',
            'inference':'No labels, IDs, filenames, timestamps or elapsed/row position. Warm-up2, gaps reset both branch histories. No smoothing, hysteresis, cooldown, pulsing or point adjustment.',
            'normalization':'Keep existing fold-training-only feature scales and separate normal calibration residual scales. OOF-to-final-refit threshold transfer is not guaranteed.',
            'final_fit':'Refit the selected fixed classifier and innovation configurations on all development+normal train anchor. Unique training_rows are classifier-eligible18595; report predictor13891 rows separately, never sum overlapping rows.',
            'report_scope':'Repeated grouped OOF DEVELOPMENT selection after exploratory fixed-OR diagnosis; not independent validation or a blind test.',
            'contributions_note':CONTRIBUTIONS_NOTE,'old_test_access':False}


def _read_sources(root: Path) -> tuple[dict,dict,dict[str,np.ndarray],dict[str,np.ndarray]]:
    results=[];arrays=[]
    for directory in [root/'artifacts/v4/development',root/'artifacts/v2-development']:
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        for name,key in [('selected_oof_scores.npz','oof_scores_sha256'),('selected_bundle.joblib','model_sha256')]:
            if hashlib.sha256((directory/name).read_bytes()).hexdigest()!=report[key]:
                raise ValueError('Frozen hybrid source hash mismatch.')
        with np.load(directory/'selected_oof_scores.npz',allow_pickle=False) as stored:
            arrays.append({key:stored[key].copy() for key in stored.files})
        results.append(report)
    classifier,normal=results
    if classifier['selected']['candidate']!=CLASSIFIER_CONFIG or normal['selected']['candidate']!=INNOVATION_CONFIG:
        raise ValueError('Frozen branch configuration differs from V5 protocol.')
    if classifier['selected']['threshold']!=CLASSIFIER_THRESHOLD:
        raise ValueError('Classifier threshold changed from the frozen V4 value.')
    return classifier,normal,arrays[0],arrays[1]


def _trace(base: dict,learned: np.ndarray) -> dict:
    result=dict(base,scores=dict(base['scores'],learned=learned),thresholds=dict(base['thresholds'],learned=1.))
    result['contributions']=np.full((len(learned),8),np.nan)
    result['contributions_kind']='unavailable_in_saved_classifier_oof'
    result['contributions_note']='OOF traces preserve branch scores; no fold-specific classifier attribution is claimed.'
    return result


def _feasible(metric: dict,budget: float) -> bool:
    return metric['normal_alert_fraction'] is not None and metric['false_episodes_per_hour'] is not None and metric['normal_alert_fraction']<=.05 and metric['false_episodes_per_hour']<=budget


def select_thresholds(records: list[dict],channels: dict,class_scores: dict,baseline: dict,budget: float=2.) -> tuple[dict | None,list[dict],dict]:
    if not .25<=budget<=10.:
        raise ValueError('Budget must be between0.25 and10 false onsets per normal hour.')
    traces={record['id']:_trace(baseline[record['id']],fuse_scores(class_scores[record['id']],None,CLASSIFIER_THRESHOLD)) for record in records}
    metrics,_=_point(records,traces,1.)
    control={'index':0,'aggregation':None,'threshold_index':0,'threshold':1.,'residual_threshold':None,
             'classifier_threshold':CLASSIFIER_THRESHOLD,'metrics':metrics,'feasible':_feasible(metrics,budget)}
    audits=[control]
    for index,kind in enumerate(AGGREGATIONS,1):
        residual={record['id']:aggregate_residuals(channels[record['id']],kind) for record in records}
        normal=np.concatenate([residual[record['id']][~record['labels']&np.isfinite(residual[record['id']])] for record in records])
        thresholds=np.unique(np.r_[np.quantile(normal,np.linspace(.90,1.,101)),np.nextafter(normal.max(),np.inf)])
        thresholds=thresholds[np.isfinite(thresholds)&(thresholds>0)]
        for threshold_index,threshold in enumerate(thresholds):
            traces={record['id']:_trace(baseline[record['id']],fuse_scores(class_scores[record['id']],residual[record['id']],CLASSIFIER_THRESHOLD,float(threshold))) for record in records}
            metrics,_=_point(records,traces,1.)
            audits.append({'index':index,'aggregation':kind,'threshold_index':threshold_index,'threshold':1.,
                           'residual_threshold':float(threshold),'classifier_threshold':CLASSIFIER_THRESHOLD,
                           'metrics':metrics,'feasible':_feasible(metrics,budget)})
    return select_hybrid_candidate(audits),audits,control


def _selected_traces(records: list[dict],channels: dict,class_scores: dict,baseline: dict,selected: dict) -> dict:
    output={}
    for record in records:
        key=record['id'];kind=selected['aggregation']
        residual=aggregate_residuals(channels[key],kind) if kind is not None else None
        learned=fuse_scores(class_scores[key],residual,CLASSIFIER_THRESHOLD,selected['residual_threshold'])
        output[key]=_trace(baseline[key],learned)
    return output


def _hybrid_feature_names() -> list[str]:
    keys=[sensor['key'] for sensor in SENSORS]
    predictor=[f'innovation:centered_lag{lag}:{key}' for lag in [2,1] for key in keys]
    predictor.extend(f'innovation:past_std2:{key}' for key in keys)
    predictor.extend(f'innovation:endpoint_delta2:{key}' for key in keys)
    return ['classifier:'+name for name in feature_names()]+predictor


def run_study(root: Path) -> dict:
    root=Path(root);target=root/'artifacts/hybrid-development';target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();plan=study_plan(root);path=target/'plan.json'
    if path.exists():
        if json.loads(path.read_text(encoding='utf-8'))!=plan:
            raise ValueError('Frozen hybrid dependencies or protocol changed; use a separately versioned study.')
    else:
        _write_json(path,plan)
    classifier_source,normal_source,class_scores,saved_max=_read_sources(root)
    legacy,records,anchor,calibration=_load_development(root)
    by_id={record['id']:record for record in records}
    expected=set(by_id)
    if set(class_scores)!=expected or set(saved_max)!=expected:
        raise ValueError('Hybrid OOF sources do not contain the exact development recording set.')
    baseline={record['id']:score_recording_v1(legacy,record) for record in records}
    channels={};folds=[]
    for fold in plan['folds']:
        normal,evidence=fit_innovation([anchor]+[by_id[key] for key in fold['train_ids']],calibration,INNOVATION_CONFIG)
        evidence.update(fold=fold['fold'],heldout_ids=fold['heldout_ids'])
        if set(evidence['training_ids'])&set(fold['heldout_ids']):
            raise ValueError('Hybrid normal reconstruction leaked a held-out recording.')
        folds.append(evidence)
        for key in fold['heldout_ids']:
            predicted=innovation_scores(normal,by_id[key])
            np.testing.assert_allclose(predicted['scores'],saved_max[key],atol=1e-8,rtol=0,equal_nan=True)
            if len(class_scores[key])!=len(predicted['scores']) or not np.array_equal(np.isfinite(class_scores[key]),np.isfinite(predicted['scores'])):
                raise ValueError('Hybrid branch eligibility is inconsistent.')
            channels[key]=predicted['contributions']
    np.savez_compressed(target/'residual_oof_channels.npz',**channels)
    (target/'classifier_oof_scores.npz').write_bytes((root/'artifacts/v4/development/selected_oof_scores.npz').read_bytes())
    selected,audits,control=select_thresholds(records,channels,class_scores,baseline,2.)
    _write_json(target/'candidate_audits.json',audits)
    report={'schema_version':1,'scope':'hybrid_grouped_oof_development_selection','selected':selected,
            'classifier_control':control,'plan_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_snapshot':SNAPSHOT,'old_test_scored':False,'reconstructed_folds':folds,
            'source_models':{'classifier':classifier_source['model_sha256'],'innovation':normal_source['model_sha256']},
            'channel_oof_sha256':hashlib.sha256((target/'residual_oof_channels.npz').read_bytes()).hexdigest(),
            'classifier_oof_sha256':hashlib.sha256((target/'classifier_oof_scores.npz').read_bytes()).hexdigest(),
            'limitations':['This is repeated development selection after an exploratory fixed-OR diagnostic, not independent validation.',
                          'Known-state supervised training is retained; unknown-fault detection is not established.',
                          'OR cannot reduce the classifier pointwise false-alert time and group burdens may exceed the aggregate target.',
                          'Thresholds from OOF fold models may not transfer to final refitted models; calibration rates are not guarantees.',
                          'Relative alert scores and sensor deviations are not calibrated physical-failure probabilities or causal attributions.']}
    promoted=selected is not None and selected['index']!=0 and (not control['feasible'] or _rank(selected)<_rank(control))
    report['promote']=bool(promoted)
    if selected is not None:
        traces=_selected_traces(records,channels,class_scores,baseline,selected)
        metrics,reports=_point(records,traces,1.)
        availability=_availability(records,traces)
        availability['segments_with_any_hybrid_score']=availability.pop('segments_with_any_v2_score')
        report.update(metrics=metrics,methods=aggregate_reports(reports),availability=availability,
                      group_slices={group:aggregate_reports([result for record,result in zip(records,reports) if record['group']==group])
                                    for group in sorted({record['group'] for record in records})},
                      per_recording=[{'id':record['id'],'group':record['group'],'metrics':result['metrics'],'events':result['events']}
                                     for record,result in zip(records,reports)])
        np.savez_compressed(target/'selected_oof_scores.npz',**{key:trace['scores']['learned'] for key,trace in traces.items()})
        report['oof_scores_sha256']=hashlib.sha256((target/'selected_oof_scores.npz').read_bytes()).hexdigest()
    _write_json(target/'selection_before_refit.json',report)
    if not promoted:
        report['status']='retain_classifier_only' if selected is not None else 'no_feasible_candidate'
    else:
        classifier,class_evidence=fit_classifier([anchor]+records,CLASSIFIER_CONFIG)
        normal,normal_evidence=fit_innovation([anchor]+records,calibration,INNOVATION_CONFIG)
        names=_hybrid_feature_names()
        candidate={'algorithm':'ComplementaryClassifierInnovationOR','aggregation':selected['aggregation'],
                   'classifier':CLASSIFIER_CONFIG,'innovation':INNOVATION_CONFIG}
        bundle={'classifier_bundle':classifier,'innovation_bundle':normal,'aggregation':selected['aggregation'],
                'classifier_threshold':CLASSIFIER_THRESHOLD,'residual_threshold':selected['residual_threshold'],
                'window':2,'threshold':1.,'candidate':candidate,'feature_names':names,'plan':plan,
                'algorithm':candidate['algorithm'],'version':'hybrid-5.0.0',
                'objective':'complementary_known_state_and_normal_innovation_detection',
                'training_scope':'Known-state supervised classifier plus normal-only innovation predictor; grouped development fitting and selection.',
                'score_note':'Maximum of independently normalized branch alert scores; not a calibrated physical-failure probability.',
                'training_evidence':{'classifier':class_evidence,'innovation':normal_evidence}}
        joblib.dump(bundle,target/'selected_bundle.joblib',compress=3)
        report['model_sha256']=hashlib.sha256((target/'selected_bundle.joblib').read_bytes()).hexdigest()
        report['final_fit']={'training_rows':class_evidence['training_rows'],
                             'classifier_training_rows':class_evidence['training_rows'],
                             'innovation_training_rows':normal_evidence['training_rows'],'feature_count':len(names),
                             'classifier':class_evidence,'innovation':normal_evidence,
                             'note':'Unique eligible observations counted once; branch training rows overlap and must not be summed.'}
        report['status']='selected_hybrid_development_candidate'
    report['study_seconds']=round(time.perf_counter()-started,3)
    _write_json(target/'report.json',report)
    return report


def load_development_bundle(root: Path) -> dict:
    target=Path(root)/'artifacts/hybrid-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    if not report['promote']:
        raise ValueError('The development protocol retained the existing classifier; no new hybrid is promoted.')
    path=target/'selected_bundle.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['model_sha256']:
        raise ValueError('Hybrid model artifact hash mismatch.')
    model=joblib.load(path);model['study_report']=report
    return model


def compare_development_budget(root: Path,bundle: dict,budget: float) -> dict:
    target=Path(root)/'artifacts/hybrid-development'
    report=json.loads((target/'report.json').read_text(encoding='utf-8'))
    sources=[('residual_oof_channels.npz','channel_oof_sha256'),('classifier_oof_scores.npz','classifier_oof_sha256')]
    arrays=[]
    for name,key in sources:
        path=target/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=report[key]:
            raise ValueError('Saved hybrid branch OOF hash mismatch.')
        with np.load(path,allow_pickle=False) as saved:
            arrays.append({record:saved[record].copy() for record in saved.files})
    legacy,records,_,_=_load_development(Path(root))
    baseline={record['id']:score_recording_v1(legacy,record) for record in records}
    selected,_,control=select_thresholds(records,arrays[0],arrays[1],baseline,budget)
    if selected is None:
        return {'scope':report['scope'],'budget':budget,'feasible':False,'note':'No feasible frozen-family point; active artifacts unchanged.'}
    traces=_selected_traces(records,arrays[0],arrays[1],baseline,selected)
    _,reports=_point(records,traces,1.)
    return {'scope':report['scope'],'budget':budget,'feasible':True,'selected':selected,
            'thresholds':dict(traces[records[0]['id']]['thresholds']),
            'methods':aggregate_reports(reports),'calibration':{'learned':selected},
            'note':'Development-only exploration within the fixed aggregation family; classifier threshold remains fixed and no model or audit is changed.'}
