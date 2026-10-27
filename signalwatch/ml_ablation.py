"""Development-only controlled comparison of a fixed nested HGB bank."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
import time
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss
from threadpoolctl import threadpool_limits

from signalwatch.dataset import SNAPSHOT
from signalwatch.evaluation import aggregate_reports,intervals
from signalwatch.ml import score_recording_v1
from signalwatch.ml_calibrated import fit_ensemble,inner_splits,verify_development_sources
from signalwatch.ml_hybrid import aggregate_residuals,fuse_scores
from signalwatch.ml_supervised import causal_features,classifier_scores
from signalwatch.ml_v2 import _load_development,_point,_write_json

ARMS=['A','B','C','D','E']
QUANTILES=[.95,.975,.99,.995,.999,1.]
FIXED_CLASSIFIER_THRESHOLD=.8580258155093349
RESIDUAL_THRESHOLD=16.174573325357347
E_REPRODUCTION_ATOL=1e-12
PROTOCOL='docs/ML_V8_ABLATION_PLAN.md'
SOURCE_NAMES=['development/plan.json','development/report.json','development/classifier_oof_scores.npz',
              'development/residual_oof_channels.npz','development/selected_oof_scores.npz']
ACTIVE_FILES=['model.joblib','metadata.json','frozen_evaluation.json','selection_manifest.json','release_manifest.json','split_manifest.json']


def _hash(path: Path) -> str:return hashlib.sha256(path.read_bytes()).hexdigest()
def _read(path: Path) -> dict:return json.loads(path.read_text(encoding='utf-8'))


def assert_development_ids(ids: list[str],allowed_ids: list[str]) -> None:
    if len(ids)!=len(set(ids)) or not set(ids)<=set(allowed_ids):
        raise ValueError('Only explicitly frozen development recording IDs may be scored.')


def member_bank_scores(members: list[dict],record: dict) -> tuple[np.ndarray,np.ndarray]:
    if len(members)!=3:raise ValueError('The controlled bank requires exactly three fixed members.')
    raw=[];calibrated=[]
    for member in members:
        fixed=member['model'].calibrated_classifiers_[0].estimator.estimator
        raw_member=dict(member,model=fixed)
        raw.append(classifier_scores(raw_member,record)['scores'])
        calibrated.append(classifier_scores(member,record)['scores'])
    raw=np.column_stack(raw);calibrated=np.column_stack(calibrated)
    if not np.array_equal(np.isfinite(raw),np.isfinite(calibrated)):
        raise ValueError('Paired raw/calibrated eligibility differs.')
    if not np.all(np.isfinite(raw)==np.isfinite(raw[:,[0]])):
        raise ValueError('Member eligibility differs.')
    return raw,calibrated


def derive_arms(control: np.ndarray,raw: np.ndarray,calibrated: np.ndarray) -> dict:
    control=np.asarray(control,float);raw=np.asarray(raw,float);calibrated=np.asarray(calibrated,float)
    if raw.shape!=calibrated.shape or raw.shape!=(len(control),3):
        raise ValueError('Controlled arm score shapes differ.')
    finite=np.isfinite(control)
    if not np.array_equal(np.isfinite(raw),np.repeat(finite[:,None],3,axis=1)) or not np.array_equal(np.isfinite(calibrated),np.isfinite(raw)):
        raise ValueError('Controlled arm finite eligibility differs.')
    for values in [control,raw,calibrated]:
        if np.any((values[np.isfinite(values)]<0)|(values[np.isfinite(values)]>1)):
            raise ValueError('Probability diagnostics require actual probabilities in[0,1].')
    return {'A':control.copy(),'B':raw[:,0].copy(),'C':calibrated[:,0].copy(),
            'D':raw.mean(axis=1),'E':calibrated.mean(axis=1)}


def _losses(y: np.ndarray,p: np.ndarray) -> dict:
    if not len(y):return {'rows':0,'positive_rows':0,'prevalence':None,'brier_loss':None,'log_loss':None,'average_precision':None}
    return {'rows':len(y),'positive_rows':int(y.sum()),'prevalence':float(y.mean()),
            'brier_loss':float(brier_score_loss(y,p,pos_label=1)),
            'log_loss':float(log_loss(y,np.column_stack([1-p,p]),labels=[0,1])),
            'average_precision':float(average_precision_score(y,p)) if len(np.unique(y))==2 else None}


def probability_diagnostics(records: list[dict],scores: dict) -> dict:
    per_record={};ys=[];ps=[]
    for record in records:
        values=np.asarray(scores[record['id']],float);labels=np.asarray(record['labels'],int)
        if values.shape!=labels.shape or not np.isin(labels,[0,1]).all():raise ValueError('Probability labels must be explicit binary[0,1].')
        finite=np.isfinite(values);p=values[finite];y=labels[finite]
        if np.any((p<0)|(p>1)):raise ValueError('Probability loss cannot use fused ratios or values outside[0,1].')
        per_record[record['id']]=_losses(y,p);ys.append(y);ps.append(p)
    pooled=_losses(np.concatenate(ys),np.concatenate(ps))
    macro={key:float(np.mean([row[key] for row in per_record.values() if row[key] is not None]))
           if any(row[key] is not None for row in per_record.values()) else None
           for key in ['brier_loss','log_loss','average_precision']}
    return {'pooled':pooled,'macro_record':macro,'per_recording':per_record,
            'scope':'Unweighted classifier-only probabilities; pooled rows and equal-record macro. AP macro excludes single-class records.',
            'log_loss_clipping':'Installed scikit-learn float64 machine epsilon, explicit labels[0,1].',
            'interpretation':'Brier/log loss are proper forecast losses, not measurements of calibration alone.'}


def normal_thresholds(values: np.ndarray) -> list[dict]:
    values=np.asarray(values,float)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Thresholds require finite normal OOF probabilities.')
    proposals=[(f'q{q:g}',float(np.quantile(values,q))) for q in QUANTILES]+[('above_max',float(np.nextafter(values.max(),np.inf)))]
    rows=[]
    for order,(label,threshold) in enumerate(proposals):
        if not np.isfinite(threshold) or threshold<=0:continue
        existing=next((row for row in rows if row['threshold']==threshold),None)
        if existing is not None:existing['quantile_labels'].append(label)
        else:rows.append({'threshold':threshold,'threshold_order':order,'quantile_labels':[label]})
    return rows


def _rank(point: dict) -> tuple:
    metric=point['metrics']
    return (-metric['detected_events'],metric['false_episodes'],metric['false_alert_seconds'],
            point['worst_family_false_episodes_per_hour'],
            metric['median_delay_seconds'] if metric['median_delay_seconds'] is not None else float('inf'),
            metric['mean_delay_seconds'] if metric['mean_delay_seconds'] is not None else float('inf'),
            point['arm_order'],point['threshold_order'])


def select_development_candidate(points: list[dict],include_control: bool=False) -> dict | None:
    eligible=[point for point in points if point['feasible'] and (include_control or point['arm']!='A')]
    return min(eligible,key=_rank) if eligible else None


def _trace(base: dict,score: np.ndarray) -> dict:
    output=dict(base,scores=dict(base['scores'],learned=score),thresholds=dict(base['thresholds'],learned=1.))
    output.update(contributions=np.full((len(score),8),np.nan),contributions_kind='unavailable_in_controlled_oof',
                  contributions_note='Controlled OOF scores; no model attribution is claimed.')
    return output


def _point_details(records: list[dict],scores: dict,residuals: dict,baseline: dict,arm: str,threshold: float) -> tuple[dict,dict]:
    traces={r['id']:_trace(baseline[r['id']],fuse_scores(scores[r['id']],residuals[r['id']],threshold,RESIDUAL_THRESHOLD)) for r in records}
    metric,reports=_point(records,traces,1.)
    classifier_traces={r['id']:_trace(baseline[r['id']],scores[r['id']]/threshold) for r in records}
    classifier_metric,classifier_reports=_point(records,classifier_traces,1.)
    groups={group:aggregate_reports([report for record,report in zip(records,reports) if record['group']==group])['learned'] for group in sorted({r['group'] for r in records})}
    finite_rates=[row['false_episodes_per_hour'] for row in groups.values() if row['false_episodes_per_hour'] is not None]
    worst=max(finite_rates) if finite_rates else float('inf')
    output={'arm':arm,'arm_order':ARMS.index(arm),'classifier_threshold':threshold,'residual_threshold':RESIDUAL_THRESHOLD,
            'threshold':1.,'metrics':metric,'group_slices':groups,'worst_family_false_episodes_per_hour':worst,
            'worst_family_normal_alert_fraction':max(row['normal_alert_fraction'] for row in groups.values() if row['normal_alert_fraction'] is not None),
            'classifier_only_metrics':classifier_metric,
            'per_recording':[{'id':r['id'],'group':r['group'],'metrics':rep['metrics']['learned'],'events':rep['events'],
                              'classifier_only_metrics':crep['metrics']['learned'],'classifier_only_events':crep['events']}
                             for r,rep,crep in zip(records,reports,classifier_reports)]}
    return output,{'fused':traces,'classifier':classifier_traces,'reports':reports,'classifier_reports':classifier_reports}


def _screen(metric: dict,reference: dict) -> list[str]:
    reasons=[]
    if metric['detected_events']<26:reasons.append('development_recall_floor')
    if metric['false_episodes_per_hour'] is None or metric['false_episodes_per_hour']>2.:reasons.append('false_onset_budget')
    if metric['normal_alert_fraction'] is None or metric['normal_alert_fraction']>.05:reasons.append('normal_alert_time_budget')
    if metric['false_episodes']>=reference['false_episodes']:reasons.append('no_strict_false_count_improvement')
    if metric['false_alert_seconds']>reference['false_alert_seconds']:reasons.append('reference_normal_alert_time')
    if metric['observed_normal_seconds']!=reference['observed_normal_seconds']:raise ValueError('Controlled comparison changed normal exposure.')
    return reasons


def _paired(records: list[dict],left: dict,right: dict) -> dict:
    output=[]
    for record in records:
        key=record['id'];gaps=record['gap_before'];labels=record['labels']
        row={'id':key}
        for score_kind in ['fused','classifier']:
            first=left[score_kind][key]['scores']['learned'];second=right[score_kind][key]['scores']['learned']
            a=np.isfinite(first)&(first>1);b=np.isfinite(second)&(second>1)
            onset_a={i for i,j in intervals(a,gaps)};onset_b={i for i,j in intervals(b,gaps)}
            row[score_kind]={'left_only_alert_samples':int((a&~b).sum()),'right_only_alert_samples':int((b&~a).sum()),
                             'left_only_normal_samples':int((a&~b&~labels).sum()),'right_only_normal_samples':int((b&~a&~labels).sum()),
                             'left_only_onsets':[float(record['elapsed'][i]) for i in sorted(onset_a-onset_b)],
                             'right_only_onsets':[float(record['elapsed'][i]) for i in sorted(onset_b-onset_a)]}
        output.append(row)
    return {'per_recording':output,
            'note':'Different outerfold sigmoid mappings, quantile interpolation and finite ties can change pooled-cutoff crossing sets.'}


def evaluate_arm_grid(records: list[dict],arms: dict,residuals: dict,baseline: dict,reference: dict) -> dict:
    candidates=[];fixed=[];by_label={};details={}
    for arm in ARMS:
        normal=np.concatenate([arms[arm][r['id']][~r['labels']&np.isfinite(arms[arm][r['id']])] for r in records])
        by_label[arm]={}
        for grid in normal_thresholds(normal):
            point,trace=_point_details(records,arms[arm],residuals,baseline,arm,grid['threshold'])
            point.update(grid);point['classifier_threshold']=point.pop('threshold');point['threshold']=1.
            reasons=_screen(point['metrics'],reference);point.update(feasible=not reasons,rejection_reasons=reasons)
            candidates.append(point)
            for label in grid['quantile_labels']:by_label[arm][label]=trace
            details[(arm,grid['threshold_order'])]=trace
        point,trace=_point_details(records,arms[arm],residuals,baseline,arm,FIXED_CLASSIFIER_THRESHOLD)
        point.update(comparison='fixed_probability_scale_sensitivity_not_equal_burden',threshold_order=-1)
        fixed.append(point);by_label[arm]['fixed_cutoff']=trace
    paired={f'{a}_vs_{b}':{label:_paired(records,by_label[a][label],by_label[b][label])
                           for label in by_label[a] if label in by_label[b]}
            for a,b in [('B','C'),('D','E')]}
    return {'candidate_audits':candidates,'fixed_cutoff_comparisons':fixed,'paired_alert_onset_comparisons':paired,
            'best_all':select_development_candidate(candidates,True),
            'best_control_A':select_development_candidate([point for point in candidates if point['arm']=='A'],True),
            'best_new':select_development_candidate(candidates)}


def _verify_sources(root: Path) -> dict:
    archive=root/'artifacts/v5';preservation=_read(archive/'preservation_manifest.json')
    for name in SOURCE_NAMES:
        if _hash(archive/name)!=preservation['files'][name]:raise ValueError('V5 development source hash mismatch.')
    report=_read(root/'artifacts/calibrated-development/report.json')
    for name,key in [('plan.json','plan_sha256'),('classifier_oof_scores.npz','classifier_oof_sha256')]:
        if _hash(root/'artifacts/calibrated-development'/name)!=report[key]:raise ValueError('V7 development source hash mismatch.')
    return report


def study_plan(root: Path) -> dict:
    root=Path(root);_verify_sources(root);v7=_read(root/'artifacts/calibrated-development/plan.json')
    source=_read(root/'artifacts/v5/development/plan.json')
    if source['development_ids']!=v7['development_ids'] or source['folds']!=v7['folds'] or source['source_hashes']!=v7['source_hashes']:
        raise ValueError('Controlled source folds or development hashes disagree.')
    paths=[PROTOCOL,'signalwatch/ml_ablation.py','tests/test_ml_ablation.py','scripts/study_ablation.py']
    paths+=['artifacts/v5/preservation_manifest.json']+['artifacts/v5/'+name for name in SOURCE_NAMES]
    paths+=['artifacts/calibrated-development/'+name for name in ['plan.json','report.json','classifier_oof_scores.npz']]
    paths+=['artifacts/v1/'+name for name in ['preservation_manifest.json','model.joblib','metadata.json','split_manifest.json']]
    paths+=['signalwatch/'+name for name in ['ml_calibrated.py','ml_hybrid.py','ml_v2.py','ml_supervised.py','ml_supervised_weighted.py','dataset.py','evaluation.py']]
    return {'study':'controlled-ablation-v8','schema_version':1,'source_snapshot':SNAPSHOT,
            'protocol_path':PROTOCOL,'protocol_sha256':_hash(root/PROTOCOL),
            'dependencies':{name:_hash(root/name) for name in paths},'development_ids':source['development_ids'],
            'source_hashes':source['source_hashes'],'folds':source['folds'],'nested_folds':v7['nested_folds'],
            'arms':{'A':'Saved V5 outer-full8-fit OOF, not final all-development in-sample predictions.',
                    'B':'Raw probability of fixed inner member0.','C':'Same member0 with its frozen sigmoid.',
                    'D':'Mean3 raw member probabilities.','E':'Mean3 sigmoid probabilities, reproduce saved V7 classifier OOF.'},
            'training':'Exactly9nested HGB/calibrator fits via unchanged V7 helper; outer-held excluded from fitting/preprocessing/calibration; no final/deployment fit.',
            'losses':'Classifier-only unweighted Brier/logloss/AP, pooled+equal-record macro+perrecord; explicit labels[0,1]; sklearn float64 epsilon clipping. Proper losses do not isolate calibration.',
            'sigmoid':'Record a,b orientation, raw/calibrated saturation<=1e-12 or>=1-1e-12, unique scores/ties and perrecordAP.',
            'E_reproduction_atol':E_REPRODUCTION_ATOL,'E_reproduction_rtol':0.,'fixed_cutoff':FIXED_CLASSIFIER_THRESHOLD,
            'normal_quantiles':QUANTILES,'additional_cutoff':'nextafter(max normal,+inf)','cutoff_filter':'Deduplicate finite positive values; retain all quantile labels for paired comparisons.',
            'residual_aggregation':'max','residual_threshold':RESIDUAL_THRESHOLD,'maximum_scan_points':35,'fixed_diagnostic_points':5,
            'screen':{'min_events':26,'max_false_per_normal_hour':2.,'max_normal_alert_fraction':.05,
                      'strictly_less_false_than_saved_V5':True,'normal_alert_seconds_no_more_than_saved_V5':True},
            'rank':'Maximumevents,fewerfalse,lowernormalalertseconds,lowerworstfamilyrate,median/mean delay,armA-E,thresholdorder.',
            'selection_outputs':['best_all','best_control_A','best_new'],'new_arms':['B','C','D','E'],
            'paired_comparisons':['B/C','D/E'],'old_test_access':False,'deployment_fit':False,'activation':False,
            'inference_guard':'Use only exact12developmentIDs, parsed actualsha256 matched frozen hashes, exactfold/sourcearray eligibility. No catalog/test loader, no audit files or final-model files read.',
            'interpretation':'Controlled development research only; separately tuned cutoffs confound action comparisons; repeated development cannot establish generalization.'}


def _array_read(path: Path) -> dict:
    with np.load(path,allow_pickle=False) as stored:return {key:stored[key].copy() for key in stored.files}


def _sigmoid_summary(members: list[dict],raw_scores: dict,cal_scores: dict) -> list[dict]:
    output=[]
    for index,member in enumerate(members):
        sigmoid=member['model'].calibrated_classifiers_[0].calibrators[0]
        raw=np.concatenate([values[:,index][np.isfinite(values[:,index])] for values in raw_scores.values()])
        cal=np.concatenate([values[:,index][np.isfinite(values[:,index])] for values in cal_scores.values()])
        stats=lambda values:{'rows':len(values),'unique_values':len(np.unique(values)),
                             'duplicate_score_rows':len(values)-len(np.unique(values)),
                             'near_zero':int((values<=1e-12).sum()),'near_one':int((values>=1-1e-12).sum())}
        output.append({'member':index,'a':float(sigmoid.a_),'b':float(sigmoid.b_),
                       'orientation':'increasing' if sigmoid.a_<0 else 'decreasing' if sigmoid.a_>0 else 'constant',
                       'raw':stats(raw),'sigmoid':stats(cal)})
    return output


def run_study(root: Path) -> dict:
    root=Path(root);target=root/'artifacts/ablation-development';target.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();plan=study_plan(root);plan_path=target/'plan.json'
    if plan_path.exists():
        if _read(plan_path)!=plan:raise ValueError('Frozen V8 protocol/dependencies changed; no fits permitted.')
    else:_write_json(plan_path,plan)
    if (target/'report.json').exists():raise ValueError('The one controlled development run already completed; refusing another fit.')
    active={name:_hash(root/'artifacts'/name) for name in ACTIVE_FILES}
    science={name:value for name,value in plan['dependencies'].items() if name.startswith('signalwatch/')}
    _write_json(target/'prefit_manifest.json',{'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                'plan_sha256':_hash(plan_path),'protocol_sha256':plan['protocol_sha256'],
                'active_hashes':active,'scientific_hashes':science,
                'environment':{'python':platform.python_version(),'sklearn':sklearn.__version__,'numpy':np.__version__},
                'authorization':'One development-only9fit experiment; no test inference or final deployment fit.'})
    legacy,records,anchor,calibration=_load_development(root)
    verify_development_sources(plan,records,anchor,calibration);assert_development_ids([r['id'] for r in records],plan['development_ids'])
    by_id={r['id']:r for r in records};baseline={key:score_recording_v1(legacy,r) for key,r in by_id.items()}
    archive=root/'artifacts/v5/development';control=_array_read(archive/'classifier_oof_scores.npz')
    channel=_array_read(archive/'residual_oof_channels.npz');v5fused=_array_read(archive/'selected_oof_scores.npz')
    expected_e=_array_read(root/'artifacts/calibrated-development/classifier_oof_scores.npz')
    if any(set(values)!=set(by_id) for values in [control,channel,v5fused,expected_e]):raise ValueError('Source OOF recording IDs mismatch.')
    residual={key:aggregate_residuals(values,'max') for key,values in channel.items()}
    for key,r in by_id.items():
        _,indices,_=causal_features(r);finite=np.zeros(len(r['values']),bool);finite[indices]=True
        if channel[key].shape!=(len(finite),8) or any(values[key].shape!=(len(finite),) for values in [control,v5fused,expected_e]):raise ValueError('Source OOF shape mismatch.')
        if any(not np.array_equal(np.isfinite(values[key]),finite) for values in [control,residual,v5fused,expected_e]):raise ValueError('Source OOF eligibility mismatch.')
    original,original_reports=_point(records,{key:_trace(baseline[key],v5fused[key]) for key in by_id},1.)
    if original!=_read(archive/'report.json')['metrics']:raise ValueError('Saved original V5 operating point does not reproduce.')
    banks_raw={};banks_cal={};evidence=[]
    for fold in plan['folds']:
        assert_development_ids(fold['train_ids']+fold['heldout_ids'],plan['development_ids'])
        if set(fold['train_ids'])&set(fold['heldout_ids']):raise ValueError('Outer training/heldout overlap.')
        members,fit_evidence=fit_ensemble([by_id[key] for key in fold['train_ids']],anchor,fold['heldout_ids'])
        fold_raw={};fold_cal={}
        for key in fold['heldout_ids']:
            raw,cal=member_bank_scores(members,by_id[key]);fold_raw[key]=raw;fold_cal[key]=cal
            np.testing.assert_allclose(cal.mean(axis=1),expected_e[key],atol=E_REPRODUCTION_ATOL,rtol=0,equal_nan=True)
            banks_raw[key]=raw;banks_cal[key]=cal
        evidence.append({'fold':fold['fold'],'outer_train_ids':fold['train_ids'],'outer_heldout_ids':fold['heldout_ids'],
                         'fit_calibration':fit_evidence,'sigmoids':_sigmoid_summary(members,fold_raw,fold_cal)})
    arms={arm:{} for arm in ARMS}
    for key in by_id:
        for arm,score in derive_arms(control[key],banks_raw[key],banks_cal[key]).items():arms[arm][key]=score
    np.savez_compressed(target/'raw_member_oof_scores.npz',**banks_raw)
    np.savez_compressed(target/'calibrated_member_oof_scores.npz',**banks_cal)
    for arm in ARMS:np.savez_compressed(target/f'arm_{arm}_oof_scores.npz',**arms[arm])
    evaluated=evaluate_arm_grid(records,arms,residual,baseline,original)
    _write_json(target/'candidate_audits.json',evaluated.pop('candidate_audits'))
    report={'schema_version':1,'scope':'development_only_controlled_component_ablation','status':'development_candidate_only' if evaluated['best_new'] else 'no_feasible_new_arm',
            'promote':False,'old_test_scored':False,'deployment_fit':False,'activation':False,'source_snapshot':SNAPSHOT,
            'plan_sha256':_hash(plan_path),'protocol_sha256':plan['protocol_sha256'],'original_v5_reference_methods':aggregate_reports(original_reports),
            'nested_fit_evidence':evidence,'actual_hgb_fits':sum(len(f['fit_calibration']['members']) for f in evidence),
            'E_classifier_oof_reproduces_v7':{'atol':E_REPRODUCTION_ATOL,'rtol':0.,'passed':True},
            'probability_diagnostics':{arm:probability_diagnostics(records,arms[arm]) for arm in ARMS},
            'candidate_audits_sha256':_hash(target/'candidate_audits.json'),
            'score_artifacts_sha256':{path.name:_hash(path) for path in target.glob('*.npz')},
            'limitations':['A/B differ in fit size/composition; A is not a new independent model.',
                          'B/C and D/E share fitted base estimators; nonlinear calibration and averaging need not commute.',
                          'Pooled outerfold mappings and quantiles differ; within-record AP and paired crossings are reported.',
                          'Per-arm cutoff tuning changes actions and prevents attributing tuned alarm differences to calibration alone.',
                          'Repeated development research cannot establish generalization or resolve unseen-regime false-alert control.',
                          'The shared residual branch can mask classifier changes; classifier-only evidence is included.'],**evaluated}
    if report['actual_hgb_fits']!=9:raise ValueError('Unexpected model fit count.')
    unchanged_active=active=={name:_hash(root/'artifacts'/name) for name in ACTIVE_FILES}
    unchanged_science=all(_hash(root/name)==digest for name,digest in science.items())
    if not unchanged_active or not unchanged_science:raise ValueError('Protected active/scientific files changed during study.')
    report['preservation']={'active_files_unchanged':True,'scientific_files_unchanged':True}
    report['study_seconds']=round(time.perf_counter()-started,3);_write_json(target/'report.json',report)
    return report
