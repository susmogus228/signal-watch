"""One frozen final-model regression and reused-transfer diagnostic."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

STUDIES={'artifacts/trend-development':'outputs/v10-diagnostics',
         'artifacts/trend-backbones-development':'outputs/v11-diagnostics',
         'artifacts/regime-development':'outputs/v12-diagnostics'}
FEATURE_COUNTS={'artifacts/trend-development':186,
                'artifacts/trend-backbones-development':186,
                'artifacts/regime-development':198}


def _target(root: Path,study_directory: str) -> Path:
    if study_directory not in STUDIES:raise ValueError('Unknown frozen trend study directory.')
    return Path(root)/STUDIES[study_directory]


def classifier_for_study(study_directory: str):
    if study_directory not in STUDIES:raise ValueError('Unknown frozen trend study directory.')
    if study_directory=='artifacts/regime-development':
        from signalwatch.ml_regimes import classifier_scores
    else:
        from signalwatch.ml_trends import classifier_scores
    return classifier_scores


def probability_trace(probabilities: np.ndarray,threshold: float) -> dict:
    values=np.asarray(probabilities,float)
    if values.ndim!=1 or not np.isfinite(threshold) or threshold<=0:
        raise ValueError('Direct classifier scores need a positive finite cutoff and one vector.')
    if np.isinf(values).any() or np.any((values[np.isfinite(values)]<0)|(values[np.isfinite(values)]>1)):
        raise ValueError('Classifier probabilities must be in [0,1] or unavailable NaN.')
    missing=np.full(len(values),np.nan)
    return {'scores':{'learned':values/threshold,'robust':missing.copy(),'persistence':missing.copy()},
            'thresholds':{'learned':1.,'robust':1.,'persistence':1.}}


def learned_events(report: dict) -> list[dict]:
    unmeasured={'baseline_overlap','contributing_sensors','contributions_kind','contributions_note'}
    return [{key:value for key,value in event.items() if key not in unmeasured} for event in report['events']]


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def verify_final_identity(root: Path,development: dict,study_directory: str='artifacts/trend-development') -> None:
    _target(root,study_directory)
    target=Path(root)/study_directory
    if _hash(target/'selected_bundle.joblib')!=development['model_sha256']:
        raise ValueError('Final model hash differs from the development report.')
    selection_path=target/'selection_before_final_fit.json'
    if _hash(selection_path)!=development['selection_before_final_fit_sha256']:
        raise ValueError('Pre-fit selection identity changed.')
    selection=_read(selection_path)
    if selection['selected']!=development['selected'] or selection['plan_sha256']!=development['plan_sha256']:
        raise ValueError('Final candidate does not match the pre-fit selection.')
    digest=_hash(target/'candidate_audits.json')
    if digest!=development['candidate_audits_sha256'] or digest!=selection['candidate_audits_sha256']:
        raise ValueError('Development candidate evidence changed.')
    for arm,digest in selection['oof_hashes'].items():
        if _hash(target/f'{arm}_oof_scores.npz')!=digest:
            raise ValueError('Frozen selected OOF evidence changed.')


def verify_bundle_identity(bundle: dict,development: dict,study_directory: str='artifacts/trend-development') -> None:
    if study_directory not in STUDIES:raise ValueError('Unknown frozen trend study directory.')
    feature_count=FEATURE_COUNTS[study_directory]
    selected=development['selected']
    if (bundle.get('arm')!=selected['arm'] or bundle.get('threshold')!=selected['threshold']
            or bundle.get('plan_sha256')!=development['plan_sha256']
            or bundle.get('selection_before_final_fit_sha256')!=development['selection_before_final_fit_sha256']
            or development.get('feature_count',feature_count)!=feature_count
            or len(bundle.get('feature_names',[]))!=feature_count):
        raise ValueError('Loaded classifier identity/cutoff differs from the frozen winner.')


def prepare_diagnostics(root: Path,study_directory: str='artifacts/trend-development') -> dict:
    root=Path(root);target=_target(root,study_directory);target.mkdir(parents=True,exist_ok=True)
    if (target/'report.json').exists():raise ValueError('Diagnostics already completed; refusing new inference.')
    development=_read(root/study_directory/'report.json')
    if development.get('selected') is None:
        raise ValueError('No feasible frozen development winner; final inference is not permitted.')
    if not development.get('model_sha256') or development.get('final_fit') is None:
        raise ValueError('No completed final fit; selected development point cannot authorize inference.')
    verify_final_identity(root,development,study_directory)
    train_plan=_read(root/study_directory/'plan.json')
    if development['plan_sha256']!=_hash(root/study_directory/'plan.json'):
        raise ValueError('Development report does not match the frozen plan.')
    if any(_hash(root/name)!=digest for name,digest in train_plan['dependencies'].items()):
        raise ValueError('Frozen model-development source changed.')
    archive=_read(root/'artifacts/v5/preservation_manifest.json')
    active=['model.joblib','metadata.json','frozen_evaluation.json','selection_manifest.json','split_manifest.json','release_manifest.json']
    for name in active:
        if _hash(root/'artifacts'/name)!=archive['files'][name]:raise ValueError('Active V5 changed before diagnostics.')
    paths=['signalwatch/trend_diagnostics.py','tests/test_trend_diagnostics.py','scripts/check_trend_model.py',
           'signalwatch/ml_trends.py','signalwatch/ml_supervised.py','signalwatch/ml_supervised_weighted.py',
           'signalwatch/ml_v2.py','signalwatch/dataset.py','signalwatch/evaluation.py','signalwatch/teaser_transfer.py',
           'docs/ML_V10_TREND_MODEL_PLAN.md','data/manifest.json','data/SKAB-DATA-README.md',
           'artifacts/v1/split_manifest.json',study_directory+'/plan.json',
           study_directory+'/report.json',study_directory+'/selected_bundle.joblib',
           study_directory+'/selection_before_final_fit.json',study_directory+'/candidate_audits.json',
           'artifacts/v5/frozen_evaluation.json','artifacts/v5/preservation_manifest.json',
           'outputs/teaser-transfer/report.json','outputs/teaser-transfer/traces.npz',
           'work/v8-independent-source/skab-teaser-v3.zip']+['artifacts/'+name for name in active]
    paths=sorted(set(paths)|set(train_plan['dependencies']))
    split=_read(root/'artifacts/v1/split_manifest.json')
    plan={'study':'sole-final-trend-classifier-reused-diagnostics','development_directory':study_directory,
          'dependencies':{name:_hash(root/name) for name in paths},'selected':development['selected'],
          'retrospective_ids':split['test_ids'],
          'inference':'One selected final classifier only, once per old-audit record and once on teaser. Immediate cutoff; no V9 policy.',
          'fresh_validation':False,'fit_or_tune':False,'activation':False,
          'reporting':'Both old28 and teaser already inspected; final regression and reused diagnostic, not a new independent test.'}
    path=target/'plan.json'
    if path.exists() and _read(path)!=plan:raise ValueError('Final diagnostic plan/source identity changed.')
    if not path.exists():path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    prefit=target/'prefit_manifest.json'
    if not prefit.exists():
        prefit.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                          'plan_sha256':_hash(path),'active_hashes':{name:_hash(root/'artifacts'/name) for name in active}},indent=2),encoding='utf-8')
    return plan


def run_diagnostics(root: Path,study_directory: str='artifacts/trend-development') -> dict:
    root=Path(root);target=_target(root,study_directory)
    if (target/'report.json').exists():raise ValueError('Diagnostics already completed; refusing new inference.')
    plan=prepare_diagnostics(root,study_directory)
    marker=target/'inference_started.json'
    if marker.exists():raise ValueError('Diagnostic inference already started; preserve and inspect the attempt.')
    marker.write_text(json.dumps({'started_at_utc':datetime.now(timezone.utc).isoformat(),
                      'plan_sha256':_hash(target/'plan.json')}),encoding='utf-8')
    import joblib
    from signalwatch.dataset import load_builtin
    from signalwatch.evaluation import inspect_recording,aggregate_reports
    classifier_scores=classifier_for_study(study_directory)
    from signalwatch.ml_supervised import causal_features as original_features
    from signalwatch.teaser_transfer import load_teaser_archive,transfer_metrics
    bundle=joblib.load(root/study_directory/'selected_bundle.joblib')
    verify_bundle_identity(bundle,_read(root/study_directory/'report.json'),study_directory)
    cutoff=float(bundle['threshold'])
    feature_count=FEATURE_COUNTS[study_directory]
    if len(bundle['feature_names'])!=feature_count:raise ValueError('Final classifier representation differs from the frozen model.')
    reports=[];records=[];scores_by_record={};grouped={}
    for key in plan['retrospective_ids']:
        record=load_builtin(root,key);probability=classifier_scores(bundle,record)['scores']
        finite=np.zeros(len(record['values']),bool);finite[original_features(record)[1]]=True
        if not np.array_equal(np.isfinite(probability),finite):raise ValueError('Classifier changed common score availability.')
        trace=probability_trace(probability,cutoff);report=inspect_recording(record,trace)
        reports.append(report);grouped.setdefault(record['group'],[]).append(report);scores_by_record[key]=trace['scores']['learned']
        records.append({'id':key,'group':record['group'],'source_sha256':record['sha256'],
                        'metrics':report['metrics']['learned'],'events':learned_events(report)})
    metrics=aggregate_reports(reports)['learned']
    reference=_read(root/'artifacts/v5/frozen_evaluation.json')['methods']['learned']
    if metrics['observed_normal_seconds']!=reference['observed_normal_seconds']:
        raise ValueError('Old-audit comparison changed normal exposure.')
    np.savez_compressed(target/'old_audit_scores.npz',**scores_by_record)
    record=load_teaser_archive(root/'work/v8-independent-source/skab-teaser-v3.zip')
    probability=classifier_scores(bundle,record)['scores'];normalized=probability_trace(probability,cutoff)['scores']['learned']
    with np.load(root/'outputs/teaser-transfer/traces.npz',allow_pickle=False) as source:
        if not np.array_equal(source['values'],record['values']) or not np.array_equal(source['timestamps'],np.asarray(record['timestamps'])):
            raise ValueError('Reused teaser pivot differs from the prior source.')
        control=source['v5_score'].copy()
    if not np.array_equal(np.isfinite(normalized),np.isfinite(control)):
        raise ValueError('Reused teaser score availability changed.')
    np.savez_compressed(target/'teaser_scores.npz',probability=probability,score=normalized)
    teaser=transfer_metrics(record,normalized)
    if any(_hash(root/name)!=digest for name,digest in plan['dependencies'].items()):
        raise ValueError('Diagnostic source dependencies changed during inference.')
    active=_read(target/'prefit_manifest.json')['active_hashes']
    if any(_hash(root/'artifacts'/name)!=digest for name,digest in active.items()):raise ValueError('Active release changed.')
    report={'scope':'Single final selected classifier; old-audit regression and reused teaser diagnostic',
            'plan_sha256':_hash(target/'plan.json'),'model_sha256':_hash(root/study_directory/'selected_bundle.joblib'),
            'threshold':cutoff,'feature_count':feature_count,'decision':'immediate classifier probability above threshold; no temporal policy',
            'retrospective':{'metrics':metrics,'reference_v5_metrics':reference,'per_recording':records,
                             'groups':{key:aggregate_reports(value)['learned'] for key,value in grouped.items()}},
            'reused_teaser':{'candidate':teaser,'reference_v5':_read(root/'outputs/teaser-transfer/report.json')['methods']['v5']},
            'old_25of28_target_met':metrics['detected_events']>=25,
            'old_alert_rate_target_met':metrics['false_episodes_per_hour'] is not None and metrics['false_episodes_per_hour']<=2.,
            'old_normal_alert_time_no_worse':metrics['false_alert_seconds']<=reference['false_alert_seconds'],
            'score_artifact_hashes':{p.name:_hash(p) for p in target.glob('*.npz')},
            'final_models_scored':1,'old_recording_count':len(records),'teaser_model_inference_passes':1,
            'fresh_validation':False,'model_fits':0,'activation':False,'active_v5_unchanged':True,
            'limits':['Both diagnostic corpora already inspected; no new independent validation.',
                      'Teaser has four source envelopes, uncertain transitions and assumed background.',
                      'Only learned detector metrics are measured; baseline/attribution placeholders are excluded.']}
    (target/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report
