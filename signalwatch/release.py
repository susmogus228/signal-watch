"""Versioned local publication of a development-selected, retrospectively audited model."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn


def study_module(version: int):
    names={3:'signalwatch.ml_supervised',4:'signalwatch.ml_supervised_weighted',5:'signalwatch.ml_hybrid',
           6:'signalwatch.ml_confirmed',7:'signalwatch.ml_calibrated'}
    if version not in names:
        raise ValueError('Unsupported supervised study version.')
    return importlib.import_module(names[version])


def _read(path: Path):
    return json.loads(path.read_text(encoding='utf-8'))


def _hash(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_confirmed_acceptance(report: dict):
    """Do not activate a quieter policy by sacrificing the agreed recall/exposure."""
    learned=report['methods']['learned'];raw=report['raw_methods']['learned']
    if not (learned['total_events']==raw['total_events']==28
            and learned['detected_events']>=25
            and learned['false_episodes']<raw['false_episodes']
            and learned['false_alert_seconds']<=raw['false_alert_seconds']
            and learned['observed_normal_seconds']==raw['observed_normal_seconds']):
        raise ValueError('Confirmed alarm acceptance failed; retain the previous active release.')


def require_calibrated_acceptance(report: dict):
    """Apply the same measured goal to the previous-model reference."""
    try:
        require_confirmed_acceptance({'methods':report['methods'],'raw_methods':report['reference_methods']})
    except ValueError as error:
        raise ValueError('Calibrated model acceptance failed; retain the previous active release.') from error


def publish_release(root: Path, *, verified_reproduction: bool = False) -> dict:
    started=time.perf_counter();root=Path(root);target=root/'artifacts'
    manifest=_read(target/'release_manifest.json')
    version=manifest['study_version'];study=target/manifest['study_directory']
    development=_read(study/'report.json')
    if _hash(study/'plan.json')!=manifest['plan_sha256']:
        raise ValueError('Frozen release plan changed.')
    if development['selected']!=manifest['selected']:
        raise ValueError('Reproduced development selection differs from the frozen release.')
    child=study_module(version).load_development_bundle(root)
    current_child_sha=_hash(study/'selected_bundle.joblib')
    if current_child_sha!=manifest['audited_child_sha256'] and not verified_reproduction:
        raise ValueError('Selected model is not the model measured in the saved audit.')
    archive=target/'v1';preserved=_read(archive/'preservation_manifest.json')
    for name,digest in preserved['files'].items():
        if _hash(archive/name)!=digest:
            raise ValueError('Archived version1 evidence changed.')
    baseline=joblib.load(archive/'model.joblib')
    thresholds={**baseline['thresholds'],'learned':child['threshold']}
    bundle={'study_version':version,'baseline_bundle':baseline,'supervised_bundle':child,
            'window':2,'thresholds':thresholds,'split_manifest':baseline['split_manifest']}
    audit_path=target/manifest['audit_path']
    if _hash(audit_path)!=manifest['audit_sha256']:
        raise ValueError('Frozen retrospective audit changed.')
    report=copy.deepcopy(_read(audit_path))
    if version==6:
        require_confirmed_acceptance(report)
    if version==7:
        require_calibrated_acceptance(report)
        if len(child['classifier_ensemble'])!=3:
            raise ValueError('The frozen calibrated architecture requires exactly three classifier members.')
    if report['selected_development_model_sha256']!=manifest['audited_child_sha256']:
        raise ValueError('Audit provenance does not identify the frozen selected model.')
    original_child_path=manifest.get('audited_child_artifact')
    if original_child_path and _hash(target/original_child_path)!=manifest['audited_child_sha256']:
        raise ValueError('Preserved original audited model hash mismatch.')
    temporary=target/'model.joblib.tmp';joblib.dump(bundle,temporary,compress=3)
    model_sha=_hash(temporary)
    learned=report['methods']['learned'];target_state=report['target']
    verdict=(f"{learned['detected_events']}/{learned['total_events']} events detected; "
             f"25/28 target {'met' if target_state['recall_target_met'] else 'not met'}. "
             f"False onsets {learned['false_episodes_per_hour']:.2f}/normal hour "
             f"(development limit 2/h); normal time in alert {100*learned['normal_alert_fraction']:.2f}%. "
             'Retrospective single-testbed evidence; operational reliability is not established.')
    report.update(model_sha256=model_sha,operating_point_verdict=verdict,point_adjustment=False,
                  selection_scope=development['scope'],thresholds=thresholds,
                  development_selection={'methods':development['methods'],'scope':development['scope']},
                  original_v1_methods=_read(archive/'frozen_evaluation.json')['methods'])
    fit=development['final_fit']
    metadata={'name':'SignalWatch known-anomaly detector' if version<5 else 'SignalWatch complementary ML detector',
              'version':f'{version}.0.0','study_version':version,
              'algorithm':child['candidate']['algorithm'] if version<5 else 'HistGradientBoostingClassifier + ExtraTreesRegressor',
              'parameters':child['candidate'],
              'sklearn_version':sklearn.__version__,'window':2,'warmup_samples':2,
              'warmup_by_method':{'learned':2,'robust':30,'persistence':30},'feature_count':len(child['feature_names']),
              'features':child['feature_names'],'training_rows':fit['training_rows'],
              'training_scope':child.get('training_scope','Supervised normal and known anomalous development rows; entire recordings held out during selection.'),
              'score_note':child.get('score_note','Uncalibrated classifier score; not a physical-failure probability.'),
              'contributions_kind':'reference_deviation_not_model_attribution',
              'contributions_note':study_module(version).CONTRIBUTIONS_NOTE,
              'model_sha256':model_sha,'thresholds':thresholds,'dataset_snapshot':development['source_snapshot'],
              'selection_scope':development['scope'],'operating_point_verdict':verdict,
              'study_plan_sha256':manifest['plan_sha256'],'split_manifest':baseline['split_manifest'],
              'child_model_sha256':current_child_sha,'original_audited_child_sha256':manifest['audited_child_sha256'],
              'original_audit_sha256':manifest['audit_sha256'],
              'original_audited_child_artifact':original_child_path,
              'created_at':datetime.now(timezone.utc).isoformat(),'training_seconds':development['study_seconds'],
              'publication_seconds':round(time.perf_counter()-started,3),
              'limitations':development['limitations']+['Old test data was already examined; this is a retrospective audit.'],
              'reproduction_note':'The fixed selected study is repeated; the audit remains saved historical evidence, not a new test.'}
    if version>=5:
        metadata['development_audits_sha256']=_hash(study/'candidate_audits.json')
        metadata['branches']={
            'classifier':{'algorithm':'HistGradientBoostingClassifier','features':120,
                          'training_rows':fit['classifier_training_rows'],'threshold':child['classifier_threshold']},
            'innovation':{'algorithm':'ExtraTreesRegressor','features':32,
                          'training_rows':fit['innovation_training_rows'],'threshold':child['residual_threshold'],
                          'aggregation':child['aggregation']}}
        metadata['decision_rule']='Alert when either branch exceeds its threshold; fused score is the maximum branch ratio.'
    if version==6:
        metadata['name']='SignalWatch ML detector with confirmed alarms'
        metadata['alarm_policy']=child['alarm_policy']
        metadata['decision_rule']='Confirm only after consecutive observed raw-positive scores span the configured elapsed duration; no backfill.'
        metadata['raw_model_methods']=report['raw_methods']
        metadata['score_note']='Confirmed decision level: pending raw-positive scores are capped at 1; raw ML scores remain separately available. Neither score is a calibrated fault probability.'
        report['alarm_policy']=child['alarm_policy']
        report['decision_score_note']=metadata['score_note']
        report['development_selection']['raw_methods']=development['raw_methods']
    if version==7:
        metadata['name']='SignalWatch calibrated ML ensemble'
        metadata['algorithm']='3 sigmoid-calibrated HistGradientBoosting classifiers + ExtraTreesRegressor'
        metadata['classifier_member_count']=3
        metadata['base_estimator_count']=4
        metadata['calibrator_count']=3
        metadata['calibration_method']='sigmoid'
        metadata['calibration_scope']='Each member uses disjoint fitting and calibration recordings; nested grouped OOF selects the operating point.'
        metadata['branches']['classifier']['algorithm']='Mean of 3 sigmoid-calibrated HistGradientBoosting classifiers'
        metadata['decision_rule']='Alert when the calibrated classifier ensemble or normal predictor exceeds its development-selected threshold.'
        metadata['reference_model_methods']=report['reference_methods']
        metadata['calibration_provenance']=fit['classifier']
        report['development_selection']['reference_methods']=development['reference_methods']
    selection={'study_version':version,'scope':development['scope'],'selected':development['selected'],
               'release_choice_rule':manifest['choice_rule'],'plan_sha256':manifest['plan_sha256']}
    for name,data in [('metadata.json',metadata),('frozen_evaluation.json',report),('selection_manifest.json',selection)]:
        (target/(name+'.tmp')).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(temporary,target/'model.joblib')
    for name in ['metadata.json','frozen_evaluation.json','selection_manifest.json']:
        os.replace(target/(name+'.tmp'),target/name)
    return metadata


def reproduce_release(root: Path) -> dict:
    root=Path(root);manifest=_read(root/'artifacts/release_manifest.json')
    # Failed training must leave both active inference and its OOF exploration intact.
    stage=root/'work'/('reproduce-'+uuid.uuid4().hex)
    (stage/'artifacts').mkdir(parents=True)
    shutil.copytree(root/'data',stage/'data')
    shutil.copytree(root/'artifacts/v1',stage/'artifacts/v1')
    for name in ['split_manifest.json','release_manifest.json']:
        shutil.copy2(root/'artifacts'/name,stage/'artifacts'/name)
    for relative in manifest.get('study_dependencies',[]):
        source=root/'artifacts'/relative;destination=stage/'artifacts'/relative
        if source.is_dir():
            shutil.copytree(source,destination)
        else:
            destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
    staged_study=stage/'artifacts'/manifest['study_directory'];staged_study.mkdir(parents=True)
    plan_path=root/'artifacts'/manifest['study_directory']/'plan.json'
    shutil.copy2(plan_path,staged_study/'plan.json')
    for relative,digest in _read(plan_path).get('dependencies',{}).items():
        if not relative.startswith('signalwatch/'):
            continue
        source=root/relative
        if _hash(source)!=digest:
            raise ValueError('Scientific source differs from the frozen study; active release retained.')
        destination=stage/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    staged_audit=stage/'artifacts'/manifest['audit_path'];staged_audit.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(root/'artifacts'/manifest['audit_path'],staged_audit)
    study_module(manifest['study_version']).run_study(stage)
    # Re-scoring here verifies reproduction only; it never selects or retunes.
    from signalwatch.ml import evaluate_split
    child=study_module(manifest['study_version']).load_development_bundle(stage)
    baseline=joblib.load(stage/'artifacts/v1/model.joblib')
    wrapper={'study_version':manifest['study_version'],'supervised_bundle':child,'baseline_bundle':baseline}
    actual=evaluate_split(stage,wrapper,'test')
    expected=_read(staged_audit)
    if any(actual[key]!=expected[key] for key in ['methods','group_slices','per_recording']):
        raise ValueError('Reproduced inference differs from the saved retrospective audit; active release retained.')
    if manifest['study_version']==6:
        raw=evaluate_split(stage,dict(wrapper,study_version=5),'test')
        if any(raw[key]!=expected['raw_'+key] for key in ['methods','group_slices','per_recording']):
            raise ValueError('Raw model reproduction differs from the frozen V5 behavior; active release retained.')
    metadata=publish_release(stage,verified_reproduction=True)
    for source in staged_study.iterdir():
        destination=root/'artifacts'/manifest['study_directory']/source.name
        temporary=destination.with_suffix(destination.suffix+'.tmp')
        shutil.copy2(source,temporary);os.replace(temporary,destination)
    for name in ['model.joblib','metadata.json','frozen_evaluation.json','selection_manifest.json']:
        temporary=root/'artifacts'/(name+'.tmp')
        shutil.copy2(stage/'artifacts'/name,temporary);os.replace(temporary,root/'artifacts'/name)
    return metadata
