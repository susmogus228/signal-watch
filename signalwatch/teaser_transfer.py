"""Strict SKAB teaser pivot and source-envelope transfer diagnostics."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import platform
import zipfile
from pathlib import Path
from datetime import datetime
from datetime import timezone

import numpy as np

from signalwatch.dataset import SENSORS,parse_csv_bytes
from signalwatch.evaluation import durations,intervals

ARCHIVE_SHA256='c444f45acc329999d09bdce230a7a58256e67e5d829dafed2bc4dad4645f8801'
SOURCE_EVENTS=[
    ('shaft-instant','18:39:22','18:39:22','18:42:32','18:42:32'),
    ('shaft-gradual','18:44:36','18:45:49','18:46:51','18:46:51'),
    ('valve-first','19:06:57','19:07:37','19:10:45','19:11:31'),
    ('valve-second','19:14:40','19:16:24','19:19:15','19:21:16'),
]


def parse_long_csv(raw: bytes) -> dict:
    """Reject incomplete observations; never resample or infer missing sensors."""
    keys=[sensor['key'] for sensor in SENSORS]
    try:
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')),delimiter=';')
        if reader.fieldnames is None or len(reader.fieldnames)!=4 or set(reader.fieldnames)!={'datetime','id','value','index'}:
            raise ValueError('Teaser CSV requires exact datetime/id/value/index columns.')
        grouped={};source_rows=0
        for row in reader:
            if None in row or any(row.get(key) is None for key in reader.fieldnames):
                raise ValueError('Teaser CSV has malformed rows.')
            stamp=datetime.fromisoformat(row['datetime'])
            if stamp.tzinfo is not None:
                raise ValueError('Teaser source-clock timestamps must be naive.')
            key=row['id'];value=float(row['value'])
            if key not in keys or not np.isfinite(value):
                raise ValueError('Teaser observations require exactly eight known finite sensors.')
            observation=grouped.setdefault(stamp,{})
            if key in observation:
                raise ValueError('Duplicate timestamp/sensor pair in teaser CSV.')
            observation[key]=value;source_rows+=1
        if not grouped or any(set(row)!=set(keys) for row in grouped.values()):
            raise ValueError('Teaser CSV has missing sensor observations.')
    except (UnicodeDecodeError,TypeError,OverflowError) as exc:
        raise ValueError('Invalid teaser CSV encoding or values.') from exc
    output=io.StringIO();writer=csv.writer(output,delimiter=';',lineterminator='\n')
    writer.writerow(['datetime']+keys)
    for stamp in sorted(grouped):
        writer.writerow([stamp.isoformat()]+[repr(grouped[stamp][key]) for key in keys])
    record=parse_csv_bytes(output.getvalue().encode(),'skab-teaser-v3','SKAB teaser version 3')
    record.update(group='teaser',split='external_transfer',source_rows=source_rows,
                  source_long_sha256=hashlib.sha256(raw).hexdigest())
    return record


def annotation_masks(record: dict) -> dict:
    dates=[datetime.fromisoformat(stamp) for stamp in record['timestamps']]
    envelope=np.zeros(len(dates),bool);core=envelope.copy();events=[]
    for key,start,complete,off,end in SOURCE_EVENTS:
        stamps=[datetime.fromisoformat('2019-07-08T'+clock) for clock in [start,complete,off,end]]
        a,b,c,d=stamps
        envelope|=np.array([a<=stamp<d for stamp in dates],bool)
        core|=np.array([b<=stamp<c for stamp in dates],bool)
        events.append(dict(zip(['id','appearance_start','appearance_complete','disappearance_start','disappearance_complete'],
                               [key]+[stamp.isoformat() for stamp in stamps])))
    return {'events':events,'envelope':envelope,'core':core,'uncertain':envelope&~core,'background':~envelope}


def transfer_metrics(record: dict,scores: np.ndarray) -> dict:
    """Count source interventions separately from gap-split benchmark segments."""
    values=np.asarray(scores,float)
    if values.shape!=(len(record['values']),):
        raise ValueError('Transfer scores must match teaser observations.')
    finite=np.isfinite(values);alerts=finite&(values>1.)
    segments=intervals(alerts,record['gap_before']);onsets=np.array([a for a,b in segments],int)
    dt=durations(record);masks=annotation_masks(record)
    dates=[datetime.fromisoformat(stamp) for stamp in record['timestamps']]
    events=[]
    for source in masks['events']:
        a=datetime.fromisoformat(source['appearance_start']);b=datetime.fromisoformat(source['appearance_complete'])
        end=datetime.fromisoformat(source['disappearance_complete'])
        inside=np.array([a<=stamp<end for stamp in dates],bool)
        available=np.flatnonzero(inside&finite)
        starts=[int(i) for i in onsets if inside[i]]
        first=starts[0] if starts else None
        first_row=np.flatnonzero(inside)
        preexisting=bool(len(first_row) and alerts[first_row[0]] and
                         any(start<first_row[0]<=stop for start,stop in segments))
        events.append(dict(source,status='unobservable' if not len(available) else 'detected' if first is not None else 'missed',
                           observed_rows=int(inside.sum()),eligible_rows=len(available),
                           preexisting_alert_at_first_observed_row=preexisting,
                           first_new_alert_timestamp=record['timestamps'][first] if first is not None else None,
                           delay_from_appearance_start_seconds=(dates[first]-a).total_seconds() if first is not None else None,
                           delay_from_appearance_complete_seconds=(dates[first]-b).total_seconds() if first is not None else None))
    background=masks['background'];background_seconds=float(dt[background&finite].sum())
    background_starts=[int(i) for i in onsets if background[i]]
    core_eligible=masks['core']&finite
    return {'source_interventions':4,'observable_interventions':sum(e['status']!='unobservable' for e in events),
            'detected_source_interventions':sum(e['status']=='detected' for e in events),'events':events,
            'assumed_background_seconds':background_seconds,'unannotated_background_alert_starts':len(background_starts),
            'unannotated_background_alert_starts_per_hour':len(background_starts)*3600/background_seconds if background_seconds else None,
            'assumed_background_alert_seconds':float(dt[background&alerts].sum()),
            'assumed_background_alert_fraction':float(dt[background&alerts].sum()/background_seconds) if background_seconds else None,
            'stable_anomaly_core_eligible_rows':int(core_eligible.sum()),
            'stable_anomaly_core_alert_rows':int((core_eligible&alerts).sum()),
            'stable_anomaly_core_point_recall':float((core_eligible&alerts).sum()/core_eligible.sum()) if core_eligible.any() else None,
            'uncertain_transition_rows':int(masks['uncertain'].sum()),'scored_rows':int(finite.sum()),
            'observed_seconds':float(dt[finite].sum()),
            'episodes':[{'start_index':a,'end_index':b,'start_timestamp':record['timestamps'][a],
                         'end_timestamp':record['timestamps'][b]} for a,b in segments],
            'background_scope':'Outside author-listed intervention envelopes; assumed background, not adjudicated physical normality.',
            'event_scope':'Four source envelopes, actual episode starts, not annotation intervals split at gaps.'}


def load_teaser_archive(path: Path,expected_sha256: str=ARCHIVE_SHA256) -> dict:
    raw=Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Teaser archive hash mismatch.')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        if len(archive.namelist())!=2 or set(archive.namelist())!={'SKAB teaser.csv','testbed.png'}:
            raise ValueError('Unexpected teaser archive inventory.')
        record=parse_long_csv(archive.read('SKAB teaser.csv'))
    record['source_archive_sha256']=expected_sha256
    return record


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_transfer(root: Path) -> dict:
    root=Path(root);target=root/'outputs/teaser-transfer';target.mkdir(parents=True,exist_ok=True)
    if (target/'report.json').exists():
        raise ValueError('Transfer already completed; refusing another model inference.')
    development=json.loads((root/'artifacts/branch-policy-development/report.json').read_text())
    if development['status'] not in ['development_candidate_only','no_feasible_candidate']:
        raise ValueError('Development selection is not frozen and complete.')
    preservation=json.loads((root/'artifacts/v5/preservation_manifest.json').read_text())
    active=['model.joblib','metadata.json','frozen_evaluation.json','selection_manifest.json','split_manifest.json','release_manifest.json']
    for name in active:
        if _hash(root/'artifacts'/name)!=preservation['files'][name] or _hash(root/'artifacts/v5'/name)!=preservation['files'][name]:
            raise ValueError('V5 active/archive identity changed before transfer.')
    source='work/v8-independent-source/skab-teaser-v3.zip'
    if _hash(root/source)!=ARCHIVE_SHA256:
        raise ValueError('Teaser archive hash mismatch.')
    paths=['docs/ML_V9_BRANCH_POLICY_PLAN.md','signalwatch/teaser_transfer.py','tests/test_teaser_transfer.py',
           'scripts/check_teaser_transfer.py','signalwatch/ml_branch_policy.py','signalwatch/alarm_policy.py',
           'signalwatch/dataset.py','signalwatch/evaluation.py','signalwatch/ml_supervised.py',
           'signalwatch/ml_v2.py','signalwatch/ml_hybrid.py','artifacts/v5/preservation_manifest.json',
           'artifacts/branch-policy-development/plan.json','artifacts/branch-policy-development/report.json',source]
    paths+=['artifacts/v5/'+name for name in active]
    plan={'study':'skab-teaser-transfer-fixed-v5-v9','source_archive_sha256':ARCHIVE_SHA256,
          'dependencies':{name:_hash(root/name) for name in paths},'source_events':annotation_masks({'timestamps':[]})['events'],
          'candidate_enabled':development['selected'] is not None,'development_status':development['status'],
          'inference':'One archived V5 classifier and innovation score pass; same raw branches for optional fixed V9 policy.',
          'old_28_access':False,'fit_or_tune':False,'activation':False,
          'source_truth':'Author intervention envelopes; gradual transitions uncertain; remaining background assumed, not physically adjudicated.'}
    path=target/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:
        raise ValueError('Frozen transfer dependencies or annotations changed.')
    if not path.exists():path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    prefit=target/'prefit_manifest.json'
    if not prefit.exists():
        prefit.write_text(json.dumps({'frozen_at_utc':datetime.now(timezone.utc).isoformat(),
                          'plan_sha256':_hash(path),'python':platform.python_version(),'numpy':np.__version__,
                          'active_hashes':{name:_hash(root/'artifacts'/name) for name in active}},indent=2),encoding='utf-8')
    return plan


def run_transfer(root: Path) -> dict:
    root=Path(root);target=root/'outputs/teaser-transfer'
    if (target/'report.json').exists():
        raise ValueError('Transfer already completed; refusing another model inference.')
    plan=prepare_transfer(root)
    marker=target/'inference_started.json'
    if marker.exists():raise ValueError('Transfer inference already started; inspect the preserved attempt.')
    marker.write_text(json.dumps({'started_at_utc':datetime.now(timezone.utc).isoformat(),
                      'plan_sha256':_hash(target/'plan.json')}),encoding='utf-8')
    import joblib
    import sklearn
    from threadpoolctl import threadpool_limits
    from signalwatch.ml_branch_policy import apply_to_scored
    from signalwatch.ml_supervised import classifier_scores
    from signalwatch.ml_v2 import innovation_scores
    from signalwatch.ml_hybrid import aggregate_residuals,fuse_scores
    record=load_teaser_archive(root/'work/v8-independent-source/skab-teaser-v3.zip')
    wrapper=joblib.load(root/'artifacts/v5/model.joblib');bundle=wrapper['supervised_bundle']
    with threadpool_limits(limits=2):
        classifier=classifier_scores(bundle['classifier_bundle'],record)
        innovation=innovation_scores(bundle['innovation_bundle'],record)
    residual=aggregate_residuals(innovation['contributions'],bundle['aggregation'])
    raw=fuse_scores(classifier['scores'],residual,bundle['classifier_threshold'],bundle['residual_threshold'])
    scored={'scores':{'learned':raw},'thresholds':{'learned':1.},
            'normalized_branch_scores':{'classifier':classifier['scores']/bundle['classifier_threshold'],
                                         'innovation':residual/bundle['residual_threshold']}}
    outputs={'v5':raw};states={}
    if plan['candidate_enabled']:
        policy=apply_to_scored(scored,record);outputs['v9']=policy['scores']['learned'];states=policy['policy_state']
        if not np.array_equal(np.isfinite(raw),np.isfinite(outputs['v9'])):
            raise ValueError('Transfer policy changed score availability.')
    masks=annotation_masks(record)
    arrays={'values':record['values'],'elapsed':record['elapsed'],'gap_before':record['gap_before'],
            'timestamps':np.asarray(record['timestamps']),'classifier_ratio':scored['normalized_branch_scores']['classifier'],
            'innovation_ratio':scored['normalized_branch_scores']['innovation'],
            'envelope':masks['envelope'],'core':masks['core'],'uncertain':masks['uncertain']}
    arrays.update({key+'_score':value for key,value in outputs.items()})
    arrays.update({'residual_'+key:value for key,value in states.items()})
    np.savez_compressed(target/'traces.npz',**arrays)
    metrics={key:transfer_metrics(record,scores) for key,scores in outputs.items()}
    unchanged=all(_hash(root/name)==digest for name,digest in plan['dependencies'].items())
    active=json.loads((target/'prefit_manifest.json').read_text())['active_hashes']
    active_unchanged=all(_hash(root/'artifacts'/name)==digest for name,digest in active.items())
    if not unchanged or not active_unchanged:raise ValueError('Protected transfer dependencies or active files changed.')
    report={'scope':'Four source interventions, same testbed, limited independent-record transfer diagnostic',
            'plan_sha256':_hash(target/'plan.json'),'candidate_enabled':plan['candidate_enabled'],
            'source_archive_sha256':ARCHIVE_SHA256,'source_long_sha256':record['source_long_sha256'],
            'pivot_sha256':record['sha256'],'long_rows':record['source_rows'],'pivot_rows':len(record['values']),
            'cadence_seconds':record['cadence_seconds'],'gap_count':int(record['gap_before'].sum()),
            'model_inference_passes':{'classifier':1,'innovation':1},'fitted_models':0,
            'environment':{'sklearn':sklearn.__version__,'numpy':np.__version__},'methods':metrics,
            'traces_sha256':_hash(target/'traces.npz'),'dependencies_unchanged':True,'active_v5_unchanged':True,
            'old_28_scored':False,'activation':False,
            'limits':['Four same-testbed interventions are not field validation or the 25/28 target.',
                      'Author-listed envelopes are derived annotations; gradual transitions remain uncertain.',
                      'Unannotated background is assumed, not independently adjudicated normality.',
                      'Elapsed exposure uses existing left-endpoint intervals with gaps excluded.',
                      'Any finite observation makes an intervention observable; row counts disclose partial coverage.',
                      'No startup trimming, unit correction, fitting or threshold selection on teaser.']}
    (target/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    return report
