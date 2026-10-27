"""Causal, normal-only ExtraTrees predictor and independent baseline calibration."""
from __future__ import annotations
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import sklearn
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import ExtraTreesRegressor
from signalwatch.dataset import SENSORS, SNAPSHOT, catalog, load_builtin, manifest, parse_csv_bytes
from signalwatch.evaluation import METHODS, aggregate_reports, durations, inspect_recording, intervals

WINDOW=30
PREDEFINED_CANDIDATES=[{'n_estimators':64,'min_samples_leaf':4}, {'n_estimators':64,'min_samples_leaf':12}]
DEFAULT_BUDGET=2.0


def _json(path: Path, value: dict):
    path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def _normal(root: Path) -> dict:
    entry=next(item for item in manifest(root)['recordings'] if item['id']=='anomaly-free')
    raw=(root/entry['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=entry['sha256']:
        raise ValueError('Pinned normal source hash mismatch.')
    record=parse_csv_bytes(raw,'anomaly-free','Normal source')
    record['labels']=np.zeros(len(record['values']),bool)
    return record


def _slice(record: dict, start: int, end: int, name: str) -> dict:
    output=dict(record,id=name,name=name)
    for key in ['values','timestamps','elapsed','labels','changepoints','gap_before']:
        if record[key] is not None:
            output[key]=record[key][start:end].copy() if isinstance(record[key],np.ndarray) else record[key][start:end]
    output['elapsed']=output['elapsed']-output['elapsed'][0]
    output['gap_before'][0]=False
    return output


def calibration_record(root: Path, split: dict) -> dict:
    return _slice(_normal(root),*split['normal_blocks']['calibration'],'normal_calibration')


def _features(values: np.ndarray, window: int) -> tuple[np.ndarray,np.ndarray]:
    if len(values)<=window:
        return np.zeros((0,32)), np.zeros((0,window,8))
    history=sliding_window_view(values,(window,8))[:,0,:,:][:-1]
    features=np.concatenate([history[:,-1],history.mean(axis=1),history.std(axis=1),history[:,-1]-history[:,0]],axis=1)
    return features,history


def _eligible(record: dict,window:int) -> np.ndarray:
    eligible=np.ones(max(0,len(record['values'])-window),bool)
    for index in np.flatnonzero(record['gap_before']):
        left=max(0,index-window)
        right=min(len(eligible),index)
        eligible[left:right]=False
    return eligible


def score_recording_v1(bundle: dict, recording: dict, thresholds: dict | None = None) -> dict:
    window=bundle['window']; n=len(recording['values'])
    scores={method:np.full(n,np.nan) for method in METHODS}
    contributions=np.full((n,8),np.nan)
    if n>window:
        values=(recording['values']-bundle['center'])/bundle['scale']
        features,history=_features(values,window)
        good=_eligible(recording,window)
        if good.any():
            predicted=bundle['model'].predict(features[good])
            residual=np.abs(values[window:][good]-predicted)/bundle['residual_scale']
            contributions[window:][good]=residual
            scores['learned'][window:][good]=residual.max(axis=1)
            med=np.median(history[good],axis=1)
            mad=1.4826*np.median(np.abs(history[good]-med[:,None,:]),axis=1)
            robust=np.abs(values[window:][good]-med)/np.maximum(mad,bundle.get('robust_floor',np.ones(8)*.05))
            scores['robust'][window:][good]=robust.max(axis=1)
            persistence=np.abs(values[window:][good]-history[good,-1])/bundle.get('persistence_scale',np.ones(8))
            scores['persistence'][window:][good]=persistence.max(axis=1)
    return {'scores':scores,'thresholds':dict(thresholds or bundle['thresholds']), 'contributions':contributions,'warmup_samples':window}


def score_recording(bundle: dict, recording: dict, thresholds: dict | None = None) -> dict:
    """Dispatch an explicit version while keeping the archived baselines stable."""
    version = bundle.get('study_version')
    if version not in (2, 3, 4):
        return score_recording_v1(bundle, recording, thresholds)
    if version in (3, 4):
        from signalwatch.release import study_module
        score_learned = study_module(version).score_recording
        child = bundle['supervised_bundle']
    else:
        from signalwatch.ml_v2 import score_recording as score_learned
        child = bundle['innovation_bundle']
    baseline = score_recording_v1(bundle['baseline_bundle'], recording)
    output = score_learned(child, recording, baseline)
    if thresholds is not None:
        output['thresholds'] = dict(thresholds)
    output['warmup_by_method'] = {
        'learned': child['window'],
        'robust': bundle['baseline_bundle']['window'],
        'persistence': bundle['baseline_bundle']['window'],
    }
    return output


def calibrate(record: dict, scored: dict, budget: float) -> tuple[dict,dict]:
    thresholds,details={},{}
    dt=durations(record)
    for method in METHODS:
        values=scored['scores'][method]
        finite=np.isfinite(values)
        seconds=float(dt[finite].sum())
        # Nonmonotone episode rates are evaluated explicitly, never binary-searched.
        candidates=np.unique(np.r_[np.quantile(values[finite],np.linspace(.90,1,101)),np.nextafter(values[finite].max(),np.inf)])
        accepted=[]
        for threshold in candidates:
            count=len(intervals(finite & (values>threshold),record['gap_before']))
            rate=count*3600/seconds if seconds else float('inf')
            if rate<=budget:
                accepted.append((float(threshold),count,rate))
        threshold,count,rate=accepted[0]
        thresholds[method]=threshold
        details[method]={'threshold':threshold,'false_episodes':count,'observed_normal_seconds':seconds,
                         'false_episodes_per_hour':rate,'requested_budget':budget,
                         'one_episode_resolution_per_hour':3600/seconds,
                         'short_exposure':seconds<3600,'candidate_quantiles':[.9,1.,101],
                         'note':'Empirical calibration only; this rate is not a future guarantee.'}
    return thresholds,details


def evaluate_split(root: Path,bundle: dict,split: str,protocol: str='onset_v2') -> dict:
    reports=[]; per_recording=[]; grouped={}
    entries=[entry for entry in catalog(root) if entry['split']==split]
    for entry in entries:
        record=load_builtin(root,entry['id'])
        scored=score_recording(bundle,record)
        report=inspect_recording(record,scored,protocol)
        reports.append(report)
        grouped.setdefault(record['group'],[]).append(report)
        per_recording.append({'id':record['id'],'group':record['group'],'rows':len(record['values']),
                              'metrics':report['metrics'],'events':report['events']})
    return {'scope':split,'recording_count':len(entries),'methods':aggregate_reports(reports),'per_recording':per_recording,'evaluation_protocol':protocol,
            'group_slices':{group:aggregate_reports(items) for group,items in grouped.items()}}


def train_bundle(root: Path) -> dict:
    if (Path(root)/'artifacts/release_manifest.json').exists():
        from signalwatch.release import reproduce_release
        return reproduce_release(Path(root))
    return train_bundle_v1(root)


def train_bundle_v1(root: Path) -> dict:
    root=Path(root); started=time.perf_counter()
    target=root/'artifacts';target.mkdir(exist_ok=True)
    normal=_normal(root); n=len(normal['values'])
    split_path=target/'split_manifest.json'
    split={'schema_version':1,'dataset_snapshot':SNAPSHOT,'window':WINDOW,
           'normal_source_sha256':normal['sha256'],'normal_blocks':{'train':[0,int(n*.55)],
           'calibration':[int(n*.55)+WINDOW,int(n*.80)],'audit':[int(n*.80)+WINDOW,n]},
           'purge_samples':WINDOW,'boundary_rule':'Half-open chronological blocks; next block starts30 samples after the preceding endpoint. Warm-up repeats inside each block.',
           'validation_ids':[item['id'] for item in manifest(root)['recordings'] if item['split']=='validation'],
           'test_ids':[item['id'] for item in manifest(root)['recordings'] if item['split']=='test']+['normal_audit'],
           'candidates':PREDEFINED_CANDIDATES,'alert_budget':DEFAULT_BUDGET,
           'selection_rule':'Maximize validation event recall minus 0.02 times validation false episodes/hour; tie uses candidate order.',
           'event_rule':'Contiguous label1 intervals, split at gaps. A hit requires an actual alert within the interval. No point adjustment.',
           'episode_rule':'Consecutive above-threshold samples; no merging, smoothing, cooldown or persistence requirement. Reset at gaps.',
           'gap_rule':'Reset when timestamp interval >3 times per-recording median cadence; gaps excluded from observed exposure.',
           'exposure_rule':'Left-endpoint intervals between observed samples, excluding gaps and unscored warm-up; terminal sample contributes0 seconds.'}
    if split_path.exists():
        old=json.loads(split_path.read_text(encoding='utf-8'))
        if old!=split:
            raise ValueError('Frozen split/config manifest differs; explicit versioned study is required.')
    else:
        _json(split_path,split)
    train=_slice(normal,*split['normal_blocks']['train'],'normal_train')
    calibration=calibration_record(root,split)
    center=np.median(train['values'],axis=0)
    scale=np.quantile(train['values'],.75,axis=0)-np.quantile(train['values'],.25,axis=0)
    scale=np.maximum(scale,np.maximum(np.std(train['values'],axis=0)*.1,1e-6))
    standardized=(train['values']-center)/scale
    features,_=_features(standardized,WINDOW)
    eligible=_eligible(train,WINDOW)
    X=features[eligible];y=standardized[WINDOW:][eligible]
    persistence_scale=np.maximum(np.median(np.abs(np.diff(standardized,axis=0)),axis=0),.03)
    trials=[];bundles=[]
    for candidate in PREDEFINED_CANDIDATES:
        model=ExtraTreesRegressor(**candidate,random_state=20261023,n_jobs=2,max_features=1.)
        model.fit(X,y)
        residual_scale=np.maximum(np.median(np.abs(y-model.predict(X)),axis=0),.03)
        bundle={'model':model,'window':WINDOW,'center':center,'scale':scale,'residual_scale':residual_scale,
                'persistence_scale':persistence_scale,'robust_floor':np.maximum(np.std(standardized,axis=0)*.05,.03),
                'thresholds':dict.fromkeys(METHODS,1.),'split_manifest':split}
        thresholds,cal=calibrate(calibration,score_recording(bundle,calibration),DEFAULT_BUDGET)
        bundle['thresholds']=thresholds
        # Preserve the original predefined selection objective. The corrected
        # onset-based reporting must not retrospectively select a new model.
        validation=evaluate_split(root,bundle,'validation','legacy_overlap_v1')
        learned=validation['methods']['learned']
        objective=(learned['event_recall'] or 0)-.02*(learned['false_episodes_per_hour'] or 0)
        trials.append({'parameters':candidate,'objective':objective,'methods':validation['methods'],'calibration':cal})
        bundles.append(bundle)
    selected=int(np.argmax([trial['objective'] for trial in trials]));bundle=bundles[selected]
    selection={'selected_candidate':selected,'selection_scope':'validation_only','trials':trials,
               'frozen_before_final_test':True,'split_manifest_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest()}
    # Selection is saved before the only final-test scoring pass in this training run.
    _json(target/'selection_manifest.json',selection)
    frozen=evaluate_split(root,bundle,'test')
    frozen.update(schema_version=2,dataset_snapshot=SNAPSHOT,selection=selection,
                  thresholds=bundle['thresholds'],scope_note='Recorded single-testbed audit; not production reliability, transfer performance or physical diagnosis.',
                  point_adjustment=False,definitions={key:split[key] for key in ['event_rule','episode_rule','gap_rule','exposure_rule']})
    frozen['definitions']['event_rule']='An annotated segment is detected only when a new alert episode STARTS inside it. An already active pre-onset alarm does not detect the segment. False episode onset is outside every annotation. No point adjustment.'
    frozen['selection_protocol']='legacy_overlap_v1 retained solely to reproduce the original model selection; final metrics and API events use onset_v2.'
    legacy_path=target/'frozen_evaluation_legacy_overlap.json'
    if legacy_path.exists():
        frozen['legacy_report']={'path':legacy_path.name,'sha256':hashlib.sha256(legacy_path.read_bytes()).hexdigest(),
                                 'note':'Superseded overlap-based event metrics; retained for transparent correction history.'}
    burden=frozen['methods']['learned']['normal_alert_fraction']
    frozen['operating_point_verdict']='Failed practical alert-burden check: the learned model stays in alert for almost all observed time. Event recall alone is misleading.' if burden is not None and burden>.5 else 'Inspect event recall together with normal alert time and baseline results; no production readiness claim.'
    frozen['reporting_revision']='Version2 corrects event matching to episode onsets and retains normal-alert exposure, point precision and group slices. Model, thresholds, selection and split are unchanged.'
    temporary=target/'model.joblib.tmp'
    joblib.dump(bundle,temporary,compress=3)
    model_sha=hashlib.sha256(temporary.read_bytes()).hexdigest()
    metadata={'name':'SignalWatch normal predictor','version':'1.0.0','algorithm':'ExtraTreesRegressor',
              'sklearn_version':sklearn.__version__,'window':WINDOW,'warmup_samples':WINDOW,'feature_count':32,
              'features':'Previous observation, past-window mean, population standard deviation and endpoint change;8 sensors each.',
              'sensor_keys':[sensor['key'] for sensor in SENSORS],'training_rows':len(X),'training_source_rows':len(train['values']),
              'training_scope':'Normal train block only; labels never enter scoring.',
              'parameters':PREDEFINED_CANDIDATES[selected],'model_sha256':model_sha,
              'thresholds':bundle['thresholds'],'calibration':trials[selected]['calibration'],
              'dataset_snapshot':SNAPSHOT,'training_seconds':round(time.perf_counter()-started,3),
              'created_at':datetime.now(timezone.utc).isoformat(),'split_manifest':split,
              'limitations':['Only one physical testbed and limited normal calibration exposure.',
                            'Prediction residuals are not fault probabilities or explanations of physical causes.',
                            'Empirical threshold budgets do not guarantee held-out false-alert rates.',
                            'Baselines can outperform the learned model. No production or equipment-transfer claim.']}
    metadata['operating_point_verdict']=frozen['operating_point_verdict']
    metadata['limitations'].insert(0,f"Frozen learned detector alarms for{100*frozen['methods']['learned']['alert_fraction']:.1f}% of scored time; inspect the operating-point verdict before use.")
    frozen['model_sha256']=model_sha
    _json(target/'frozen_evaluation.json.tmp',frozen)
    _json(target/'metadata.json.tmp',metadata)
    os.replace(temporary,target/'model.joblib')
    os.replace(target/'frozen_evaluation.json.tmp',target/'frozen_evaluation.json')
    os.replace(target/'metadata.json.tmp',target/'metadata.json')
    return metadata


def load_bundle(root: Path) -> dict:
    root=Path(root)
    metadata=json.loads((root/'artifacts/metadata.json').read_text(encoding='utf-8'))
    path=root/'artifacts/model.joblib'
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['model_sha256']:
        raise ValueError('Trusted model artifact hash does not match metadata.')
    bundle=joblib.load(path)
    bundle['metadata']=metadata
    return bundle
