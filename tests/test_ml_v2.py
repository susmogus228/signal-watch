import numpy as np
import pytest

from signalwatch.ml_v2 import (
    development_plan, innovation_scores, normal_training_arrays, score_recording,
    select_candidate,
)


class ZeroPredictor:
    def predict(self, X):
        return np.zeros((len(X), 8))


def record(count=100):
    rng=np.random.default_rng(42)
    return {'id':'synthetic-unit-test','values':rng.normal(size=(count,8)),
            'gap_before':np.zeros(count,bool),'labels':np.zeros(count,bool)}


def bundle(window=2):
    return {'model':ZeroPredictor(),'window':window,'scale':np.ones(8),
            'residual_scale':np.ones(8),'threshold':3.}


def test_prefix_translation_and_label_invariance():
    rec=record()
    full=innovation_scores(bundle(),rec)
    short={key:value[:40].copy() if isinstance(value,np.ndarray) else value for key,value in rec.items()}
    short['labels']=~short['labels']
    prefix=innovation_scores(bundle(),short)
    np.testing.assert_allclose(full['scores'][:40],prefix['scores'],equal_nan=True)
    shifted=dict(rec,values=rec['values']+np.arange(8)*100.)
    np.testing.assert_allclose(full['scores'],innovation_scores(bundle(),shifted)['scores'],equal_nan=True,atol=1e-11)
    renamed=dict(rec,id='another-name',timestamps=['irrelevant']*100)
    np.testing.assert_allclose(full['scores'],innovation_scores(bundle(),renamed)['scores'],equal_nan=True)


def test_two_sample_warmup_and_gap_reset():
    rec=record()
    rec['gap_before'][40]=True
    output=innovation_scores(bundle(),rec)
    assert np.isnan(output['scores'][:2]).all()
    assert np.isfinite(output['scores'][2:40]).all()
    assert np.isnan(output['scores'][40:42]).all()
    fresh={key:value[40:].copy() if isinstance(value,np.ndarray) else value for key,value in rec.items()}
    fresh['gap_before'][0]=False
    np.testing.assert_allclose(output['scores'][40:],innovation_scores(bundle(),fresh)['scores'],equal_nan=True)
    empty=record(0)
    assert len(innovation_scores(bundle(),empty)['scores'])==0


def test_normal_windows_exclude_anomaly_target_and_history_and_gaps():
    rec=record(20);rec['labels'][8]=True;rec['gap_before'][15]=True
    X,y,indices=normal_training_arrays(rec,2,np.ones(8))
    assert set(indices).isdisjoint({8,9,10,15,16})
    assert len(X)==len(y)==len(indices)
    assert np.isfinite(X).all()


def test_only_learned_is_replaced():
    rec=record();base={'scores':{'learned':np.zeros(100),'robust':np.ones(100),'persistence':np.ones(100)*2},
                            'thresholds':dict.fromkeys(['learned','robust','persistence'],1.)}
    result=score_recording(bundle(),rec,base)
    np.testing.assert_array_equal(result['scores']['robust'],base['scores']['robust'])
    np.testing.assert_array_equal(result['scores']['persistence'],base['scores']['persistence'])
    assert not np.isnan(base['scores']['learned']).any()


def test_feasibility_precedes_recall_and_run_folds_are_disjoint():
    candidates=[{'index':0,'feasible':False,'metrics':{'event_recall':1,'normal_alert_fraction':.7,'false_episodes_per_hour':.1}},
                {'index':1,'feasible':True,'metrics':{'event_recall':.2,'normal_alert_fraction':.01,'false_episodes_per_hour':1}}]
    assert select_candidate(candidates)['index']==1
    assert select_candidate(candidates[:1]) is None
    plan=development_plan(['a','b','c','d','e','f'],{})
    for fold in plan['folds']:
        assert set(fold['train_ids']).isdisjoint(fold['heldout_ids'])
    assert sorted(key for fold in plan['folds'] for key in fold['heldout_ids'])==list('abcdef')


def test_fitted_model_preserves_translation_and_uses_separate_calibration():
    from signalwatch.ml_v2 import _fit
    train=record(140)
    train['id']='train-run'
    calibration=record(120)
    calibration['id']='separate-calibration'
    calibration['values']=calibration['values']*1.7+40
    fitted,evidence=_fit([train],calibration,{'window':2,'min_samples_leaf':4,'n_estimators':64})
    assert evidence['training_ids']==['train-run']
    assert evidence['calibration_rows']==118
    assert evidence['training_rows']==138
    original=innovation_scores(fitted,train)
    shifted=innovation_scores(fitted,dict(train,values=train['values']+np.arange(8)*50.))
    np.testing.assert_allclose(original['scores'],shifted['scores'],equal_nan=True,atol=1e-10)


def test_actual_study_fit_groups_and_frozen_plan_hash():
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts/v2-development'
    if not (target/'report.json').exists():
        pytest.skip('Run the bounded development study for the artifact audit.')
    report=json.loads((target/'report.json').read_text())
    assert report['old_test_scored'] is False
    assert hashlib.sha256((target/'plan.json').read_bytes()).hexdigest()==report['plan_sha256']
    audits=json.loads((target/'candidate_audits.json').read_text())
    for candidate in audits:
        for fold in candidate['folds']:
            assert set(fold['training_ids']).isdisjoint(fold['heldout_ids'])
            assert set(fold['normal_training_windows']).isdisjoint(fold['heldout_ids'])
    assert report['metrics']['normal_alert_fraction']<=.05
    assert report['metrics']['false_episodes_per_hour']<=2.


def test_budget_comparison_reads_archive_and_preserves_artifacts(monkeypatch):
    import hashlib
    from pathlib import Path
    from signalwatch import ml
    from signalwatch.ml_v2 import compare_development_budget,load_development_bundle
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts/v2-development'
    if not (target/'report.json').exists():
        pytest.skip('Requires the fixed development study.')
    def forbidden_active_load(*args,**kwargs):
        raise AssertionError('Active model must not supply legacy comparison baselines.')
    monkeypatch.setattr(ml,'load_bundle',forbidden_active_load)
    paths=[target/'report.json',root/'artifacts/model.joblib',root/'artifacts/frozen_evaluation.json']
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    model=load_development_bundle(root)
    result=compare_development_budget(root,model,2.)
    assert result['scope']=='grouped_oof_development_selection'
    assert result['methods']['learned']==model['study_report']['metrics']
    assert before=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
