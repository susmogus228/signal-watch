import numpy as np
import pytest

from signalwatch.ml_supervised import causal_features, classifier_scores, fit_model, score_recording, study_plan


def record(n=100):
    rng=np.random.default_rng(2026)
    values=rng.normal(size=(n,8))
    labels=np.arange(n)>=n//2
    values[labels,0]+=3.
    return {'id':'unit','values':values,'labels':labels,'gap_before':np.zeros(n,bool)}


class DummyClassifier:
    classes_=np.array([0,1])
    def predict_proba(self,X):
        p=1/(1+np.exp(-X[:,0]))
        return np.c_[1-p,p]


def bundle():
    return {'model':DummyClassifier(),'window':2,'feature_center':np.zeros(120),'feature_scale':np.ones(120),
            'sensor_scale':np.ones(8),'threshold':.8}


def test_prefix_current_label_id_time_independence_and_probability_range():
    rec=record()
    full=classifier_scores(bundle(),rec)
    short={key:value[:40].copy() if isinstance(value,np.ndarray) else value for key,value in rec.items()}
    short['labels']=~short['labels'];short['id']='renamed'
    short['timestamps']=['not-a-feature']*40
    prefix=classifier_scores(bundle(),short)
    np.testing.assert_allclose(full['scores'][:40],prefix['scores'],equal_nan=True)
    assert np.all((full['scores'][2:]>=0)&(full['scores'][2:]<=1))


def test_two_sample_reference_uses_past_and_resets_at_gap():
    rec=record(80)
    rec['gap_before'][30]=True
    X,indices,contrib=causal_features(rec)
    assert set([0,1,30,31]).isdisjoint(indices)
    assert set([2,29,32,79]).issubset(indices)
    row=np.flatnonzero(indices==2)[0]
    np.testing.assert_allclose(X[row,-8:],rec['values'][2]-rec['values'][:2].mean(axis=0))
    row=np.flatnonzero(indices==32)[0]
    np.testing.assert_allclose(X[row,-8:],rec['values'][32]-rec['values'][30:32].mean(axis=0))
    fresh={key:value[30:].copy() if isinstance(value,np.ndarray) else value for key,value in rec.items()}
    fresh['gap_before'][0]=False
    a=classifier_scores(bundle(),rec)['scores'][30:]
    b=classifier_scores(bundle(),fresh)['scores']
    np.testing.assert_allclose(a,b,equal_nan=True)


def test_early_long_windows_use_only_available_past():
    rec=record(10)
    X,indices,contrib=causal_features(rec)
    first=X[0]
    np.testing.assert_allclose(first[16:24],rec['values'][:2].mean(axis=0))
    np.testing.assert_allclose(first[88:96],rec['values'][:2].mean(axis=0))
    empty=classifier_scores(bundle(),record(0))
    assert len(empty['scores'])==0


def test_fit_group_preprocessing_isolation_and_baselines_unchanged():
    train=record(160);train['id']='training'
    candidate={'algorithm':'ExtraTreesClassifier','n_estimators':128,'min_samples_leaf':12}
    fitted,evidence=fit_model([train],candidate)
    assert evidence['training_ids']==['training']
    X,indices,_=causal_features(train)
    np.testing.assert_allclose(fitted['feature_center'],np.median(X,axis=0))
    assert evidence['normal_rows']>0 and evidence['anomalous_rows']>0
    base={'scores':{'learned':np.zeros(160),'robust':np.ones(160),'persistence':np.ones(160)*2},
          'thresholds':dict.fromkeys(['learned','robust','persistence'],1.)}
    scored=score_recording(fitted,train,base)
    np.testing.assert_array_equal(scored['scores']['robust'],base['scores']['robust'])
    np.testing.assert_array_equal(scored['scores']['persistence'],base['scores']['persistence'])
    assert not np.isnan(base['scores']['learned']).any()


def test_folds_and_feature_schema_are_predeclared():
    plan=study_plan(['a','b','c','d','e','f'],{})
    assert len(plan['candidates'])==4
    assert len(plan['feature_names'])==120
    assert plan['smoothing_samples']==1
    for fold in plan['folds']:
        assert set(fold['train_ids']).isdisjoint(fold['heldout_ids'])


def test_actual_selected_model_prefix_range_and_heldout_fit_audit():
    import hashlib,json
    from pathlib import Path
    from signalwatch.dataset import load_builtin
    from signalwatch.ml_supervised import load_development_bundle
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts/supervised-development'
    if not (target/'report.json').exists():
        pytest.skip('Run the supervised development study for the artifact check.')
    selected=load_development_bundle(root)
    report=selected['study_report']
    assert report['old_test_scored'] is False
    assert hashlib.sha256((target/'plan.json').read_bytes()).hexdigest()==report['plan_sha256']
    audits=json.loads((target/'candidate_audits.json').read_text())
    for candidate in audits:
        for fold in candidate['folds']:
            assert set(fold['training_ids']).isdisjoint(fold['heldout_ids'])
            assert set(fold['training_rows_by_recording']).isdisjoint(fold['heldout_ids'])
    rec=load_builtin(root,'valve1-0')
    full=classifier_scores(selected,rec)['scores']
    short=dict(rec,values=rec['values'][:100],gap_before=rec['gap_before'][:100],labels=~rec['labels'][:100])
    prefix=classifier_scores(selected,short)['scores']
    np.testing.assert_allclose(full[:100],prefix,equal_nan=True)
    finite=full[np.isfinite(full)]
    assert ((finite>=0)&(finite<=1)).all()


def test_budget_exploration_preserves_active_and_prior_studies(monkeypatch):
    import hashlib
    from pathlib import Path
    from signalwatch import ml
    from signalwatch.ml_supervised import compare_development_budget,load_development_bundle
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts/supervised-development'
    if not (target/'report.json').exists():
        pytest.skip('Requires supervised development artifacts.')
    def forbidden(*args,**kwargs):
        raise AssertionError('Active model cannot supply the fixed legacy baseline.')
    monkeypatch.setattr(ml,'load_bundle',forbidden)
    paths=[target/'report.json',root/'artifacts/v1/model.joblib',root/'artifacts/v2-development/report.json',
           root/'artifacts/model.joblib',root/'artifacts/frozen_evaluation.json']
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    bundle=load_development_bundle(root)
    result=compare_development_budget(root,bundle,2.)
    assert result['methods']['learned']==bundle['study_report']['metrics']
    assert result['scope']=='supervised_grouped_oof_development_selection'
    assert before=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
