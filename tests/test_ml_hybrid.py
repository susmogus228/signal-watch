import numpy as np
import pytest

from signalwatch.ml_hybrid import aggregate_residuals, fuse_scores, select_hybrid_candidate, score_recording
from signalwatch.evaluation import inspect_recording


def test_aggregation_is_spatial_and_keeps_missing_rows():
    channels=np.array([[1.,2,3,4,5,6,7,8],[np.nan]*8,[0.]*8])
    np.testing.assert_allclose(aggregate_residuals(channels,'max'),[8,np.nan,0],equal_nan=True)
    np.testing.assert_allclose(aggregate_residuals(channels,'top2_mean'),[7.5,np.nan,0],equal_nan=True)
    np.testing.assert_allclose(aggregate_residuals(channels,'all8_mean'),[4.5,np.nan,0],equal_nan=True)


def test_fusion_requires_both_finite_and_matches_raw_or():
    p=np.array([np.nan,.4,.9,.2,.5]);r=np.array([1.,np.nan,1.,4.,2.])
    fused=fuse_scores(p,r,.5,2.)
    assert np.isnan(fused[:2]).all()
    np.testing.assert_allclose(fused[2:],[1.8,2.,1.])
    np.testing.assert_array_equal(fused>1,np.isfinite(p)&np.isfinite(r)&((p>.5)|(r>2.)))
    with pytest.raises(ValueError):
        fuse_scores(p,r,.5,0.)


def test_existing_alarm_suppresses_a_later_branch_onset():
    n=20;labels=np.zeros(n,bool);labels[8:15]=True
    p=np.zeros(n);p[9:14]=.9
    residual=np.zeros(n);residual[5:11]=4.
    fused=fuse_scores(p,residual,.5,2.)
    record={'id':'unit','values':np.zeros((n,8)),'labels':labels,'elapsed':np.arange(n,dtype=float),
            'gap_before':np.zeros(n,bool),'cadence_seconds':1.}
    scores=dict.fromkeys(['learned','robust','persistence'],fused)
    report=inspect_recording(record,{'scores':scores,'thresholds':dict.fromkeys(scores,1.),'contributions':np.zeros((n,8))})
    assert report['metrics']['learned']['detected_events']==0
    assert report['metrics']['learned']['false_episodes']==1


def test_exact_ties_retain_classifier_control_and_constraints_win():
    metric={'event_recall':.8,'normal_alert_fraction':.01,'false_episodes_per_hour':1.}
    points=[{'index':1,'threshold_index':0,'feasible':True,'metrics':metric},
            {'index':0,'threshold_index':0,'feasible':True,'metrics':metric},
            {'index':2,'threshold_index':0,'feasible':False,'metrics':dict(metric,event_recall=1.)}]
    assert select_hybrid_candidate(points)['index']==0


class Classifier:
    classes_=np.array([0,1])
    def predict_proba(self,X):
        p=1/(1+np.exp(-X[:,0]))
        return np.c_[1-p,p]


class Predictor:
    def predict(self,X):
        return np.zeros((len(X),8))


def child_bundle():
    classifier={'model':Classifier(),'feature_center':np.zeros(120),'feature_scale':np.ones(120),
                'sensor_scale':np.ones(8),'window':2,'threshold':.8}
    innovation={'model':Predictor(),'scale':np.ones(8),'residual_scale':np.ones(8),'window':2,'threshold':2.}
    return {'classifier_bundle':classifier,'innovation_bundle':innovation,'aggregation':'max',
            'classifier_threshold':.8,'residual_threshold':2.,'window':2,'threshold':1.}


def test_hybrid_prefix_gap_labels_and_baseline_preservation():
    rng=np.random.default_rng(13);n=80
    record={'id':'unit','values':rng.normal(size=(n,8)),'labels':np.zeros(n,bool),'gap_before':np.zeros(n,bool)}
    record['gap_before'][40]=True
    base={'scores':{'learned':np.zeros(n),'robust':np.ones(n),'persistence':np.full(n,2.)},
          'thresholds':dict.fromkeys(['learned','robust','persistence'],1.)}
    full=score_recording(child_bundle(),record,base)
    assert np.isnan(full['scores']['learned'][40:42]).all()
    prefix={key:value[:50].copy() if isinstance(value,np.ndarray) else value for key,value in record.items()}
    prefix['id']='other';prefix['labels']=~prefix['labels'];prefix['timestamps']=['ignored']*50
    smallbase={'scores':{key:value[:50] for key,value in base['scores'].items()},'thresholds':base['thresholds']}
    short=score_recording(child_bundle(),prefix,smallbase)
    np.testing.assert_allclose(full['scores']['learned'][:50],short['scores']['learned'],equal_nan=True)
    np.testing.assert_array_equal(full['scores']['robust'],base['scores']['robust'])
    np.testing.assert_array_equal(full['scores']['persistence'],base['scores']['persistence'])


def test_frozen_plan_group_isolation_and_dependency_integrity():
    import hashlib,json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    path=root/'artifacts/hybrid-development/plan.json'
    if not path.exists():
        pytest.skip('Requires the frozen hybrid development plan.')
    plan=json.loads(path.read_text())
    assert len(plan['candidates'])==4
    assert plan['classifier_threshold']==.9954052245087474
    assert plan['old_test_access'] is False
    for fold in plan['folds']:
        assert set(fold['train_ids']).isdisjoint(fold['heldout_ids'])
    for filename,expected in plan['dependencies'].items():
        assert hashlib.sha256((root/filename).read_bytes()).hexdigest()==expected
    assert 'artifacts/v4/development/selected_oof_scores.npz' in plan['dependencies']
    assert not any(name.startswith('artifacts/supervised-weighted-development') for name in plan['dependencies'])


def test_actual_selected_hybrid_branch_reconciliation_and_prefix():
    from pathlib import Path
    from signalwatch.dataset import load_builtin
    from signalwatch.ml_hybrid import load_development_bundle
    root=Path(__file__).resolve().parents[1]
    if not (root/'artifacts/hybrid-development/report.json').exists():
        pytest.skip('Requires the bounded hybrid development study.')
    model=load_development_bundle(root)
    rec=load_builtin(root,'valve1-0');rec['gap_before'][100]=True
    n=len(rec['values'])
    base={'scores':{'learned':np.zeros(n),'robust':np.ones(n),'persistence':np.ones(n)*2},
          'thresholds':dict.fromkeys(['learned','robust','persistence'],1.)}
    full=score_recording(model,rec,base)
    expected=fuse_scores(full['branch_scores']['classifier'],full['branch_scores']['innovation'],
                         model['classifier_threshold'],model['residual_threshold'])
    np.testing.assert_allclose(full['scores']['learned'],expected,equal_nan=True)
    assert np.isnan(expected[100:102]).all()
    short=dict(rec,id='unrelated',values=rec['values'][:150],gap_before=rec['gap_before'][:150],labels=~rec['labels'][:150])
    shortbase={'scores':{key:value[:150] for key,value in base['scores'].items()},'thresholds':base['thresholds']}
    np.testing.assert_allclose(expected[:150],score_recording(model,short,shortbase)['scores']['learned'],equal_nan=True)
    assert len(model['feature_names'])==152


def test_saved_channel_scores_budget_reconciliation_and_no_writes(monkeypatch):
    import hashlib,json
    from pathlib import Path
    from signalwatch import ml
    from signalwatch.ml_hybrid import load_development_bundle,compare_development_budget
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts/hybrid-development'
    if not (target/'report.json').exists():
        pytest.skip('Requires the bounded hybrid development study.')
    report=json.loads((target/'report.json').read_text())
    for filename,key in [('residual_oof_channels.npz','channel_oof_sha256'),('classifier_oof_scores.npz','classifier_oof_sha256')]:
        assert hashlib.sha256((target/filename).read_bytes()).hexdigest()==report[key]
    def forbidden(*args,**kwargs):
        raise AssertionError('Hybrid comparisons must not load active models.')
    monkeypatch.setattr(ml,'load_bundle',forbidden)
    paths=[target/'report.json',root/'artifacts/model.joblib',root/'artifacts/frozen_evaluation.json',
           root/'artifacts/v4/development/report.json',root/'artifacts/v2-development/report.json']
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    model=load_development_bundle(root)
    result=compare_development_budget(root,model,2.)
    assert result['methods']['learned']==report['metrics']
    assert result['selected']==report['selected']
    assert before=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
