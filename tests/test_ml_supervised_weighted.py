import numpy as np
import pytest

from signalwatch.ml_supervised import causal_features, classifier_scores
from signalwatch.ml_supervised_weighted import sample_weights, fit_model, score_recording, study_plan


def record(n=30,name='a'):
    rng=np.random.default_rng(6)
    labels=np.arange(n)>=n//2
    values=rng.normal(size=(n,8));values[labels,0]+=2
    return {'id':name,'values':values,'labels':labels,'gap_before':np.zeros(n,bool)}


def test_record_equal_mass_and_class_balance_including_normal_anchor():
    a=record(30,'a');b=record(15,'b');b['labels'][:]=False
    indices=[causal_features(rec)[1] for rec in [a,b]]
    weights,evidence=sample_weights([a,b],indices,'record_equal_class_balanced')
    n=sum(map(len,indices));part=weights[:len(indices[0])]
    assert weights.sum()==pytest.approx(n)
    assert part.sum()==pytest.approx(n/2)
    assert weights[len(indices[0]):].sum()==pytest.approx(n/2)
    labels=a['labels'][indices[0]]
    assert part[labels].sum()==pytest.approx(n/4)
    assert part[~labels].sum()==pytest.approx(n/4)


def test_segment_mass_splits_at_label_changes_and_gaps():
    rec=record(30)
    rec['labels'][:]=False;rec['labels'][8:12]=True;rec['labels'][16:]=True
    rec['gap_before'][22]=True
    indices=causal_features(rec)[1]
    weights,evidence=sample_weights([rec],[indices],'segment_equal_class_balanced')
    labels=rec['labels'][indices]
    assert weights[labels].sum()==pytest.approx(len(indices)/2)
    pieces=[(indices>=8)&(indices<12),(indices>=16)&(indices<22),indices>=24]
    masses=[float(weights[mask].sum()) for mask in pieces]
    assert masses==pytest.approx([len(indices)/6]*3)
    assert set(indices).isdisjoint({0,1,22,23})


def test_unit_length_empty_eligibility_does_not_gain_weight():
    a=record(12,'a');empty=record(1,'empty')
    weights,evidence=sample_weights([a,empty],[causal_features(a)[1],np.zeros(0,dtype=int)],'record_equal_class_balanced')
    assert len(weights)==10
    assert weights.sum()==pytest.approx(10)
    assert evidence['nonempty_record_count']==1


def test_weighted_fit_causality_labels_ids_range_and_baselines():
    train=record(120)
    candidate={'algorithm':'ExtraTreesClassifier','n_estimators':128,'min_samples_leaf':4,'weighting':'record_equal_class_balanced'}
    model,evidence=fit_model([train],candidate)
    assert evidence['weight_sum']==pytest.approx(118)
    full=classifier_scores(model,train)['scores']
    prefix=dict(train,id='different',values=train['values'][:50],labels=~train['labels'][:50],gap_before=train['gap_before'][:50])
    np.testing.assert_allclose(full[:50],classifier_scores(model,prefix)['scores'],equal_nan=True)
    assert ((full[2:]>=0)&(full[2:]<=1)).all()
    base={'scores':{'learned':np.zeros(120),'robust':np.ones(120),'persistence':np.full(120,2.)},
          'thresholds':dict.fromkeys(['learned','robust','persistence'],1.)}
    result=score_recording(model,train,base)
    np.testing.assert_array_equal(result['scores']['robust'],base['scores']['robust'])
    np.testing.assert_array_equal(result['scores']['persistence'],base['scores']['persistence'])


def test_exact_four_candidate_plan_and_folds():
    plan=study_plan(['a','b','c','d','e','f'],{})
    assert len(plan['candidates'])==4
    assert plan['statistical_bootstrap']['performed'] is False
    for fold in plan['folds']:
        assert set(fold['train_ids']).isdisjoint(fold['heldout_ids'])


def test_actual_weighted_artifact_audit_prefix_and_gap_reset():
    import hashlib,json
    from pathlib import Path
    from signalwatch.dataset import load_builtin
    from signalwatch.ml_supervised_weighted import load_development_bundle
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts/supervised-weighted-development'
    if not (target/'report.json').exists():
        pytest.skip('Requires the fixed weighted development study.')
    model=load_development_bundle(root)
    report=model['study_report']
    assert report['old_test_scored'] is False
    assert hashlib.sha256((target/'plan.json').read_bytes()).hexdigest()==report['plan_sha256']
    assert hashlib.sha256((target/'selected_oof_scores.npz').read_bytes()).hexdigest()==report['oof_scores_sha256']
    audits=json.loads((target/'candidate_audits.json').read_text())
    for candidate in audits:
        for fold in candidate['folds']:
            assert set(fold['training_ids']).isdisjoint(fold['heldout_ids'])
            weights=fold['sample_weights']
            for row in weights['per_recording'].values():
                assert row['weight_sum']==pytest.approx(weights['eligible_rows']/weights['nonempty_record_count'])
    rec=load_builtin(root,'valve1-0')
    rec['gap_before'][100]=True
    full=classifier_scores(model,rec)['scores']
    assert np.isnan(full[100:102]).all()
    short=dict(rec,values=rec['values'][:150],gap_before=rec['gap_before'][:150],labels=~rec['labels'][:150])
    np.testing.assert_allclose(full[:150],classifier_scores(model,short)['scores'],equal_nan=True)
    fresh=dict(rec,values=rec['values'][100:],gap_before=rec['gap_before'][100:].copy())
    fresh['gap_before'][0]=False
    np.testing.assert_allclose(full[100:],classifier_scores(model,fresh)['scores'],equal_nan=True)


def test_weighted_budget_reproduces_selection_and_preserves_prior_artifacts(monkeypatch):
    import hashlib
    from pathlib import Path
    from signalwatch import ml
    from signalwatch.ml_supervised_weighted import load_development_bundle,compare_development_budget
    root=Path(__file__).resolve().parents[1]
    target=root/'artifacts/supervised-weighted-development'
    if not (target/'report.json').exists():
        pytest.skip('Requires fixed weighted development artifacts.')
    def forbidden(*args,**kwargs):
        raise AssertionError('Active model is not the fixed baseline source.')
    monkeypatch.setattr(ml,'load_bundle',forbidden)
    paths=[target/'report.json',root/'artifacts/v1/model.joblib',root/'artifacts/v2-development/report.json',
           root/'artifacts/supervised-development/report.json',root/'artifacts/model.joblib',root/'artifacts/frozen_evaluation.json']
    before={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
    model=load_development_bundle(root)
    comparison=compare_development_budget(root,model,2.)
    assert comparison['methods']['learned']==model['study_report']['metrics']
    assert comparison['scope']=='supervised_weighted_grouped_oof_development_selection'
    assert before=={path:hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
