import hashlib
import json

import numpy as np
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from signalwatch.ml_calibrated import (
    calibration_weights, inner_splits, select_candidate, verify_development_sources,
    fit_ensemble, ensemble_scores, compare_development_budget,
)


def record(key,n=24):
    rng=np.random.default_rng(sum(map(ord,key)))
    labels=np.arange(n)>=n//2
    values=rng.normal(size=(n,8))+labels[:,None]*2
    return {'id':key,'group':'unit','values':values,'labels':labels,'elapsed':np.arange(n,dtype=float),
            'gap_before':np.zeros(n,bool),'cadence_seconds':1.,'sha256':key+'hash'}


def test_inner_record_splits_are_stable_disjoint_and_cover_calibration_once():
    ids=['r'+str(i) for i in range(8)]
    splits=inner_splits(ids)
    assert splits==inner_splits(ids[::-1])
    assert sorted(key for split in splits for key in split['calibration_ids'])==sorted(ids)
    for split in splits:
        assert not set(split['fit_ids'])&set(split['calibration_ids'])
        assert set(split['fit_ids'])|set(split['calibration_ids'])==set(ids)


def test_equal_record_calibration_does_not_balance_classes():
    weights=calibration_weights([np.zeros(3,int),np.array([0,0,0,0,0,1])])
    assert weights[:3].sum()==pytest.approx(weights[3:].sum())
    assert weights[3]==weights[-1]
    assert weights.sum()==pytest.approx(9)


def test_new_qualifying_floor_competes_without_identity_override():
    metric={'detected_events':26,'false_episodes':1,'normal_alert_fraction':.002,
            'median_delay_seconds':4.,'mean_delay_seconds':5.}
    points=[{'index':1,'threshold_order':0,'feasible':True,'metrics':metric},
            {'index':2,'threshold_order':0,'feasible':True,'metrics':dict(metric,detected_events=27)}]
    assert select_candidate(points)['index']==2
    assert select_candidate([]) is None


def test_sources_reject_changed_loaded_bytes_and_anchor():
    plan={'development_ids':['a'],'source_hashes':{'a':'ahash','anomaly-free':'nhash'}}
    verify_development_sources(plan,[{'id':'a','sha256':'ahash'}],{'sha256':'nhash'},{'sha256':'nhash'})
    with pytest.raises(ValueError,match='source hash'):
        verify_development_sources(plan,[{'id':'a','sha256':'changed'}],{'sha256':'nhash'},{'sha256':'nhash'})


def test_real_calibrated_ensemble_groups_probability_prefix_labels_and_gaps():
    records=[record('r'+str(i)) for i in range(6)]
    anchor=record('anchor');anchor['labels'][:]=False
    members,evidence=fit_ensemble(records,anchor)
    assert len(members)==3
    for pair in evidence['members']:
        assert not set(pair['fit_ids'])&set(pair['calibration_ids'])
        assert 'anchor' in pair['fit_ids'] and 'anchor' not in pair['calibration_ids']
    unseen=record('unseen',30);unseen['gap_before'][15]=True
    before=ensemble_scores(members,unseen)
    valid=np.isfinite(before['scores'])
    assert np.all((before['scores'][valid]>=0)&(before['scores'][valid]<=1))
    assert np.isnan(before['scores'][15:17]).all()
    changed=dict(unseen,id='not-feature',labels=None)
    np.testing.assert_allclose(ensemble_scores(members,changed)['scores'],before['scores'],equal_nan=True)
    prefix={key:(value[:12].copy() if isinstance(value,np.ndarray) else value) for key,value in unseen.items()}
    np.testing.assert_allclose(ensemble_scores(members,prefix)['scores'],before['scores'][:12],equal_nan=True)
    with pytest.raises(ValueError,match='Outer-held'):
        fit_ensemble(records,anchor,['r0'])


def test_frozen_estimator_calibration_leaves_actual_hgb_predictions_unchanged():
    rng=np.random.default_rng(17);X=rng.normal(size=(80,3));y=(X[:,0]>0).astype(int)
    with threadpool_limits(limits=2):
        hgb=HistGradientBoostingClassifier(max_iter=8,min_samples_leaf=3,early_stopping=False,random_state=1).fit(X,y)
        before=hgb.predict_proba(X).copy()
        calibrator=CalibratedClassifierCV(FrozenEstimator(hgb),method='sigmoid')
        calibrator.fit(X+1.,1-y)
        np.testing.assert_array_equal(hgb.predict_proba(X),before)
    assert list(calibrator.classes_)==[0,1]


def test_budget_uses_sha_verified_cached_points_only(tmp_path):
    target=tmp_path/'artifacts/calibrated-development';target.mkdir(parents=True)
    point={'index':1,'threshold_order':0,'rejection_reasons':['false_onset_budget'],
           'metrics':{'detected_events':26,'false_episodes':1,'false_episodes_per_hour':3.,
                      'normal_alert_fraction':.002,'median_delay_seconds':1.,'mean_delay_seconds':2.},
           'methods':{'learned':{'detected_events':26}}}
    path=target/'candidate_audits.json';path.write_text(json.dumps([point]))
    bundle={'study_report':{'candidate_audits_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                            'scope':'dev','reference_methods':{'learned':{'detected_events':28}}}}
    assert not compare_development_budget(tmp_path,bundle,2.)['feasible']
    assert compare_development_budget(tmp_path,bundle,4.)['selected']['index']==1
    path.write_text('[]')
    with pytest.raises(ValueError,match='hash'):
        compare_development_budget(tmp_path,bundle,4.)


def test_fused_scorer_preserves_baselines_and_has_no_confirmation_semantics(monkeypatch):
    import signalwatch.ml_calibrated as module
    n=8;classifier=np.r_[np.nan,np.nan,np.full(n-2,.8)]
    channels=np.full((n,8),2.);channels[:2]=np.nan
    monkeypatch.setattr(module,'ensemble_scores',lambda members,record:{'scores':classifier,'contributions':channels})
    monkeypatch.setattr(module,'innovation_scores',lambda bundle,record:{'contributions':channels})
    base={'scores':{'learned':np.zeros(n),'robust':np.arange(n),'persistence':-np.arange(n)},
          'thresholds':{'learned':10.,'robust':2.,'persistence':3.}}
    bundle={'classifier_ensemble':[{}, {}, {}],'innovation_bundle':{},'aggregation':'max',
            'classifier_threshold':.4,'residual_threshold':4.}
    output=module.score_recording(bundle,record('unseen',n),base)
    np.testing.assert_array_equal(output['scores']['robust'],base['scores']['robust'])
    np.testing.assert_array_equal(output['scores']['persistence'],base['scores']['persistence'])
    np.testing.assert_allclose(output['scores']['learned'][2:],2.)
    assert output['thresholds']['learned']==1.
    assert 'raw_scores' not in output and 'alarm_policy' not in output
    assert 'reference' in output['contributions_note'].lower()
