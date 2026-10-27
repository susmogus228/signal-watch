import numpy as np
import pytest

from signalwatch.ml_ablation import (
    assert_development_ids,derive_arms,member_bank_scores,normal_thresholds,
    probability_diagnostics,select_development_candidate,
)
from signalwatch.ml_calibrated import fit_ensemble


def record(key,n=24):
    rng=np.random.default_rng(sum(map(ord,key)));labels=np.arange(n)>=n//2
    return {'id':key,'values':rng.normal(size=(n,8))+labels[:,None]*2,'labels':labels,
            'elapsed':np.arange(n,dtype=float),'gap_before':np.zeros(n,bool),'cadence_seconds':1.}


def test_arms_use_same_bank_and_exact_fixed_member_zero():
    raw=np.array([[np.nan]*3,[.1,.5,.9],[.2,.4,.8]])
    cal=np.array([[np.nan]*3,[.2,.3,.4],[.4,.5,.6]])
    arms=derive_arms(np.array([np.nan,.7,.8]),raw,cal)
    np.testing.assert_array_equal(arms['B'],raw[:,0])
    np.testing.assert_array_equal(arms['C'],cal[:,0])
    np.testing.assert_allclose(arms['D'][1:],[.5,1.4/3])
    np.testing.assert_allclose(arms['E'][1:],[.3,.5])
    with pytest.raises(ValueError,match='eligibility'):
        derive_arms(np.array([.1,.7,.8]),raw,cal)


def test_probability_losses_single_class_macro_and_pooled_are_explicit():
    rows=[{'id':'a','labels':np.array([0,0],bool)},{'id':'b','labels':np.array([0,1,1],bool)}]
    result=probability_diagnostics(rows,{'a':np.array([.1,.2]),'b':np.array([.1,.8,.9])})
    expected=(.01+.04+.01+.04+.01)/5
    assert result['pooled']['brier_loss']==pytest.approx(expected)
    assert result['per_recording']['a']['average_precision'] is None
    assert np.isfinite(result['per_recording']['a']['log_loss'])
    assert result['macro_record']['brier_loss']==pytest.approx((.025+.02)/2)


def test_threshold_dedup_retains_quantile_labels_and_above_max():
    grid=normal_thresholds(np.array([.2,.2,.2]))
    assert len(grid)==2
    assert len(grid[0]['quantile_labels'])==6
    assert grid[1]['threshold']>.2 and grid[1]['quantile_labels']==['above_max']


def test_rank_worst_family_then_delays_and_excludes_control_for_new_candidate():
    metric={'detected_events':26,'false_episodes':1,'false_alert_seconds':4.,'median_delay_seconds':2.,'mean_delay_seconds':3.}
    rows=[{'arm':'A','arm_order':0,'threshold_order':0,'feasible':True,'metrics':dict(metric,detected_events=28),'worst_family_false_episodes_per_hour':0.},
          {'arm':'B','arm_order':1,'threshold_order':0,'feasible':True,'metrics':metric,'worst_family_false_episodes_per_hour':2.},
          {'arm':'C','arm_order':2,'threshold_order':0,'feasible':True,'metrics':metric,'worst_family_false_episodes_per_hour':1.}]
    assert select_development_candidate(rows)['arm']=='C'
    assert select_development_candidate(rows[:1]) is None
    assert select_development_candidate([]) is None


def test_non_development_record_is_rejected_before_any_score():
    assert_development_ids(['dev0'],['dev0','dev1'])
    with pytest.raises(ValueError,match='development'):
        assert_development_ids(['test0'],['dev0','dev1'])


def test_actual_fixed_pair_raw_extraction_and_calibration_prefix_gaps_labels():
    rows=[record('r'+str(i)) for i in range(6)];anchor=record('anchor');anchor['labels'][:]=False
    members,evidence=fit_ensemble(rows,anchor,['heldout'])
    unseen=record('heldout',28);unseen['gap_before'][14]=True
    raw,cal=member_bank_scores(members,unseen)
    assert raw.shape==cal.shape==(28,3)
    assert np.isnan(raw[14:16]).all() and np.isnan(cal[14:16]).all()
    assert np.all((raw[np.isfinite(raw)]>=0)&(raw[np.isfinite(raw)]<=1))
    changed=dict(unseen,labels=None,id='not-a-feature')
    raw2,cal2=member_bank_scores(members,changed)
    np.testing.assert_allclose(raw,raw2,equal_nan=True);np.testing.assert_allclose(cal,cal2,equal_nan=True)
    prefix={key:(value[:10] if isinstance(value,np.ndarray) else value) for key,value in unseen.items()}
    r,c=member_bank_scores(members,prefix)
    np.testing.assert_allclose(r,raw[:10],equal_nan=True);np.testing.assert_allclose(c,cal[:10],equal_nan=True)
    assert all(x['frozen_hgb_predictions_unchanged'] for x in evidence['members'])
