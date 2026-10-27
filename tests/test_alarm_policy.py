import numpy as np
import pytest

from signalwatch.alarm_policy import apply_confirmation
from signalwatch.evaluation import inspect_recording


def test_zero_duration_is_exact_identity_with_same_finite_exposure():
    raw=np.array([np.nan,.2,1.,2.,np.inf,-1.,3.])
    out=apply_confirmation(raw,np.arange(len(raw),dtype=float),np.zeros(len(raw),bool),0.)
    np.testing.assert_array_equal(out['decision_score'],raw)
    np.testing.assert_array_equal(out['confirmed'],np.isfinite(raw)&(raw>1))
    assert not out['pending'].any()


def test_irregular_elapsed_time_and_strict_observed_activation_no_backfill():
    elapsed=np.array([0.,.3,1.7,2.1,2.2])
    raw=np.array([2.,2.,2.,2.,1.])
    out=apply_confirmation(raw,elapsed,np.zeros(5,bool),2.)
    np.testing.assert_array_equal(out['confirmed'],[False,False,False,True,False])
    np.testing.assert_array_equal(out['pending'],[True,True,True,False,False])
    np.testing.assert_array_equal(out['decision_score'],[1.,1.,1.,2.,1.])
    np.testing.assert_array_equal(out['raw_episode_start'],[0,0,0,0,-1])
    prefix=apply_confirmation(raw[:3],elapsed[:3],np.zeros(3,bool),2.)
    np.testing.assert_array_equal(prefix['decision_score'],out['decision_score'][:3])


def test_gaps_nonfinite_nonpositive_reset_and_no_periodic_rearming():
    raw=np.array([2.,2.,2.,2.,2.,2.])
    gap=np.zeros(6,bool);gap[3]=True
    out=apply_confirmation(raw,np.arange(6.),gap,1.)
    np.testing.assert_array_equal(out['confirmed'],[False,True,True,False,True,True])
    np.testing.assert_array_equal(out['raw_episode_start'],[0,0,0,3,3,3])
    raw=np.array([2.,2.,np.nan,2.,2.,-1.,2.,2.,2.])
    out=apply_confirmation(raw,np.arange(9.),np.zeros(9,bool),2.)
    assert np.isnan(out['decision_score'][2])
    np.testing.assert_array_equal(np.flatnonzero(out['confirmed']),[8])
    assert out['raw_episode_start'][3]==3
    assert out['raw_episode_start'][6]==6
    assert np.array_equal(np.isfinite(raw),np.isfinite(out['decision_score']))


def metric_for(raw,labels,duration):
    n=len(raw);elapsed=np.arange(n,dtype=float);gaps=np.zeros(n,bool)
    out=apply_confirmation(raw,elapsed,gaps,duration)
    record={'id':'unit','values':np.zeros((n,8)),'labels':labels,'elapsed':elapsed,
            'gap_before':gaps,'cadence_seconds':1.}
    scores=dict.fromkeys(['learned','robust','persistence'],out['decision_score'])
    report=inspect_recording(record,{'scores':scores,'thresholds':dict.fromkeys(scores,1.),'contributions':np.zeros((n,8))})
    return report['metrics']['learned'],out


def test_normal_pending_start_confirmed_inside_anomaly_is_actual_new_hit():
    raw=np.zeros(12);raw[1:8]=2.;labels=np.zeros(12,bool);labels[3:7]=True
    metric,out=metric_for(raw,labels,3.)
    assert metric['detected_events']==1
    assert metric['mean_delay_seconds']==1.
    assert metric['false_episodes']==0
    assert out['raw_episode_start'][4]==1
    assert out['confirmed'][4] and not out['confirmed'][3]


def test_anomalous_pending_start_confirmed_in_normal_time_is_false():
    raw=np.zeros(12);raw[3:10]=2.;labels=np.zeros(12,bool);labels[3:5]=True
    metric,out=metric_for(raw,labels,3.)
    assert metric['detected_events']==0
    assert metric['false_episodes']==1
    assert out['raw_episode_start'][6]==3
    assert out['confirmed'][6] and not labels[6]


def test_invalid_time_shape_duration_and_exact_boundary():
    with pytest.raises(ValueError):
        apply_confirmation(np.ones(3),np.array([0.,2.,1.]),np.zeros(3,bool),1.)
    with pytest.raises(ValueError):
        apply_confirmation(np.ones(3),np.arange(3.),np.zeros(3,bool),-1.)
    out=apply_confirmation(np.full(3,2.),np.array([0.,1.,2.]),np.zeros(3,bool),2.)
    np.testing.assert_array_equal(out['confirmed'],[False,False,True])
