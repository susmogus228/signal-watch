import numpy as np
import hashlib
import json
import pytest

from signalwatch.ml_confirmed import apply_to_scored, choose_policy, compare_development_budget, evaluate_policies


def test_decision_exposure_raw_baselines_and_label_independence():
    raw=np.array([np.nan,2.,2.,2.,.5,2.,2.])
    record={'id':'a','elapsed':np.arange(7.),'gap_before':np.zeros(7,bool),'labels':np.zeros(7,bool)}
    base={'scores':{'learned':raw.copy(),'robust':np.arange(7.),'persistence':np.arange(7.)*2},
          'thresholds':dict.fromkeys(['learned','robust','persistence'],1.),
          'normalized_branch_scores':{'classifier':np.ones(7),'innovation':raw.copy()}}
    result=apply_to_scored(base,record,2.)
    np.testing.assert_array_equal(result['raw_scores']['learned'],raw)
    np.testing.assert_array_equal(result['scores']['robust'],base['scores']['robust'])
    np.testing.assert_array_equal(result['scores']['persistence'],base['scores']['persistence'])
    np.testing.assert_array_equal(np.isfinite(result['scores']['learned']),np.isfinite(raw))
    np.testing.assert_array_equal(base['scores']['learned'],raw)
    other=dict(record,id='changed',labels=~record['labels'])
    np.testing.assert_array_equal(result['scores']['learned'],apply_to_scored(base,other,2.)['scores']['learned'])
    assert result['alarm_policy']['confirmation_seconds']==2.
    assert result['policy_state']['raw_episode_start'][3]==1


def test_policy_ranking_rejects_lost_raw_hits_and_delay_exposure_changes():
    base={'false_episodes':5,'normal_alert_fraction':.01,'median_delay_seconds':2.,'mean_delay_seconds':3.}
    rows=[{'index':0,'feasible':True,'metrics':base},
          {'index':1,'feasible':False,'metrics':dict(base,false_episodes=0)},
          {'index':2,'feasible':True,'metrics':dict(base,false_episodes=2,normal_alert_fraction=.005)}]
    assert choose_policy(rows)['index']==2
    assert choose_policy(rows[:1])['index']==0


def test_budget_uses_only_verified_saved_candidates_and_keeps_nonbudget_guards(tmp_path):
    target=tmp_path/'artifacts/confirmed-development';target.mkdir(parents=True)
    metric={'false_episodes':1,'false_episodes_per_hour':1.,'normal_alert_fraction':.01,
            'median_delay_seconds':2.,'mean_delay_seconds':3.}
    rows=[{'index':0,'feasible':False,'rejection_reasons':['false_onset_budget'],
           'metrics':dict(metric,false_episodes=3,false_episodes_per_hour=3.),'methods':{'learned':{'name':'control'}}},
          {'index':1,'feasible':True,'rejection_reasons':[], 'metrics':metric,'methods':{'learned':{'name':'chosen'}}},
          {'index':2,'feasible':False,'rejection_reasons':['lost_raw_detected_event_ids'],
           'metrics':dict(metric,false_episodes=0,false_episodes_per_hour=0.),'methods':{'learned':{'name':'lost'}}}]
    path=target/'candidate_audits.json';path.write_text(json.dumps(rows))
    report={'candidate_audits_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'scope':'development-only','raw_methods':{'learned':{'name':'raw'}}}
    bundle={'study_report':report}
    answer=compare_development_budget(tmp_path,bundle,4.)
    assert answer['selected']['index']==1
    assert answer['methods']['learned']['name']=='chosen'
    assert answer['raw_methods']['learned']['name']=='raw'
    assert not compare_development_budget(tmp_path,bundle,.5)['feasible']
    path.write_text('[]')
    with pytest.raises(ValueError,match='hash'):
        compare_development_budget(tmp_path,bundle,2.)


def test_equal_hit_count_cannot_replace_a_lost_raw_event_with_a_gained_event():
    n=100;raw=np.zeros(n);raw[3]=2.;raw[9:14]=2.
    labels=np.zeros(n,bool);labels[3]=True;labels[10:13]=True
    record={'id':'paired','group':'unit','values':np.zeros((n,8)),
            'labels':labels,'elapsed':np.arange(n,dtype=float),
            'gap_before':np.zeros(n,bool),'cadence_seconds':1.}
    baseline={'scores':{'learned':raw,'robust':np.zeros(n),'persistence':np.zeros(n)},
              'thresholds':dict.fromkeys(['learned','robust','persistence'],1.)}
    _,points,_=evaluate_policies([record],{'paired':raw},{'paired':baseline})
    delayed=points[1]
    assert delayed['metrics']['detected_events']==points[0]['metrics']['detected_events']==1
    assert len(delayed['lost_event_ids'])==len(delayed['gained_event_ids'])==1
    assert 'lost_raw_detected_event_ids' in delayed['rejection_reasons']
    assert not delayed['feasible']
    assert all(point['normal_exposure_equal'] for point in points)


def test_loaded_source_bytes_must_match_frozen_development_and_normal_hashes():
    from signalwatch.ml_confirmed import verify_development_sources
    plan={'development_ids':['unit'],'source_hashes':{'unit':'correct','anomaly-free':'normal'}}
    records=[{'id':'unit','sha256':'correct'}]
    anchor={'sha256':'normal'};calibration={'sha256':'normal'}
    verify_development_sources(plan,records,anchor,calibration)
    with pytest.raises(ValueError,match='source hash'):
        verify_development_sources(plan,[{'id':'unit','sha256':'changed'}],anchor,calibration)
    with pytest.raises(ValueError,match='normal source hash'):
        verify_development_sources(plan,records,anchor,{'sha256':'changed'})
