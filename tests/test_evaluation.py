import numpy as np
import pytest

from signalwatch.evaluation import inspect_recording


def test_delay_false_episode_and_gap_exposure():
    labels = np.zeros(20, bool)
    labels[5:9] = True
    scores = np.zeros(20)
    scores[[7, 8, 15]] = 2
    gap = np.zeros(20, bool)
    gap[10] = True
    elapsed = np.arange(20, dtype=float)
    elapsed[10:] += 100
    record = {'id': 'test', 'values': np.zeros((20,8)), 'elapsed': elapsed,
              'labels': labels, 'gap_before': gap, 'cadence_seconds': 1}
    scored = {'scores': {key: scores.copy() for key in ['learned','robust','persistence']},
              'thresholds': {key: 1 for key in ['learned','robust','persistence']},
              'contributions': np.zeros((20,8))}
    output = inspect_recording(record, scored)
    metric = output['metrics']['learned']
    assert metric['detected_events'] == 1
    assert metric['missed_events'] == 0
    assert metric['false_episodes'] == 1
    assert metric['mean_delay_seconds'] == 2
    assert metric['observed_normal_seconds'] < 20
    assert metric['false_episodes_per_hour'] == pytest.approx(3600 / metric['observed_normal_seconds'])


def test_no_labels_means_no_quality_metrics():
    record = {'id':'u','values':np.zeros((100,8)), 'elapsed':np.arange(100.),
              'labels':None, 'gap_before':np.zeros(100,bool), 'cadence_seconds':1}
    scores = {key: np.zeros(100) for key in ['learned','robust','persistence']}
    result = inspect_recording(record, {'scores':scores,'thresholds':dict.fromkeys(scores,1), 'contributions':np.zeros((100,8))})
    assert result['metrics'] is None


def test_continuous_alarm_cannot_hide_normal_alert_time():
    labels=np.zeros(100,bool);labels[40:60]=True
    record={'id':'always','values':np.zeros((100,8)),'elapsed':np.arange(100.),
            'labels':labels,'gap_before':np.zeros(100,bool),'cadence_seconds':1}
    scores={key:np.full(100,2.) for key in ['learned','robust','persistence']}
    output=inspect_recording(record,{'scores':scores,'thresholds':dict.fromkeys(scores,1),'contributions':np.zeros((100,8))})
    result=output['metrics']['learned']
    assert result['false_episodes']==1
    assert result['normal_alert_fraction']==1
    assert result['point_precision']==.2
    assert result['detected_events']==0
    assert result['missed_events']==1
    assert result['mean_delay_seconds'] is None


def test_new_episode_inside_event_has_actual_onset_delay():
    labels=np.zeros(100,bool);labels[40:70]=True
    record={'id':'onsets','values':np.zeros((100,8)),'elapsed':np.arange(100.),
            'labels':labels,'gap_before':np.zeros(100,bool),'cadence_seconds':1}
    signal=np.zeros(100);signal[10:45]=2;signal[48:55]=2
    scores={key:signal for key in ['learned','robust','persistence']}
    output=inspect_recording(record,{'scores':scores,'thresholds':dict.fromkeys(scores,1),'contributions':np.zeros((100,8))})
    result=output['metrics']['learned']
    assert result['detected_events']==1
    assert result['false_episodes']==1
    assert result['mean_delay_seconds']==8


def test_validation_comparison_preserves_frozen_artifact():
    from pathlib import Path
    from signalwatch.ml import load_bundle
    from signalwatch.evaluation import compare_validation
    root=Path(__file__).resolve().parents[1]
    path=root/'artifacts/frozen_evaluation.json'
    if not path.exists():
        pytest.skip('Requires trained artifact.')
    before=path.read_bytes()
    bundle=load_bundle(root)
    if bundle.get('study_version') in (6,7):
        from signalwatch.release import study_module
        output=study_module(bundle['study_version']).compare_development_budget(root,bundle['supervised_bundle'],2.)
    elif bundle.get('study_version')==5:
        from signalwatch.development import compare_saved_hybrid
        output=compare_saved_hybrid(root,bundle,2.)
    else:
        output=compare_validation(root,bundle,2.)
    expected={3:'supervised_grouped_oof_development_selection',4:'supervised_weighted_grouped_oof_development_selection',
              5:'hybrid_grouped_oof_development_selection',6:'confirmed_policy_oof_development_selection',
              7:'nested_calibrated_grouped_oof_development_selection'}.get(bundle.get('study_version'),'validation_only')
    assert output['scope']==expected
    assert path.read_bytes()==before


def test_observed_reference_deviation_is_not_called_model_attribution():
    labels=np.zeros(10,bool);labels[5:8]=True
    record={'id':'reference','values':np.zeros((10,8)),'elapsed':np.arange(10.),
            'labels':labels,'gap_before':np.zeros(10,bool),'cadence_seconds':1}
    scores={key:np.zeros(10) for key in ['learned','robust','persistence']}
    scored={'scores':scores,'thresholds':dict.fromkeys(scores,1),'contributions':np.ones((10,8)),
            'contributions_kind':'reference_deviation_not_model_attribution',
            'contributions_note':'Observed deviation from the initial two-sample reference; not model attribution or a physical cause.'}
    event=inspect_recording(record,scored)['events'][0]
    assert event['contributions_kind']=='reference_deviation_not_model_attribution'
    assert event['evidence']==scored['contributions_note']
    assert 'prediction residuals' not in event['evidence']
