import hashlib
import json

import pytest

from signalwatch.development import compare_saved_hybrid


def fixture(tmp_path):
    points=[]
    for index,recall,rate,fraction in [(0,.5,0.,.01),(1,.8,1.5,.02),(2,.9,3.,.02),(3,.95,1.,.06)]:
        points.append({'index':index,'threshold_index':0,'threshold':1.,'feasible':False,
                       'metrics':{'event_recall':recall,'false_episodes_per_hour':rate,'normal_alert_fraction':fraction}})
    path=tmp_path/'artifacts/hybrid-development/candidate_audits.json'
    path.parent.mkdir(parents=True);path.write_text(json.dumps(points),encoding='utf-8')
    bundle={'study_version':5,'metadata':{'development_audits_sha256':hashlib.sha256(path.read_bytes()).hexdigest()},
            'thresholds':{'learned':1.,'robust':2.,'persistence':3.},
            'supervised_bundle':{'study_report':{'scope':'hybrid_grouped_oof_development_selection',
             'methods':{'learned':{},'robust':{'detected_events':3},'persistence':{'detected_events':2}}}}}
    return path,bundle


@pytest.mark.parametrize('budget,index',[(1.,0),(2.,1),(4.,2)])
def test_budget_filters_saved_operating_points_without_rescoring(tmp_path,budget,index):
    path,bundle=fixture(tmp_path);before=path.read_bytes()
    result=compare_saved_hybrid(tmp_path,bundle,budget)
    assert result['selected']['index']==index
    assert result['selected']['feasible'] is True
    assert result['methods']['robust']=={'detected_events':3}
    assert result['scope']=='hybrid_grouped_oof_development_selection'
    assert path.read_bytes()==before


def test_changed_candidate_evidence_is_rejected(tmp_path):
    path,bundle=fixture(tmp_path);path.write_text('[]')
    with pytest.raises(ValueError,match='hash mismatch'):
        compare_saved_hybrid(tmp_path,bundle,2.)


def test_budget_bounds_still_apply(tmp_path):
    _,bundle=fixture(tmp_path)
    with pytest.raises(ValueError,match='Budget'):
        compare_saved_hybrid(tmp_path,bundle,0.)
