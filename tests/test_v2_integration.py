import numpy as np
import pytest
import signalwatch.ml as ml
import signalwatch.ml_v2 as v2
import signalwatch.ml_supervised as supervised
from types import SimpleNamespace


class LastPredictor:
    def predict(self, X):
        return X[:,:8]


def legacy_bundle():
    return {'window':30,'model':LastPredictor(),'center':np.zeros(8),'scale':np.ones(8),
            'residual_scale':np.ones(8),'thresholds':{'learned':3.,'robust':4.,'persistence':5.}}


def test_v2_dispatch_preserves_legacy_baselines_and_threshold_override(monkeypatch):
    record={'values':np.arange(640.).reshape(80,8),'gap_before':np.zeros(80,bool)}
    baseline=legacy_bundle()
    original=ml.score_recording(baseline,record)
    def innovation(bundle, data, base):
        assert bundle=={'window':2,'threshold':7.}
        assert data is record
        return {**base,'scores':{**base['scores'],'learned':np.ones(80)*9},
                'thresholds':{**base['thresholds'],'learned':7.},'warmup_samples':2}
    monkeypatch.setattr(v2,'score_recording',innovation)
    wrapped={'study_version':2,'baseline_bundle':baseline,'innovation_bundle':{'window':2,'threshold':7.}}
    result=ml.score_recording(wrapped,record)
    for key in ('robust','persistence'):
        np.testing.assert_allclose(result['scores'][key],original['scores'][key],equal_nan=True)
        assert result['thresholds'][key]==original['thresholds'][key]
    assert result['warmup_by_method']=={'learned':2,'robust':30,'persistence':30}
    assert result['scores']['learned'][0]==9
    override={'learned':11.,'robust':12.,'persistence':13.}
    assert ml.score_recording(wrapped,record,override)['thresholds']==override


def test_explicit_v1_scorer_remains_available():
    record={'values':np.ones((50,8)),'gap_before':np.zeros(50,bool)}
    expected=ml.score_recording(legacy_bundle(),record)
    actual=ml.score_recording_v1(legacy_bundle(),record)
    for key in expected['scores']:
        np.testing.assert_allclose(expected['scores'][key],actual['scores'][key],equal_nan=True)


def test_supervised_dispatch_preserves_baselines_and_observation_semantics(monkeypatch):
    record={'values':np.ones((50,8)),'gap_before':np.zeros(50,bool)}
    baseline=legacy_bundle()
    def classify(bundle, data, base):
        return {**base,'scores':{**base['scores'],'learned':np.ones(50)*.8},
                'thresholds':{**base['thresholds'],'learned':.7},
                'contributions_kind':'reference_deviation_not_model_attribution'}
    monkeypatch.setattr(supervised,'score_recording',classify)
    wrapped={'study_version':3,'baseline_bundle':baseline,'supervised_bundle':{'window':2}}
    result=ml.score_recording(wrapped,record)
    assert result['thresholds']['learned']==.7
    assert result['contributions_kind']=='reference_deviation_not_model_attribution'
    assert result['warmup_by_method']=={'learned':2,'robust':30,'persistence':30}
    original=ml.score_recording_v1(baseline,record)
    for method in ['robust','persistence']:
        np.testing.assert_allclose(result['scores'][method],original['scores'][method],equal_nan=True)


@pytest.mark.parametrize('version', [5, 6, 7])
def test_hybrid_dispatch_uses_fused_scores_and_keeps_baseline_warmup(monkeypatch, version):
    from signalwatch import release
    record={'values':np.ones((50,8)),'gap_before':np.zeros(50,bool)}
    baseline=legacy_bundle();child={'window':2,'threshold':1.}
    def fuse(received, data, base):
        assert received is child and data is record
        return {**base,'scores':{**base['scores'],'learned':np.ones(50)*1.2},
                'thresholds':{**base['thresholds'],'learned':1.}}
    monkeypatch.setattr(release,'study_module',lambda version:SimpleNamespace(score_recording=fuse))
    result=ml.score_recording({'study_version':version,'supervised_bundle':child,'baseline_bundle':baseline},record)
    assert result['scores']['learned'][10]==1.2
    assert result['thresholds']['learned']==1.
    assert result['warmup_by_method']=={'learned':2,'robust':30,'persistence':30}
