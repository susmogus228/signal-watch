import numpy as np
import pytest
import json
import hashlib

from signalwatch.trend_diagnostics import probability_trace,learned_events,run_diagnostics,verify_final_identity,verify_bundle_identity,prepare_diagnostics


def test_direct_probability_decision_does_not_delay_or_smooth_crossings():
    p=np.array([np.nan,np.nan,.8,.2,.9])
    trace=probability_trace(p,.5)
    np.testing.assert_allclose(trace['scores']['learned'],p/.5,equal_nan=True)
    assert np.flatnonzero(trace['scores']['learned']>1).tolist()==[2,4]


@pytest.mark.parametrize('threshold',[0.,-1.,np.nan,np.inf])
def test_invalid_cutoff_is_not_silently_replaced(threshold):
    with pytest.raises(ValueError):probability_trace(np.array([.1,.9]),threshold)


def test_unmeasured_baseline_and_attribution_fields_are_not_exported_as_results():
    event={'id':'e','status':'missed','delay_seconds':None,'baseline_overlap':{'robust':False},
           'contributing_sensors':[],'contributions_kind':'placeholder'}
    assert learned_events({'events':[event]})==[{'id':'e','status':'missed','delay_seconds':None}]


def test_completed_diagnostic_refuses_repeat_inference_before_loading_inputs(tmp_path):
    target=tmp_path/'outputs/v10-diagnostics';target.mkdir(parents=True)
    (target/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='already completed'):run_diagnostics(tmp_path)


def test_final_bundle_must_match_frozen_winner_not_only_feature_count():
    report={'selected':{'arm':'B','threshold':.8},'plan_sha256':'plan','selection_before_final_fit_sha256':'selection'}
    bundle={'arm':'B','threshold':.8,'plan_sha256':'plan','selection_before_final_fit_sha256':'selection','feature_names':['f']*186}
    verify_bundle_identity(bundle,report)
    for key,value in [('arm','C'),('threshold',.5),('plan_sha256','other'),('selection_before_final_fit_sha256','other')]:
        with pytest.raises(ValueError):verify_bundle_identity(dict(bundle,**{key:value}),report)


@pytest.mark.parametrize('study_directory',['artifacts/trend-development','artifacts/trend-backbones-development','artifacts/regime-development'])
def test_final_file_identity_rejects_wrong_model_before_any_inference(tmp_path,study_directory):
    target=tmp_path/study_directory;target.mkdir(parents=True)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    model=target/'selected_bundle.joblib';model.write_bytes(b'generated-model')
    points=target/'candidate_audits.json';points.write_text('[]')
    selected={'arm':'B','threshold':.8}
    selection=target/'selection_before_final_fit.json'
    selection.write_text(json.dumps({'selected':selected,'plan_sha256':'plan','candidate_audits_sha256':digest(points),'oof_hashes':{}}))
    report={'selected':selected,'plan_sha256':'plan','model_sha256':digest(model),
            'selection_before_final_fit_sha256':digest(selection),'candidate_audits_sha256':digest(points)}
    verify_final_identity(tmp_path,report,study_directory)
    model.write_bytes(b'wrong-model')
    with pytest.raises(ValueError,match='model'):verify_final_identity(tmp_path,report,study_directory)


def test_completed_backbone_study_has_its_own_repeat_inference_guard(tmp_path):
    target=tmp_path/'outputs/v11-diagnostics';target.mkdir(parents=True)
    (target/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='already completed'):
        run_diagnostics(tmp_path,'artifacts/trend-backbones-development')


def test_unknown_study_path_is_rejected_without_creating_outputs(tmp_path):
    with pytest.raises(ValueError,match='Unknown'):
        run_diagnostics(tmp_path,'../outside')
    assert not (tmp_path/'outputs').exists()


def test_regime_bundle_requires_198_features_and_the_frozen_semantic_identity():
    report={'selected':{'arm':'K2','threshold':.8},'plan_sha256':'plan',
            'selection_before_final_fit_sha256':'selection','feature_count':198}
    bundle={'arm':'K2','threshold':.8,'plan_sha256':'plan',
            'selection_before_final_fit_sha256':'selection','feature_names':['f']*198}
    verify_bundle_identity(bundle,report,'artifacts/regime-development')
    with pytest.raises(ValueError):
        verify_bundle_identity(dict(bundle,feature_names=['f']*186),report,'artifacts/regime-development')
    with pytest.raises(ValueError):
        verify_bundle_identity(bundle,dict(report,feature_count=186),'artifacts/regime-development')


def test_regime_dispatch_uses_the_fitted_regime_transform_not_the_186_column_scorer(monkeypatch):
    import sys
    from types import SimpleNamespace
    from signalwatch.trend_diagnostics import classifier_for_study
    marker=object()
    def regime_scores(bundle,record):
        assert bundle is marker and record=={'id':'example'}
        return {'scores':np.array([np.nan,np.nan,.7])}
    monkeypatch.setitem(sys.modules,'signalwatch.ml_regimes',SimpleNamespace(classifier_scores=regime_scores))
    result=classifier_for_study('artifacts/regime-development')(marker,{'id':'example'})
    np.testing.assert_allclose(result['scores'],[np.nan,np.nan,.7],equal_nan=True)


def test_completed_regime_study_refuses_repeat_inference_before_model_loading(tmp_path):
    target=tmp_path/'outputs/v12-diagnostics';target.mkdir(parents=True)
    (target/'report.json').write_text('{}')
    with pytest.raises(ValueError,match='already completed'):
        run_diagnostics(tmp_path,'artifacts/regime-development')


def test_selected_point_without_a_completed_final_fit_cannot_authorize_inference(tmp_path):
    study=tmp_path/'artifacts/regime-development';study.mkdir(parents=True)
    (study/'report.json').write_text(json.dumps({'selected':{'arm':'K2','threshold':.8},
                                                'status':'final_fit_rejected','final_fit':None}))
    with pytest.raises(ValueError,match='completed final'):
        prepare_diagnostics(tmp_path,'artifacts/regime-development')
    assert not (tmp_path/'outputs/v12-diagnostics/inference_started.json').exists()
