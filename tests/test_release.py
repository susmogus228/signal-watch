import hashlib
import json
from types import SimpleNamespace

import pytest

from signalwatch import release


def write(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_initial_publication_rejects_a_different_child_with_same_selected_point(tmp_path,monkeypatch):
    artifacts=tmp_path/'artifacts';study=artifacts/'study';study.mkdir(parents=True)
    (study/'plan.json').write_bytes(b'{}')
    (study/'selected_bundle.joblib').write_bytes(b'replaced-model')
    write(study/'report.json',{'selected':{'index':0}})
    write(artifacts/'release_manifest.json',{'study_version':3,'study_directory':'study',
          'selected':{'index':0},'plan_sha256':hashlib.sha256(b'{}').hexdigest(),
          'audited_child_sha256':'original-audited-model'})
    monkeypatch.setattr(release,'study_module',lambda version:SimpleNamespace(load_development_bundle=lambda root:{}))
    with pytest.raises(ValueError,match='not the model measured'):
        release.publish_release(tmp_path)
    assert not (artifacts/'model.joblib').exists()


def test_failed_reproduction_retains_active_artifacts_and_exploration(tmp_path,monkeypatch):
    artifacts=tmp_path/'artifacts'
    write(artifacts/'release_manifest.json',{'study_version':3,'study_directory':'study','audit_path':'audit/report.json'})
    write(artifacts/'split_manifest.json',{})
    write(artifacts/'v1/preserved.json',{})
    write(artifacts/'study/plan.json',{})
    write(artifacts/'study/report.json',{'original':'OOF evidence'})
    write(artifacts/'audit/report.json',{})
    write(artifacts/'metadata.json',{'original':'active metadata'})
    write(tmp_path/'data/manifest.json',{})
    before={str(p.relative_to(artifacts)):p.read_bytes() for p in artifacts.rglob('*') if p.is_file()}
    def fail(stage):
        assert stage!=tmp_path
        write(stage/'artifacts/study/report.json',{'replacement':'must not reach active'})
        raise RuntimeError('Training failed after partial output')
    monkeypatch.setattr(release,'study_module',lambda version:SimpleNamespace(run_study=fail))
    with pytest.raises(RuntimeError,match='Training failed'):
        release.reproduce_release(tmp_path)
    after={str(p.relative_to(artifacts)):p.read_bytes() for p in artifacts.rglob('*') if p.is_file()}
    assert before==after


@pytest.mark.parametrize('changed',[False,True])
def test_reproduction_stages_frozen_scientific_sources_and_rejects_changes(tmp_path,monkeypatch,changed):
    artifacts=tmp_path/'artifacts';source=tmp_path/'signalwatch/scientific.py'
    source.parent.mkdir();source.write_bytes(b'fixed scientific implementation')
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    write(artifacts/'release_manifest.json',{'study_version':5,'study_directory':'study','audit_path':'audit/report.json'})
    write(artifacts/'split_manifest.json',{})
    write(artifacts/'v1/preserved.json',{})
    write(artifacts/'study/plan.json',{'dependencies':{'signalwatch/scientific.py':digest}})
    write(artifacts/'audit/report.json',{})
    write(artifacts/'metadata.json',{'original':'active'})
    write(tmp_path/'data/manifest.json',{})
    if changed:
        source.write_bytes(b'changed scientific implementation')
    calls=[]
    def study(stage):
        calls.append(stage)
        assert (stage/'signalwatch/scientific.py').read_bytes()==b'fixed scientific implementation'
        raise RuntimeError('Stop after verified staging')
    monkeypatch.setattr(release,'study_module',lambda version:SimpleNamespace(run_study=study))
    if changed:
        with pytest.raises(ValueError,match='Scientific source'):
            release.reproduce_release(tmp_path)
        assert calls==[]
    else:
        with pytest.raises(RuntimeError,match='verified staging'):
            release.reproduce_release(tmp_path)
    assert json.loads((artifacts/'metadata.json').read_text())=={'original':'active'}


@pytest.mark.parametrize('hits,false_count,normal_seconds,exposure', [
    (24, 8, 100., 18208.), (26, 56, 100., 18208.),
    (26, 8, 273., 18208.), (26, 8, 100., 18000.),
])
def test_confirmed_release_rejects_recall_burden_or_exposure_regressions(hits,false_count,normal_seconds,exposure):
    raw={'total_events':28,'detected_events':26,'false_episodes':56,
         'false_alert_seconds':272.,'observed_normal_seconds':18208.}
    learned={**raw,'detected_events':hits,'false_episodes':false_count,
             'false_alert_seconds':normal_seconds,'observed_normal_seconds':exposure}
    with pytest.raises(ValueError,match='acceptance'):
        release.require_confirmed_acceptance({'methods':{'learned':learned},'raw_methods':{'learned':raw}})


def test_confirmed_release_accepts_measured_improvement_without_claiming_two_per_hour():
    raw={'total_events':28,'detected_events':26,'false_episodes':56,
         'false_alert_seconds':272.,'observed_normal_seconds':18208.}
    learned={**raw,'detected_events':25,'false_episodes':12,'false_alert_seconds':120.}
    release.require_confirmed_acceptance({'methods':{'learned':learned},'raw_methods':{'learned':raw}})


def test_confirmation_retraining_also_checks_raw_model_before_replacement(tmp_path,monkeypatch):
    import signalwatch.ml as ml
    artifacts=tmp_path/'artifacts'
    write(artifacts/'release_manifest.json',{'study_version':6,'study_directory':'study','audit_path':'audit/report.json'})
    write(artifacts/'split_manifest.json',{})
    write(artifacts/'v1/preserved.json',{})
    write(artifacts/'study/plan.json',{})
    expected={'methods':{'confirmed':'same'},'group_slices':{},'per_recording':[],
              'raw_methods':{'raw':'original'},'raw_group_slices':{},'raw_per_recording':[]}
    write(artifacts/'audit/report.json',expected)
    write(artifacts/'metadata.json',{'active':'unchanged'})
    write(tmp_path/'data/manifest.json',{})
    module=SimpleNamespace(run_study=lambda stage:None,load_development_bundle=lambda stage:{})
    monkeypatch.setattr(release,'study_module',lambda version:module)
    monkeypatch.setattr(release.joblib,'load',lambda path:{})
    versions=[]
    def evaluate(stage,bundle,split):
        versions.append(bundle['study_version'])
        if bundle['study_version']==6:
            return {key:expected[key] for key in ['methods','group_slices','per_recording']}
        return {'methods':{'raw':'CHANGED'},'group_slices':{},'per_recording':[]}
    monkeypatch.setattr(ml,'evaluate_split',evaluate)
    monkeypatch.setattr(release,'publish_release',lambda *a,**k:pytest.fail('Must not publish a changed raw model'))
    with pytest.raises(ValueError,match='Raw model'):
        release.reproduce_release(tmp_path)
    assert versions==[6,5]
    assert json.loads((artifacts/'metadata.json').read_text())=={'active':'unchanged'}


def test_calibrated_acceptance_compares_with_preserved_reference_not_a_fake_raw_score():
    reference={'total_events':28,'detected_events':26,'false_episodes':56,
               'false_alert_seconds':272.,'observed_normal_seconds':18208.}
    learned={**reference,'detected_events':25,'false_episodes':8,'false_alert_seconds':100.}
    report={'methods':{'learned':learned},'reference_methods':{'learned':reference}}
    release.require_calibrated_acceptance(report)
    report['methods']['learned']['detected_events']=24
    with pytest.raises(ValueError,match='acceptance'):
        release.require_calibrated_acceptance(report)
