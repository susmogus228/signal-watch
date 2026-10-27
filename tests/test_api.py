import json
import time
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient

import signalwatch.api as api


@pytest.fixture
def client(tmp_path, monkeypatch):
    root = tmp_path / "project"
    (root / "artifacts").mkdir(parents=True)
    (root / "artifacts/metadata.json").write_text(json.dumps({"version": "test", "model_sha256": "test-hash", "window": 10}))
    (root / "artifacts/frozen_evaluation.json").write_text(json.dumps({"methods": {}, "scope": "test fixture"}))
    sensors = [{"key": "Pressure", "label": "Pressure", "unit": "bar"}]

    def recording(raw=None, recording_id="known", name="Recorded run", mapping=None):
        if raw == b"bad":
            raise ValueError("CSV is missing the required sensor columns")
        if raw is None and recording_id != "known":
            raise KeyError(recording_id)
        return {"id": recording_id, "name": name, "group": "test", "split": "test", "values": np.ones((100, 1)),
                "timestamps": [str(i) for i in range(100)], "elapsed": np.arange(100.0),
                "labels": np.array([False] * 80 + [True] * 20), "changepoints": None,
                "gap_before": np.array([False] * 100), "cadence_seconds": 1.0, "sha256": "content-hash"}

    def score(bundle, data, thresholds=None):
        count = len(data["values"])
        array = np.array([np.nan if i < 10 else 2.0 for i in range(count)])
        return {"scores": {key: array for key in ("learned", "robust", "persistence")},
                "thresholds": {key: 2.0 for key in ("learned", "robust", "persistence")}}

    def inspect(data, scored):
        assert len(data["values"]) == len(scored["scores"]["learned"])
        labelled = data["labels"] is not None
        return {"annotations": [{"start_index": 80, "end_index": 99}] if labelled else [],
                "episodes": {key: [] for key in ("learned", "robust", "persistence")}, "events": [],
                "metrics": {"learned": {"total_events": 1}} if labelled else None}

    monkeypatch.setattr(api, "SENSORS", sensors)
    monkeypatch.setattr(api, "catalog", lambda root: [{"id": "known", "name": "Recorded run", "rows": 100, "split": "test"}])
    monkeypatch.setattr(api, "load_builtin", lambda root, recording_id: recording(recording_id=recording_id))
    monkeypatch.setattr(api, "parse_csv_bytes", recording)
    monkeypatch.setattr(api, "load_bundle", lambda root: {"window": 10, "metadata": {"version": "test"}})
    monkeypatch.setattr(api, "score_recording", score)
    monkeypatch.setattr(api, "inspect_recording", inspect)
    monkeypatch.setattr(api, "compare_validation", lambda root, bundle, budget: {"scope": "validation", "budget": budget})
    with TestClient(api.create_app(root, tmp_path / "runtime")) as test_client:
        yield test_client


def test_audit_and_causal_replay_have_separate_information(client):
    audit = client.get("/api/recordings/known").json()
    assert audit["cursor"] == 100 and audit["annotations"]
    assert audit["scores"]["learned"][10] == 1.0
    assert audit["scores"]["learned"][0] is None
    replay = client.get("/api/recordings/known?mode=replay&cursor=30").json()
    assert len(replay["elapsed"]) == len(replay["signals"][0]["values"]) == 30
    assert len(replay["scores"]["learned"]) == 30
    assert replay["annotations"] == [] and replay["metrics"] is None
    assert replay["complete"] is False
    assert client.get("/api/recordings/known?mode=replay&cursor=0").json()["elapsed"] == []


def test_zero_raw_threshold_keeps_alert_boundary_without_dividing_by_epsilon(client, monkeypatch):
    def score(bundle, data):
        n=len(data['values'])
        return {'scores':{'learned':np.resize([0.,.2],n),'robust':np.ones(n),'persistence':np.ones(n)},
                'thresholds':{'learned':0.,'robust':2.,'persistence':2.}}
    monkeypatch.setattr(api,'score_recording',score)
    view=client.get('/api/recordings/known').json()
    assert view['scores']['learned'][:2]==[1.,1.2]
    assert view['scores']['robust'][0]==.5
    assert view['raw_thresholds']['learned']==0.
    assert view['score_normalization']['learned']=='(score + 1) / (threshold + 1)'


def test_branch_context_uses_only_the_visible_event_interval(client,monkeypatch):
    def score(bundle,data):
        n=len(data['values'])
        classifier=np.r_[np.zeros(90),np.full(10,1.5)][:n]
        innovation=np.r_[np.full(50,2.),np.zeros(50)][:n]
        return {'scores':dict.fromkeys(['learned','robust','persistence'],np.ones(n)),
                'thresholds':dict.fromkeys(['learned','robust','persistence'],1.),
                'normalized_branch_scores':{'classifier':classifier,'innovation':innovation}}
    def inspect(data,scored):
        n=len(data['values'])
        return {'events':[{'id':'visible','start_index':0,'end_index':n-1}], 'episodes':{},'annotations':[],'metrics':None}
    monkeypatch.setattr(api,'score_recording',score)
    monkeypatch.setattr(api,'inspect_recording',inspect)
    audit=client.get('/api/recordings/known').json()['events'][0]['detector_branches']
    assert audit['classifier']=={'crossed_threshold':True,'max_relative_level':1.5}
    replay=client.get('/api/recordings/known?mode=replay&cursor=30').json()['events'][0]['detector_branches']
    assert replay['classifier']=={'crossed_threshold':False,'max_relative_level':0.}
    assert replay['innovation']=={'crossed_threshold':True,'max_relative_level':2.}


def test_changed_saved_evidence_produces_a_clear_service_error(client,monkeypatch):
    import signalwatch.development as development
    client.app.state.service.bundle['study_version']=5
    def unavailable(*args):
        raise ValueError('Saved development candidate evidence hash mismatch.')
    monkeypatch.setattr(development,'compare_saved_hybrid',unavailable)
    response=client.post('/api/comparisons',json={'budget':2})
    assert response.status_code==503
    assert 'evidence' in response.json()['detail'].lower()


def test_persisted_session_advance_reset_and_delete(client):
    session = client.post("/api/replays", json={"dataset_id": "known"}).json()
    session_id = session["id"]
    result = client.post(f"/api/replays/{session_id}/advance", json={"count": 15}).json()
    assert result["cursor"] == 15 and len(result["elapsed"]) == 15
    assert client.get(f"/api/replays/{session_id}").json()["cursor"] == 15
    assert client.post(f"/api/replays/{session_id}/advance", json={"count": 1000}).json()["cursor"] == 100
    assert client.post(f"/api/replays/{session_id}/reset").json()["cursor"] == 0
    assert client.delete(f"/api/replays/{session_id}").status_code == 200
    assert client.get(f"/api/replays/{session_id}").status_code == 404


def test_upload_deletion_removes_sessions_and_catalog_entry(client):
    response = client.post("/api/uploads", json={"filename": "sensors.csv", "csv": "valid fixture"})
    assert response.status_code == 201
    dataset_id = response.json()["dataset"]["id"]
    session_id = client.post("/api/replays", json={"dataset_id": dataset_id}).json()["id"]
    assert dataset_id in [item["id"] for item in client.get("/api/catalog").json()["datasets"]]
    assert client.delete(f"/api/uploads/{dataset_id}").status_code == 200
    assert client.get(f"/api/replays/{session_id}").status_code == 404
    assert client.get(f"/api/recordings/{dataset_id}").status_code == 404
    assert client.delete("/api/uploads/known").status_code == 403


def test_validation_and_errors_do_not_echo_raw_upload(client):
    response = client.post("/api/uploads", json={"filename": "sensors.csv", "csv": "bad"})
    assert response.status_code == 422
    secret = "private uploaded contents"
    response = client.post("/api/uploads", json={"filename": [], "csv": secret})
    assert response.status_code == 422 and secret not in response.text
    assert client.get("/api/recordings/missing").status_code == 404
    assert client.post("/api/replays", json={"dataset_id": "missing"}).status_code == 404
    assert client.get("/api/recordings/known?mode=replay&cursor=-1").status_code == 422
    assert client.post("/api/replays/missing/advance", json={"count": 0}).status_code == 422


def test_exploration_preserves_frozen_evidence_and_export_is_real(client):
    before = client.get("/api/evaluation").json()
    result = client.post("/api/comparisons", json={"budget": 2}).json()
    assert result["scope"] == "validation"
    assert client.get("/api/evaluation").json() == before
    assert client.post("/api/comparisons", json={"budget": 999}).status_code == 422
    exported = client.get("/api/recordings/known/export")
    assert exported.status_code == 200
    assert "attachment" in exported.headers["content-disposition"]
    assert exported.json()["view"]["recording"]["id"] == "known"
    assert exported.json()["model"]["model_sha256"] == "test-hash"
    assert client.get("/openapi.json").status_code == 200


def test_cross_origin_mutations_are_rejected_and_oversized_body_is_bounded(client):
    assert client.post("/api/replays", json={"dataset_id": "known"}, headers={"Origin": "https://example.com"}).status_code == 403
    response = client.post("/api/uploads", content=b"x" * (6 * 1024 * 1024 + 1), headers={"Content-Type": "application/json"})
    assert response.status_code == 413


def test_changed_model_requires_an_explicit_replay_restart(client):
    session_id = client.post("/api/replays", json={"dataset_id": "known"}).json()["id"]
    client.app.state.service.metadata["model_sha256"] = "changed-model"
    assert client.get(f"/api/replays/{session_id}").status_code == 409
    assert client.post(f"/api/replays/{session_id}/advance", json={"count": 1}).status_code == 409
    reset = client.post(f"/api/replays/{session_id}/reset")
    assert reset.status_code == 200 and reset.json()["cursor"] == 0
    assert reset.json()["model_sha256"] == "changed-model"


def test_training_job_conflict_and_explicit_failure_state(client, monkeypatch):
    entered = threading.Event()
    release = threading.Event()

    def fail_training(root):
        entered.set()
        release.wait(3)
        raise RuntimeError("Internal diagnostic that must not be returned")

    monkeypatch.setattr(api, "train_bundle", fail_training)
    job = client.post("/api/jobs/train").json()
    try:
        assert entered.wait(2)
        assert client.post("/api/jobs/train").status_code == 409
    finally:
        release.set()
    for _ in range(100):
        state = client.get(f"/api/jobs/{job['id']}").json()
        if state["status"] == "failed":
            break
        time.sleep(0.01)
    assert state["status"] == "failed" and state["finished_at"]
    assert "Internal diagnostic" not in json.dumps(state)
    assert client.get("/api/model").json()["model_sha256"] == "test-hash"
    assert client.get("/api/jobs/missing").status_code == 404


def test_missing_artifact_has_a_clear_unavailable_state(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "catalog", lambda root: [])

    def missing(root):
        raise FileNotFoundError("fixture")

    monkeypatch.setattr(api, "load_bundle", missing)
    with TestClient(api.create_app(tmp_path, tmp_path / "runtime")) as empty:
        assert empty.get("/api/health").json()["model_ready"] is False
        assert empty.get("/api/model").status_code == 503
        assert empty.get("/api/evaluation").status_code == 503
        assert empty.get("/api/catalog").json()["datasets"] == []


def test_deleted_upload_cannot_repopulate_recording_cache(client, monkeypatch):
    upload_id = client.post('/api/uploads', json={'filename':'qa.csv','csv':'valid fixture'}).json()['dataset']['id']
    original = api.parse_csv_bytes
    entered, release = threading.Event(), threading.Event()
    def slow_parse(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, 'parse_csv_bytes', slow_parse)
    responses = []
    reader = threading.Thread(target=lambda: responses.append(client.get(f'/api/recordings/{upload_id}')))
    reader.start()
    try:
        assert entered.wait(2)
        assert client.delete(f'/api/uploads/{upload_id}').status_code == 200
    finally:
        release.set()
        reader.join(4)
    assert responses[0].status_code == 404
    assert upload_id not in client.app.state.service.records
    assert client.get(f'/api/recordings/{upload_id}').status_code == 404


def test_deleted_upload_cannot_repopulate_score_cache(client, monkeypatch):
    upload_id = client.post('/api/uploads', json={'filename':'qa.csv','csv':'valid fixture'}).json()['dataset']['id']
    original = api.score_recording
    entered, release = threading.Event(), threading.Event()
    def slow_score(*args, **kwargs):
        entered.set()
        assert release.wait(3)
        return original(*args, **kwargs)
    monkeypatch.setattr(api, 'score_recording', slow_score)
    responses = []
    reader = threading.Thread(target=lambda: responses.append(client.get(f'/api/recordings/{upload_id}')))
    reader.start()
    try:
        assert entered.wait(2)
        assert client.delete(f'/api/uploads/{upload_id}').status_code == 200
    finally:
        release.set()
        reader.join(4)
    assert responses[0].status_code == 404
    assert not any(key[0] == upload_id for key in client.app.state.service.cache)


def test_view_reports_method_specific_warmup(client):
    service=client.app.state.service
    service.metadata={**service.metadata,'window':2,'warmup_by_method':{'learned':2,'robust':30,'persistence':30}}
    view=client.get('/api/recordings/known').json()
    assert view['warmup_samples']==2
    assert view['warmup_by_method']=={'learned':2,'robust':30,'persistence':30}


def test_inflight_view_keeps_its_model_metadata_snapshot(client, monkeypatch):
    original=api.score_recording
    service=client.app.state.service
    def replace_while_scoring(*args,**kwargs):
        service.metadata={**service.metadata,'model_sha256':'new-hash','window':2}
        return original(*args,**kwargs)
    monkeypatch.setattr(api,'score_recording',replace_while_scoring)
    view=client.get('/api/recordings/known').json()
    assert view['model_sha256']=='test-hash'
    assert view['warmup_samples']==10


def test_preloaded_cadence_is_declared_before_and_during_replay(client):
    catalog = client.get('/api/catalog').json()
    # Built-in fixture catalog is intentionally minimal; view metadata is authoritative.
    view = client.get('/api/recordings/known?mode=replay&cursor=0').json()
    assert view['recording']['cadence_basis'] == 'first_30_timestamp_intervals_of_complete_record'
    assert view['replay_context']['cadence_seconds'] == 1.0
    assert view['replay_context']['preloaded_metadata'] is True
    assert 'complete recording' in view['replay_context']['note']
    assert 'raw stream' in view['replay_context']['note']


def test_different_preloaded_cadence_can_change_identical_timestamp_prefix():
    from datetime import datetime, timedelta
    from signalwatch.dataset import SENSORS, parse_csv_bytes
    start = datetime(2026, 1, 1)
    records = []
    for step in (1, 10):
        seconds = [0, 1, 2, 6, 7] + [7 + step * i for i in range(1, 96)]
        rows = ['datetime;' + ';'.join(s['key'] for s in SENSORS)]
        rows += [(start + timedelta(seconds=t)).isoformat() + ';' + ';'.join(['1'] * 8) for t in seconds]
        records.append(parse_csv_bytes(('\n'.join(rows)).encode(), 'qa', 'qa'))
    assert records[0]['timestamps'][:5] == records[1]['timestamps'][:5]
    np.testing.assert_array_equal(records[0]['values'][:5], records[1]['values'][:5])
    assert [r['cadence_seconds'] for r in records] == [1., 10.]
    assert bool(records[0]['gap_before'][3]) is True
    assert bool(records[1]['gap_before'][3]) is False
    assert api.recording_metadata(records[0])['cadence_basis'] == api.recording_metadata(records[1])['cadence_basis']


def test_confirmed_view_exposes_raw_scores_and_visible_confirmation_provenance(client, monkeypatch):
    def score(bundle, data):
        n = len(data['values'])
        raw = np.full(n, 2.)
        decision = np.full(n, 2.); decision[:2] = 1.
        return {'scores': {key: decision.copy() for key in ('learned','robust','persistence')},
                'thresholds': {key: 1. for key in ('learned','robust','persistence')},
                'raw_scores': {'learned': raw},
                'alarm_policy': {'confirmation_seconds': 2., 'kind': 'continuous_elapsed_confirmation'},
                'policy_state': {'pending': np.arange(n)<2, 'confirmed': np.arange(n)>=2,
                                 'raw_episode_start': np.zeros(n, dtype=int)}}
    def inspect(data, scored):
        n = len(data['values'])
        starts = np.flatnonzero(np.asarray(scored['scores']['learned']) > 1)
        events = [] if not len(starts) else [
            {'id': 'fixture', 'status': 'unlabelled', 'start_index': int(starts[0]), 'end_index': n-1,
             'first_alert_seconds': float(starts[0]), 'delay_seconds': None}]
        return {'episodes': {key: [] for key in ('learned','robust','persistence')}, 'events': events,
                'metrics': None, 'annotations': []}
    monkeypatch.setattr(api, 'score_recording', score)
    monkeypatch.setattr(api, 'inspect_recording', inspect)
    view = client.get('/api/recordings/known?mode=replay&cursor=5').json()
    assert view['scores']['learned'] == [1., 1., 2., 2., 2.]
    assert view['raw_scores']['learned'] == [2.] * 5
    assert view['alarm_policy']['confirmation_seconds'] == 2.
    assert view['alarm_states']['pending'] == [True, True, False, False, False]
    assert view['alarm_states']['confirmed'] == [False, False, True, True, True]
    assert view['raw_metrics'] is None
    evidence = view['events'][0]['alarm_confirmation']
    assert evidence['raw_start_seconds'] == 0.
    assert evidence['confirmed_start_seconds'] == 2.
    assert evidence['wait_seconds'] == 2.
    assert all(len(x)==5 for x in view['raw_scores'].values())


@pytest.mark.parametrize('failed', [False, True])
@pytest.mark.parametrize('version', [6, 7])
def test_confirmed_budget_uses_saved_policy_selection_and_clear_errors(client, monkeypatch, failed, version):
    from types import SimpleNamespace
    from signalwatch import release
    service=client.app.state.service
    service.bundle['study_version']=version
    service.bundle['supervised_bundle']={'fixture':'confirmed child'}
    def compare(root,child,budget):
        assert child=={'fixture':'confirmed child'} and budget==2.
        if failed:
            raise ValueError('Changed evidence hash')
        return {'scope':'confirmed_grouped_oof_development_selection','budget':budget}
    monkeypatch.setattr(release,'study_module',lambda version:SimpleNamespace(compare_development_budget=compare))
    response=client.post('/api/comparisons',json={'budget':2.})
    if failed:
        assert response.status_code==503
        assert 'evidence' in response.json()['detail'].lower()
    else:
        assert response.status_code==200
        assert response.json()['scope']=='confirmed_grouped_oof_development_selection'
