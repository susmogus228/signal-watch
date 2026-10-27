from signalwatch.storage import Store


def test_upload_and_sessions_survive_restart_and_delete_together(tmp_path):
    store = Store(tmp_path)
    store.add_upload("upload-one", "sensors.csv", b"sample", {"timestamp": "time"})
    session = store.create_session("upload-one", 200, "model-abc")
    store.update_cursor(session["id"], 48)
    reopened = Store(tmp_path)
    assert reopened.get_session(session["id"])["cursor"] == 48
    assert reopened.get_upload("upload-one")["mapping"] == {"timestamp": "time"}
    assert reopened.upload_bytes("upload-one") == b"sample"
    reopened.delete_upload("upload-one")
    assert reopened.get_upload("upload-one") is None
    assert reopened.get_session(session["id"]) is None
    assert not list((tmp_path / "uploads").glob("*.csv"))


def test_cursor_is_bounded_and_unknown_session_is_missing(tmp_path):
    store = Store(tmp_path)
    session = store.create_session("known", 100, "hash")
    store.update_cursor(session["id"], 999)
    assert store.get_session(session["id"])["cursor"] == 100
    store.update_cursor(session["id"], -10)
    assert store.get_session(session["id"])["cursor"] == 0
    assert store.get_session("missing") is None
    assert store.delete_session(session["id"]) is True
    assert store.delete_session(session["id"]) is False


def test_interrupted_training_gets_a_clear_failed_state(tmp_path):
    store = Store(tmp_path)
    job = store.create_job()
    store.update_job(job["id"], "running", "Training")
    store.recover_jobs()
    recovered = store.get_job(job["id"])
    assert recovered["status"] == "failed"
    assert "restarted" in recovered["message"].lower()
    assert recovered["finished_at"]
