"""Same-origin local API for recorded replay, trusted artifacts and local uploads."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.staticfiles import StaticFiles

from . import __version__
from .dataset import SENSORS, catalog, load_builtin, parse_csv_bytes
from .evaluation import compare_validation, inspect_recording
from .ml import load_bundle, score_recording, train_bundle
from .storage import Store, now


SCOPE_NOTE = "Recorded testbed experiment; deviations do not establish a physical fault cause."
CADENCE_NOTE = ("Recorded replay uses cadence preloaded from the first 30 timestamp intervals of the complete recording. "
                "Scores use past/current sensor rows given that configuration; unknown raw stream initialization is not tested.")
BODY_LIMIT = 6 * 1024 * 1024


def portable(value):
    """Replace non-finite warm-up values with JSON null, preserving real zeros."""
    if isinstance(value, np.ndarray):
        return portable(value.tolist())
    if isinstance(value, np.generic):
        return portable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): portable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [portable(item) for item in value]
    return value


def recording_metadata(recording: dict) -> dict:
    elapsed = recording["elapsed"]
    return {
        "id": recording["id"], "name": recording["name"], "group": recording.get("group", "uploaded"),
        "split": recording.get("split", "user"), "rows": len(recording["values"]),
        "duration_seconds": float(elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else 0.0,
        "labelled": recording.get("labels") is not None,
        "cadence_seconds": float(recording["cadence_seconds"]), "sha256": recording["sha256"],
        "cadence_basis": "first_30_timestamp_intervals_of_complete_record",
    }


class LocalRequestBoundary:
    """Bound request buffering and reject cross-site writes to the local server."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in {"GET", "HEAD", "OPTIONS"}:
            await self.app(scope, receive, send)
            return
        headers = {key.decode().lower(): value.decode() for key, value in scope["headers"]}
        origin = headers.get("origin")
        if origin and urlparse(origin).hostname not in {"localhost", "127.0.0.1", "testserver"}:
            await JSONResponse({"detail": "Cross-origin writes are not allowed."}, 403)(scope, receive, send)
            return
        try:
            if int(headers.get("content-length", "0")) > BODY_LIMIT:
                await JSONResponse({"detail": "Request exceeds the 6 MiB local upload limit."}, 413)(scope, receive, send)
                return
        except ValueError:
            await JSONResponse({"detail": "Invalid content length."}, 400)(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            size += len(chunk)
            if size > BODY_LIMIT:
                await JSONResponse({"detail": "Request exceeds the 6 MiB local upload limit."}, 413)(scope, receive, send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break
        consumed = False

        async def bounded_receive():
            nonlocal consumed
            if consumed:
                return await receive()
            consumed = True
            return {"type": "http.request", "body": b"".join(chunks), "more_body": False}

        await self.app(scope, bounded_receive, send)


class Service:
    def __init__(self, root: Path, runtime: Path):
        self.root = root
        self.store = Store(runtime)
        self.store.recover_jobs()
        self.lock = threading.RLock()
        self.training = False
        self.bundle = None
        self.metadata = {}
        self.report = None
        self.cache = OrderedDict()
        self.records = OrderedDict()
        self.builtin_catalog = catalog(root)
        self.reload_artifacts()

    def reload_artifacts(self):
        try:
            bundle = load_bundle(self.root)
            metadata = json.loads((self.root / "artifacts/metadata.json").read_text(encoding="utf-8"))
            report = json.loads((self.root / "artifacts/frozen_evaluation.json").read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            if self.bundle is not None:
                raise
            return
        with self.lock:
            self.bundle, self.metadata, self.report = bundle, metadata, report
            self.cache.clear()

    @property
    def model_sha(self) -> str:
        return self.metadata.get("model_sha256", self.metadata.get("sha256", "unavailable"))

    def ready(self):
        if self.bundle is None:
            raise HTTPException(503, "No trained artifact is available. Prepare the data and run local training.")

    def get_recording(self, recording_id: str) -> dict:
        with self.lock:
            if recording_id in self.records:
                self.records.move_to_end(recording_id)
                return self.records[recording_id]
        try:
            if recording_id.startswith("upload-"):
                entry = self.store.get_upload(recording_id)
                if entry is None:
                    raise KeyError(recording_id)
                result = parse_csv_bytes(self.store.upload_bytes(recording_id), recording_id, entry["name"], entry["mapping"] or None)
            else:
                if recording_id not in {item["id"] for item in self.builtin_catalog}:
                    raise KeyError(recording_id)
                result = load_builtin(self.root, recording_id)
        except (KeyError, FileNotFoundError):
            raise HTTPException(404, "Recording not found.") from None
        with self.lock:
            self.require_upload(recording_id)
            self.records[recording_id] = result
            while len(self.records) > 8:
                self.records.popitem(last=False)
        return result

    def require_upload(self, recording_id: str):
        """Caller holds the lifecycle lock before publishing parsed/scored data."""
        if recording_id.startswith('upload-') and self.store.get_upload(recording_id) is None:
            raise HTTPException(404, 'Recording not found.')

    def all_datasets(self) -> list[dict]:
        uploads = []
        for entry in self.store.list_uploads():
            try:
                uploads.append(recording_metadata(self.get_recording(entry["id"])))
            except HTTPException:
                continue
        return [*uploads, *self.builtin_catalog]

    def view(self, recording_id: str, mode: str = "audit", cursor: int | None = None) -> dict:
        self.ready()
        recording = self.get_recording(recording_id)
        total = len(recording["values"])
        count = total if mode == "audit" else min(total, max(0, cursor or 0))
        with self.lock:
            bundle = self.bundle
            model_sha = self.model_sha
            metadata = self.metadata
        part = dict(recording)
        for key in ("values", "timestamps", "elapsed", "gap_before", "labels", "changepoints"):
            if recording.get(key) is not None:
                part[key] = recording[key][:count]
        if mode == "replay":
            part["labels"] = part["changepoints"] = None
        cache_key = (recording_id, recording["sha256"], model_sha, mode, count)
        with self.lock:
            cached = self.cache.get(cache_key)
        if cached is None:
            scored = score_recording(bundle, part)
            inspected = inspect_recording(part, scored)
            if scored.get('raw_scores'):
                raw_trace = dict(scored, scores={**scored['scores'], **scored['raw_scores']})
                raw_inspected = inspect_recording(part, raw_trace)
                inspected['raw_metrics'] = raw_inspected.get('metrics')
                inspected['raw_episodes'] = raw_inspected.get('episodes', {}).get('learned', [])
            cached = (scored, inspected)
            with self.lock:
                self.require_upload(recording_id)
                self.cache[cache_key] = cached
                while len(self.cache) > 8:
                    self.cache.popitem(last=False)
        scored, inspected = cached
        thresholds = scored["thresholds"]
        normalized, normalization = {}, {}
        for key, values in scored['scores'].items():
            threshold=float(thresholds[key]);offset=1. if threshold==0 else 0.
            normalized[key]=(np.asarray(values,dtype=float)+offset)/(threshold+offset)
            normalization[key]='(score + 1) / (threshold + 1)' if offset else 'score / threshold'
        elapsed = part["elapsed"]
        gaps = [
            {"index": int(index), "seconds": float(elapsed[index] - elapsed[index - 1])}
            for index in np.flatnonzero(part["gap_before"]) if 0 < index < count
        ]
        episodes = copy.deepcopy(inspected.get("episodes", {}))
        events = copy.deepcopy(inspected.get("events", []))
        raw_episodes = copy.deepcopy(inspected.get('raw_episodes', []))
        for event in events:
            branches={}
            for name,values in scored.get('normalized_branch_scores',{}).items():
                start=max(0,int(event.get('start_index',0)))
                stop=min(count,int(event.get('end_index',count-1))+1)
                interval=np.asarray(values,dtype=float)[start:stop]
                finite=interval[np.isfinite(interval)]
                branches[name]={'crossed_threshold':bool(np.any(finite>1.)),
                                'max_relative_level':float(finite.max()) if len(finite) else None}
            if branches:
                event['detector_branches']=branches
            state = scored.get('policy_state', {})
            first = event.get('first_alert_seconds')
            if first is not None and 'raw_episode_start' in state:
                index = int(np.searchsorted(elapsed, first))
                if index < count:
                    raw_start = int(state['raw_episode_start'][index])
                    if 0 <= raw_start <= index:
                        event['alarm_confirmation'] = {
                            'raw_start_index': raw_start, 'raw_start_seconds': float(elapsed[raw_start]),
                            'confirmed_start_index': index, 'confirmed_start_seconds': float(elapsed[index]),
                            'wait_seconds': float(elapsed[index] - elapsed[raw_start]),
                        }
        if mode == "replay":
            for entries in [*episodes.values(), events, raw_episodes]:
                for entry in entries:
                    entry["ongoing"] = bool(entry.get("end_index", -1) >= count - 1)
        metadata_window = metadata.get("window", metadata.get("config", {}).get("window", 30))
        warmup_by_method = scored.get('warmup_by_method', metadata.get('warmup_by_method',
                                    {key:int(metadata_window) for key in thresholds}))
        return portable({
            "recording": recording_metadata(recording), "mode": mode, "cursor": count, "total": total,
            "complete": count == total, "elapsed": elapsed, "timestamps": part["timestamps"],
            "signals": [{**sensor, "values": part["values"][:, index]} for index, sensor in enumerate(SENSORS)],
            "scores": normalized, "score_unit": "relative confirmed decision level" if scored.get('raw_scores') else "relative alert level",
            "raw_scores": scored.get('raw_scores', {}), "alarm_policy": scored.get('alarm_policy'),
            "alarm_states": scored.get('policy_state'), "raw_episodes": raw_episodes,
            "raw_metrics": inspected.get('raw_metrics') if mode == 'audit' and recording.get('labels') is not None else None,
            "raw_thresholds": thresholds, "score_normalization": normalization,
            "thresholds": {key: 1.0 for key in thresholds}, "gaps": gaps,
            "annotations": inspected.get("annotations", []) if mode == "audit" else [],
            "episodes": episodes, "events": events,
            "metrics": inspected.get("metrics") if mode == "audit" and recording.get("labels") is not None else None,
            "warmup_samples": int(metadata_window), "warmup_by_method": warmup_by_method,
            "model_sha256": model_sha, "scope_note": SCOPE_NOTE,
            "replay_context": {"preloaded_metadata": True, "cadence_seconds": float(recording["cadence_seconds"]),
                               "note": CADENCE_NOTE},
        })


class StrictBody(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplayCreate(StrictBody):
    dataset_id: str = Field(min_length=1, max_length=120)


class Advance(StrictBody):
    count: int = Field(ge=1, le=10000, strict=True)


class Comparison(StrictBody):
    budget: float = Field(ge=0.25, le=10, allow_inf_nan=False)


class Upload(StrictBody):
    filename: str = Field(min_length=1, max_length=200)
    csv: str = Field(min_length=1, max_length=5 * 1024 * 1024)
    mapping: dict[str, str] | None = None


def create_app(root: Path | None = None, runtime: Path | None = None) -> FastAPI:
    root = Path(root or os.environ.get("SIGNALWATCH_ROOT", Path(__file__).resolve().parents[1])).resolve()
    runtime = Path(runtime or os.environ.get("SIGNALWATCH_RUNTIME_DIR", root / "runtime")).resolve()
    app = FastAPI(title="SignalWatch local API", version=__version__, description=SCOPE_NOTE)
    app.add_middleware(LocalRequestBoundary)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    service = Service(root, runtime)
    app.state.service = service

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        fields = [".".join(map(str, entry["loc"])) + ": " + entry["msg"] for entry in exc.errors()]
        return JSONResponse({"detail": "; ".join(fields)}, status_code=422)

    @app.get("/api/health")
    def health():
        return {"status": "ok", "model_ready": service.bundle is not None,
                "dataset_count": len(service.all_datasets()), "version": __version__}

    @app.get("/api/catalog")
    def datasets():
        rows = service.all_datasets()
        default = next((row["id"] for row in rows if row.get("split") == "test"), rows[0]["id"] if rows else "")
        return {"datasets": rows, "sensors": SENSORS, "default_dataset": default}

    @app.get("/api/model")
    def model():
        service.ready()
        return portable({**service.metadata, "ready": True, "model_sha256": service.model_sha})

    @app.get("/api/evaluation")
    def evaluation():
        service.ready()
        return portable(service.report)

    @app.post("/api/comparisons")
    def compare(body: Comparison):
        service.ready()
        with service.lock:
            if service.training:
                raise HTTPException(409, "Wait for the current training job to finish.")
            bundle = service.bundle
        if bundle.get('study_version') in (5,6,7):
            from .development import compare_saved_hybrid
            try:
                if bundle['study_version'] in (6,7):
                    from .release import study_module
                    return portable(study_module(bundle['study_version']).compare_development_budget(root,bundle['supervised_bundle'],body.budget))
                return portable(compare_saved_hybrid(root,bundle,body.budget))
            except (OSError,ValueError,KeyError):
                raise HTTPException(503,'Saved development evidence is unavailable or changed. Reproduce the frozen study before comparing budgets.') from None
        return portable(compare_validation(root, bundle, body.budget))

    @app.get("/api/recordings/{recording_id}")
    def recording(recording_id: str, mode: Literal["audit", "replay"] = "audit", cursor: int = Query(default=0, ge=0, le=100000)):
        return service.view(recording_id, mode, cursor)

    @app.get("/api/recordings/{recording_id}/export")
    def export(recording_id: str):
        with service.lock:
            payload = {"exported_at": now(), "view": service.view(recording_id), "model": portable(service.metadata)}
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", recording_id)[:100]
        return JSONResponse(payload, headers={"Content-Disposition": f'attachment; filename="signalwatch-{safe_name}.json"'})

    @app.post("/api/replays", status_code=201)
    def replay_create(body: ReplayCreate):
        service.ready()
        recording = service.get_recording(body.dataset_id)
        with service.lock:
            service.require_upload(body.dataset_id)
            return service.store.create_session(body.dataset_id, len(recording["values"]), service.model_sha)

    @app.get("/api/replays")
    def replays():
        return {"sessions": service.store.list_sessions()}

    def session(session_id: str, check_model: bool = True):
        result = service.store.get_session(session_id)
        if result is None:
            raise HTTPException(404, "Replay session not found.")
        if check_model and result["model_sha"] != service.model_sha:
            raise HTTPException(409, "The model changed. Restart this replay to use the current model.")
        return result

    def session_view(entry: dict):
        return {**service.view(entry["dataset_id"], "replay", entry["cursor"]), "id": entry["id"]}

    @app.get("/api/replays/{session_id}")
    def replay_get(session_id: str):
        with service.lock:
            return session_view(session(session_id))

    @app.post("/api/replays/{session_id}/advance")
    def replay_advance(session_id: str, body: Advance):
        with service.lock:
            session(session_id)
            return session_view(service.store.advance_cursor(session_id, body.count))

    @app.post("/api/replays/{session_id}/reset")
    def replay_reset(session_id: str):
        with service.lock:
            session(session_id, check_model=False)
            service.store.reset_session(session_id, service.model_sha)
            return session_view(session(session_id))

    @app.delete("/api/replays/{session_id}")
    def replay_delete(session_id: str):
        with service.lock:
            if not service.store.delete_session(session_id):
                raise HTTPException(404, "Replay session not found.")
        return {"deleted": True}

    @app.post("/api/uploads", status_code=201)
    def upload(body: Upload):
        raw = body.csv.encode("utf-8")
        if len(raw) > 5 * 1024 * 1024:
            raise HTTPException(413, "CSV exceeds the 5 MiB limit.")
        name = re.split(r"[/\\]", body.filename)[-1]
        name = re.sub(r"[\x00-\x1f\x7f]", "", name)[:120] or "recording.csv"
        if body.mapping and (len(body.mapping) > 12 or any(len(key) > 128 or len(value) > 128 for key, value in body.mapping.items())):
            raise HTTPException(422, "Column mapping is too large.")
        upload_id = "upload-" + uuid.uuid4().hex
        try:
            data = parse_csv_bytes(raw, upload_id, name, body.mapping)
        except (ValueError, UnicodeError) as error:
            raise HTTPException(422, str(error)) from None
        service.store.add_upload(upload_id, name, raw, body.mapping or {})
        return {"dataset": recording_metadata(data)}

    @app.delete("/api/uploads/{upload_id}")
    def upload_delete(upload_id: str):
        if not upload_id.startswith("upload-"):
            raise HTTPException(403, "Bundled recordings cannot be deleted here.")
        with service.lock:
            if not service.store.delete_upload(upload_id):
                raise HTTPException(404, "Uploaded recording not found.")
            service.records.pop(upload_id, None)
            for key in list(service.cache):
                if key[0] == upload_id:
                    del service.cache[key]
        return {"deleted": True}

    @app.post("/api/jobs/train", status_code=202)
    def train():
        with service.lock:
            if service.training:
                raise HTTPException(409, "A training job is already running.")
            service.training = True
            job = service.store.create_job()

        def run():
            try:
                service.store.update_job(job["id"], "running", "Training and evaluating the fixed reproducible configuration.")
                train_bundle(root)
                service.reload_artifacts()
                service.store.update_job(job["id"], "completed", "Training complete. The artifact and frozen report are ready.")
            except Exception:
                service.store.update_job(job["id"], "failed", "Training failed. Check the local training command and data files.", "Local training did not complete.")
            finally:
                with service.lock:
                    service.training = False

        threading.Thread(target=run, daemon=True, name="signalwatch-training").start()
        return job

    @app.get("/api/jobs/{job_id}")
    def training_job(job_id: str):
        job = service.store.get_job(job_id)
        if job is None:
            raise HTTPException(404, "Training job not found.")
        return job

    frontend = root / "frontend/dist/client"
    if not frontend.exists():
        frontend = root / "frontend/dist"
    if (frontend / "index.html").exists():
        app.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    else:
        @app.get("/", response_class=HTMLResponse)
        def not_built():
            return '<h1>SignalWatch</h1><p>Build the frontend with npm run build, then restart the local server.</p><a href="/docs">API documentation</a>'
    return app


app = create_app()
