# API and module integration contract

## Current V5 behavior (2026-10-23)

The active learned detector fuses weighted gradient boosting and normal-only ExtraTrees prediction; the original predictor contract below remains historical context. `ml.score_recording` dispatches by study_version and retains archived V1 baselines. `ml.score_recording_v1` reproduces the original scorer. `train_bundle` repeats the selected release in staging; it replaces files only after per-recording/event equivalence checks. For V5, POST /api/comparisons uses development.compare_saved_hybrid and returns hybrid_grouped_oof_development_selection. It verifies candidate_audits.json against development_audits_sha256 in active model metadata, reapplies the requested budget to saved OOF points and never scores the old test. Missing, changed or malformed evidence returns HTTP 503 with a readable reproduction instruction. Budgets remain bounded to 0.25..10. The frozen scientific compare_validation path remains available for full recomputation.

View adds `warmup_by_method` (learned=2, robust=30, persistence=30), `raw_thresholds` and `score_normalization`. `score_unit` is `relative alert level`; displayed alert threshold stays 1. Positive raw thresholds use score/threshold; zero uses (score+1)/(threshold+1), declared per method. Metrics use raw scores. Events add `contributions_kind` and `contributions_note`; V5 sensor deviations describe the initial reference, not model attribution. `GET /api/evaluation` returns a saved retrospective audit with explicit separate recall-target and false-rate verdicts and provenance.

V5 metadata adds branch parameters, feature/example counts and the saved development evidence SHA. Each event adds detector_branches with classifier and innovation entries (keys follow model branch metadata), crossed_threshold and max_relative_level. This describes finite normalized branch values strictly above 1 inside the visible event interval. Replay bounds both the event interval and branch inspection to cursor; no future branch maximum or endpoint is exposed. Branch crossings do not establish fault cause or guarantee that branch started the event.


The root agent owns this contract and the API. Coordinate changes before diverging. JSON uses snake_case. API errors have {"detail":"readable message"}; validation errors never echo full uploaded CSV content. All URLs are same-origin. Recorded data and model metadata are real, never silently replaced with demos.

## Experimental integration, inactive after failed gates

The active release remains V5. Rejected V6/V7 study artifacts do not change GET /api/model or the saved active evaluation. The dispatcher and publisher support study versions6/7, but their acceptance gates prevent activating the observed failures.

View now also returns raw_scores={}, raw_episodes=[], alarm_policy=null, alarm_states=null and raw_metrics=null for ordinary V5/V7 scoring. These optional fields distinguish a future accepted confirmation policy from a different learned model. Version6 would expose raw ML traces and emitted confirmation states separately, cap pending decision scores at 1, and report actual confirmation timing. It is inactive. Version7 uses reference_methods for previous V5 results in study/audit reports; it never relabels V5 as a raw trace from the new model. Both experimental comparison paths read SHA-verified saved candidate evidence and preserve the active audit.

The calibrated publisher additionally requires exactly three classifier members and >=25/28, fewer false starts than preserved V5, no greater normal-alert seconds and identical normal exposure. V7's observed 23/28,81 false starts and 386 seconds fail that gate. Candidate API training/release was skipped; no reproducibility or activation claim follows from the development selection alone.

## Internal ML interface (AI Engineer ownership)

Implement signalwatch/dataset.py, ml.py, evaluation.py plus scripts/prepare_data.py, scripts/train.py and domain tests. Functions:

- dataset.SENSORS: list of {key,label,unit} for the exact eight SKAB sensor keys.
- dataset.load_builtin(root: Path, recording_id: str) -> dict; ids are trusted manifest entries, never arbitrary paths.
- dataset.catalog(root: Path) -> list[dict]: id,name,group,split,rows,duration_seconds,labelled,cadence_seconds,sha256. Built-ins include normal_audit as a separate virtual held-out normal recording when useful; never expose training/calibration slices as final-test evidence.
- dataset.parse_csv_bytes(raw: bytes, recording_id: str, name: str, mapping: dict|None=None) -> dict; mapping keys are canonical sensors plus timestamp, anomaly and changepoint, values are uploaded header names. With no mapping, accept exact SKAB names and datetime. Limit 100k rows / 5MiB, require >=100 rows and finite numeric sensors, strictly increasing timestamps; clear errors. Optional anomaly labels are binary and not model inputs.
- Recording dict fields: id,name,group,split,values(np.ndarray n x 8),timestamps(list[str]),elapsed(np.ndarray n),labels(np.ndarray bool or None),changepoints(np.ndarray bool or None),gap_before(np.ndarray bool),cadence_seconds(float),sha256(str). elapsed is actual seconds from first timestamp; intervals crossing a gap do not count as observed exposure.
- ml.train_bundle(root: Path) -> dict metadata: produce artifacts/model.joblib, metadata.json, frozen_evaluation.json, split_manifest.json and data inspection/provenance outputs. Preserve split memberships across retraining. JSON must be finite and portable. CLI scripts run from project root.
- ml.load_bundle(root: Path) -> dict, load only the app's trusted artifact path.
- ml.score_recording(bundle: dict, recording: dict, thresholds: dict|None=None) -> dict with scores: {learned,robust,persistence}: float arrays (NaN during warm-up permitted internally), thresholds: {learned,robust,persistence}: float. Each score depends only on past/present, and state resets on gaps. Scores divided by their own thresholds are used for shared plots. Include contributions n x 8 for learned residuals when available.
- evaluation.inspect_recording(recording: dict, scored: dict) -> dict with annotations and episodes (both in index and elapsed-time units), metrics per method and an event list. Episode merging/cooldown definitions must be frozen/documented, causal, gap-aware and consistent. UI event status is based on LEARNED method; include baseline overlap and evidence. Each event has id,title,status(detected|missed|false_alert|unlabelled),start_index,end_index,start_seconds,end_seconds,first_alert_seconds(nullable),delay_seconds(nullable),method,contributing_sensors(list of sensor keys). Title describes a channel/event, never mechanical diagnosis.
- evaluation.compare_validation(root: Path,bundle: dict,budget: float) -> dict: calibrate each method against requested empirical false episodes / observed normal hour on separate calibration material, then report validation-only results; never write frozen_evaluation.json. Bound budget 0.25..10. Return actual achieved exposure/rates; missing or short evidence explicit.

Use relative data paths in public manifests; no personal local paths. Exact artifact schema is domain-owned but must be sent to root after creation. No APIs/storage modules or frontend files belong to ML worker.

## Recorded replay timing boundary

Cadence is preloaded recording configuration, computed from the first 30 timestamp intervals of the complete CSV before replay. Recording metadata declares cadence_basis; View.replay_context declares preloaded_metadata, cadence_seconds and a readable note. Prefix score invariance is conditional on this fixed cadence/gap configuration. The application hides future sensor rows and labels, but does not demonstrate causal initialization from an unknown raw timestamp stream. Changing later calibration timestamps can change an earlier gap decision. The UI states this before replay starts.

## HTTP surface (root ownership)

- GET /api/health -> {status,model_ready,dataset_count,version}.
- GET /api/catalog -> {datasets:[catalog entries],sensors:[SENSORS],default_dataset:string}.
- GET /api/model -> metadata object plus ready:true; 503 if unavailable.
- GET /api/evaluation -> real frozen report.
- POST /api/comparisons body {budget:number} -> validation result; frozen audit unchanged.
- GET /api/recordings/{id}?mode=audit|replay&cursor=100 -> View below. audit reveals the complete saved record; replay returns only first cursor rows. Default audit for built-in event-review landing screen. Uploaded unlabeled records show unavailable quality, never false/missed labels.
- POST /api/replays body {dataset_id} -> {id,dataset_id,cursor:0,total,created_at}; persisted locally.
- GET /api/replays -> {sessions:[session summaries]}.
- GET /api/replays/{id} -> same View plus session id.
- POST /api/replays/{id}/advance body {count:1..10000} -> View at increased cursor, capped at total.
- POST /api/replays/{id}/reset -> View at cursor 0.
- DELETE /api/replays/{id} -> {deleted:true}.
- POST /api/uploads body {filename,csv,mapping?:dict} -> {dataset:catalog entry}; local persistence; schema failures 422, oversized body 413.
- DELETE /api/uploads/{id} -> {deleted:true}; delete associated cached scores/sessions and raw CSV. Built-in removal rejected.
- GET /api/recordings/{id}/export -> JSON attachment of actual audit trace/evidence and artifact provenance. No uploaded path disclosed.
- POST /api/jobs/train -> {id,status:queued|running}; no config knobs for final-test retuning. Only one training job at a time, 409 if busy.
- GET /api/jobs/{id} -> {id,status,message,created_at,finished_at,error?}; jobs have clear errors and reload artifacts only after successful atomic generation.
- GET /docs and /openapi.json -> interactive API; frontend production files served by same FastAPI app.

### View JSON

{
  "recording": {"id":"valve1-0","name":"...","group":"valve1","split":"test","rows":1000,"duration_seconds":999,"labelled":true,"cadence_seconds":1,"sha256":"..."},
  "mode":"audit", "cursor":1000, "total":1000, "complete":true,
  "elapsed":[0,1], "timestamps":["...","..."],
  "signals":[{"key":"Pressure","label":"Pressure","unit":"bar","values":[1.2,1.3]}],
  "scores":{"learned":[null,1.2],"robust":[null,0.8],"persistence":[null,0.7]},
  "score_unit":"score / own threshold", "thresholds":{"learned":1.0,"robust":1.0,"persistence":1.0},
  "gaps":[{"index":500,"seconds":10}],
  "annotations":[], "episodes":{"learned":[],"robust":[],"persistence":[]},
  "events":[], "metrics":null, "warmup_samples":30,
  "scope_note":"Recorded testbed experiment; deviations do not establish a physical fault cause."
}

All arrays have cursor length in replay. No future labels, events, durations or complete-recovery endpoints leak through replay. Audit metrics are available only with labels, and identify their full-record scope. Event entries carry optional evidence text based on actual channel contributions. A replay episode still active at cursor has ongoing:true; never show its future end. Known whole-record row count and nominal duration are allowed metadata. Large records may be bounded in UI rendering by sampling, while event boundaries/metrics always use all raw rows. Empty cursor is valid and chart shows warm-up/awaiting observations.

## Frontend ownership

Frontend Developer owns frontend/**, with root API contracts consumed by an api.ts client and TypeScript types. Build real navigation, recording/group selection, event filter/list/details, ECharts signal and normalized-score plots, annotation/alert lanes, replay controls, Compare screen, Data & Model screen, JSON export, local CSV upload/mapping/delete, train job status and docs link. No mocks/fallback numbers in delivered UI. Root may add integration fixes after the initial worker handoff; worker must announce handoff first.

Use npm dev/build, Vite proxy /api, /docs, /openapi.json to 127.0.0.1:8772. Production frontend/dist served by API (if a starter emits dist/client, tell root exact path). Assets only real icons/charts; approved screenshot is a reference, not a UI background.
