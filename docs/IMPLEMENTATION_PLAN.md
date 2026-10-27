# SignalWatch Implementation Plan

**Goal:** deliver the approved black/amber SignalWatch application with real trained ML, auditable recording replay and validated local workflows.
**Architecture:** a causal detector/evaluation core supplies a same-origin local FastAPI backend and React/ECharts frontend. Recorded SKAB data and immutable evaluation artifacts remain separate from local uploads, sessions and exploratory results.
**Tech Stack:** Python 3.11+, scikit-learn, FastAPI, SQLite, React, TypeScript, Apache ECharts, Vite.
**Spec:** docs/SPEC.md and docs/API_CONTRACT.md. User selection is approval of the second displayed concept; do not request a repeated visual approval.

## Global constraints and review focus

English sources/UI/docs; Russian chat. No global installation, indexing, remote publication, data upload or existing-project edits. Actual recordings and metrics replace mock contents. No causal fault-diagnosis claims, no future leakage, no test-label tuning, no point-adjustment scoring. Reject malformed CSV without echoing input; missing labels mean unknown metrics. Physical exposure excludes gaps. Preserve frozen evaluation during exploratory runs. Deleted uploads remove related sessions/caches. Atomic model replacement and consistent metadata during training. Responsive desktop/mobile and keyboard focus must be checked in browser.

## Memory and agents

Root owns API, storage, project config, requirements, documentation and integration. AI Engineer owns dataset.py/ml.py/evaluation.py, scripts and ML tests. Frontend Developer owns frontend/**. These workstreams are independent once the contract is written; training artifacts feed API integration. Model QA Specialist reviewed metrics/leakage. Code Reviewer and UI Designer supplied the integration and visual reviews after the configured narrower reviewer models proved unavailable. Workers handed ownership back before root integration fixes. Fresh Codebase Memory listing has no SignalWatch project; source fallback, no indexing. Earlier portfolio memories guide evidence boundaries, not current completion claims.

## Tasks

- [x] 1. Prepare isolated project runtime, retain selected reference and research, lock public data provenance and dependency versions. Verify paths and data files parse before modeling.
- [x] 2. ML worker: write failing tests for causal prefix invariance, split isolation, gaps and event metrics; implement ingestion, predictor/baselines, calibration and report generation; train with a predefined small search. Freeze manifests/hashes and report actual results, including baseline wins. Reproduce through the UI without tuning the final test.
- [x] 3. Frontend worker: build selected event-review layout and supporting Replay/Compare/Data & Model flows against the fixed contract. Verify typecheck/build and empty/error states, with real API fetches only.
- [x] 4. Root: write failing API/storage tests for replay truncation, invalid/missing data, upload/delete/export and frozen-report preservation. Implement API, safe local persistence and job lifecycle. Integrate real data/model and static frontend; run full backend tests and build.
- [x] 5. Independent ML/API/UI reviews and browser QA at reference desktop and narrow mobile sizes. Exercise replay/audit/filter/compare/upload-invalid-valid/export/training; verify delete through API tests/live request and UI confirmation/Cancel. Inspect console, fix P0-P2 findings and save design-qa.md. Keep local server open on port 8772; no deploy. Exact untested UI/device boundaries remain in VERIFICATION.md.
- [x] 6. Reconcile all agreed requirements against final evidence. Write README, model/data cards, resource/license notices, measured results and honest limitations. Confirm artifact hashes and provide the local preview and verification summary.

Meaningful tests precede implementation in each behavior owner. Use the smallest appropriate check set; rerun successful checks only after a relevant change. Do not call the whole project finished while any task above remains open.
