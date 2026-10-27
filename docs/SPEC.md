> Historical V1 implementation brief. Current complementary V5 behavior and limitations are specified in MODEL_CARD.md, ML_EVALUATION.md and the Current V5 section of API_CONTRACT.md.

# SignalWatch implementation specification

Approved direction: second displayed concept, Event Review (black), chosen by the user on 2026-10-23. Visual truth: docs/design/approved-event-review.png. Research: docs/RESEARCH_AND_DESIGN.md. Build the complete bounded local ML application described there, rather than a static prototype.

The application compares learned prediction-error detection with rolling median/MAD and persistence baselines on SKAB recordings. It must expose real trained artifacts, real withheld-run results and traces for every hit, miss and false episode. It must not claim fault diagnosis, live production integration, a new algorithm or guaranteed superiority.

## Boundaries

- Root D:/Projects/signal-watch/. Existing projects, global environment, account/profile and remote publication remain unchanged.
- Python 3.11+, scikit-learn CPU training, FastAPI same-origin local API, React/TypeScript with Apache ECharts. Project-local dependencies only. No external inference API.
- Pin SKAB GitHub snapshot b2c0d46c2971dcbfe71e26087b6d231998bb91c2; retain source hashes and GPL-3.0 notices. Project source license is not selected in this task.
- All code, comments, UI and documentation are English. Chat remains Russian.
- Uploads and replay metadata remain in runtime/ locally, with deletion controls. No uploaded model/pickle loading and no external upload.
- Do not copy illustrative numbers, sample rates or curve shapes from the reference. Use measured records and source-unit labels. Anomaly scores are not probabilities.

## Core features

1. Browse built-in recordings by split/group and open actual recorded audit state, with an episode list and signal/evidence inspector matching the chosen black/amber layout.
2. Replay sequentially with pause, step, speed and restart. API payloads in replay reveal only available rows and alerts; known labels are hidden until explicit audit. Gaps/warm-up are visible and causal state resets on experiment/gap boundaries.
3. Compare learned and robust-statistics alerts in normalized score/threshold units, plus known annotations in retrospective audit. Show contributing sensor deviations, not inferred mechanical causes.
4. Frozen evaluation: event hits/misses, first actual post-onset delay, false episodes per observed normal exposure, alert duration and ordinary point-wise PR-AUC. Validation-only exploratory comparisons never alter the frozen report.
5. Data/model view: dataset snapshot/splits, actual model hash/features/calibration, repeatable local training job with progress/error state, and actual model/data cards.
6. Local CSV import with explicit mapping to the eight expected sensor features and timestamp, optional labels, strict size/shape/time/numeric validation, export and delete. No labels means no claimed detection-quality metrics.
7. Reproducible training/evaluation scripts, meaningful tests, OpenAPI documentation, English README/run guide/resource register and accurate verification report.

## Visual contract

Black base #0f0f0f; panel #1a1a1a; border #2a2a2a; warm white text; amber #edb347 primary accent. Semantic red for misses, muted teal for completion where appropriate. System sans and monospace timestamps, 8px spacing rhythm, modest radii. Use a single outline-icon family. No decorative stock images or bespoke CSS illustration is needed: graphs are data visualizations.

At desktop, a roughly 285px event queue and broad detail canvas. Retain compact top navigation Replay / Compare / Data & Model. Audit view shows complete historical traces; sequential replay must be distinct. At 390px, stack selectors/events/charts without hiding controls or forcing page-level horizontal scroll. Chart internals may zoom; comparison tables can scroll in their own region. Provide keyboard focus, descriptive chart summaries, reduced-motion, loading/error/empty and no-model states.

## Evaluation and acceptance

Freeze run-level splits before evaluating. Fit scaling/features on normal training data only. Purge chronological block boundaries by at least the lag window. Reserve normal audit exposure and whole disturbed final-test runs. Select only a small predefined candidate set on validation. Never mix random overlapping windows, optimize on final-test labels, interpolate from future points or use point-adjustment scoring.

Tests must prove future-prefix invariance, gaps, event matching/delay/false episodes, correct rates/denominators, malformed imports, label absence, replay truncation, storage deletion and frozen-report preservation. UI verification covers desktop and mobile, navigation, replay/step/reset/audit, comparison, import/error/delete, export, training state and console errors. Check all metrics against saved artifacts before reporting completion. No performance target is promised before measurement.

## Memory and agents

Codex memory: prior portfolio conventions support evidence and scope boundaries; verify actual files rather than inheriting old completion claims. Codebase Memory: fresh list_projects contained no SignalWatch project; do not index without authorization. Use source files for this newly created codebase.

Independent workers: AI Engineer owns dataset/ML/evaluation modules, scripts and domain tests; Frontend Developer owns frontend/** against docs/API_CONTRACT.md. Root owns API/storage/repo configuration, documentation and integration. No worker edits another surface. Read-only Model QA Specialist reviews evaluation after real training; design-reviewer reviews selected reference versus browser captures. No recursive delegation or remote publication.
