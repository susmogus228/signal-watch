# SignalWatch: research and interface brief

Research date: 2026-10-23. This archived brief records the concept stage before implementation. The user then selected option 2 (black). Current implementation and measured results: [verification](VERIFICATION.md) and [evaluation](ML_EVALUATION.md). Statements below about work not yet performed refer to the original concept stage.

## User request and acceptance

Research existing solutions, define a useful difference, and show three distinct English-language interface concepts before implementation. Retain the portfolio's requirement for actual trained ML and honest evidence. The user chooses the interface before application code is created. Existing projects, banners and deployments are outside this task.

Deliverables: this source-backed brief, three independently generated desktop UI concepts, their exact prompts and original PNG files. Mock graphs and numbers must be visibly labeled illustrative. Do not treat a mockup as a measured result.

## Problem and intended user

A developer, student or test engineer has a multivariate sensor recording and needs to decide whether a learned detector offers enough benefit over a simple statistical detector. The interface supports recorded-data replay and experiment review. It does not operate equipment or send production notifications.

## Analog comparison

| Source | Confirmed strength | Design consequence |
|---|---|---|
| [Datadog Anomaly Monitor](https://docs.datadoghq.com/monitors/types/anomaly/) | Trend/seasonality-aware anomaly bands, historical and evaluation previews, trigger/recovery windows | Adopt contextual traces and explicit alert conditions. Do not present these established features as our invention. |
| [Grafana Alerting](https://grafana.com/docs/grafana/latest/alerting/monitor-status/) | Alert state overview and history | Adopt explicit states, timestamps and event inspection. Avoid a broad infrastructure dashboard for a local experiment tool. |
| [Telemanom](https://github.com/khundman/telemanom) | LSTM prediction-error detection with nonparametric thresholding on spacecraft telemetry | Compare learned forecasting to statistical baselines; do not claim prediction-error detection is new. |
| [TimeEval](https://github.com/TimeEval/TimeEval) | Reproducible anomaly-detector benchmarking and metrics | Preserve data/split/model/config provenance. A web interface alone is not research novelty. |
| [ESA-ADB](https://github.com/kplabs-pl/ESA-ADB) | Operational evaluation scenarios, real-time monitoring simulation and annotated satellite telemetry | Replay and operator-oriented evaluation already exist. Our proposal is a smaller local, inspectable workflow, not a new algorithm. The full ESA pipeline is too heavy for the proposed first release. |
| [NAB](https://github.com/numenta/NAB) | Time-sensitive detection scoring and false-positive trade-offs | Report latency and false alerts alongside event detection; do not reduce performance to a single accuracy number. |

## Proposed distinction

One compact local application answers: does learned anomaly detection help on this recording at a stated alert burden?

- Replay the same unseen run through a learned predictor and a rolling robust-statistics baseline using only past observations.
- Calibrate each method on separate validation/calibration material against the same empirical false-alert target. Final-test achieved rates may differ; show them explicitly.
- Group repeated alert points into defined alert episodes and link every hit, missed event and false episode back to its trace.
- Separate known experiment annotations from detector output. Reveal labels for retrospective audit, never feed them into the live scoring path.
- Show when the baseline wins. The expected result is a reproducible comparison, not a promised ML victory.
- Distinguish frozen evaluation from exploratory changes. Changing thresholds must not silently update the official benchmark.

This is a product and evaluation integration proposal. No claim of first-of-its-kind functionality, causal fault diagnosis, production reliability or superior performance is supported yet.

## Dataset choice

Primary candidate: [SKAB](https://github.com/waico/SKAB), a physical water-circulation testbed. The current GitHub tree contains 35 CSV files, including one anomaly-free recording; the independent read-only review counted 4,392,581 bytes across those CSV files. Eight measured sensor columns have units, with datetime, anomaly and changepoint fields in experiment files. The implementation must still reconstruct rows, missingness, sample cadence, event intervals and run independence from the selected snapshot.

[Experiment descriptions](https://github.com/waico/SKAB/blob/master/data/README.md) include valve manipulations and other disturbances. These are recorded experiment context, not causes inferred by our model. A high anomaly score must not be labeled a failure probability.

License boundary: the [GitHub source license](https://github.com/waico/SKAB/blob/master/LICENSE) is GPL-3.0, while a Kaggle mirror reports different terms. Use one pinned official GitHub snapshot, preserve attribution/notices, and document data/resource conditions before redistribution. Do not silently declare bundled SKAB resources MIT. No dataset or upstream algorithm source was copied into the app in this concept stage.

NASA SMAP/MSL is optional follow-up material, not the primary evaluation basis. Telemanom states that the released values were normalized using test-set extrema and timestamps anonymized. The independent review also found duplicate P-2 entries in its label CSV. These require a dedicated audit; do not call replay of that bundle strict raw online evaluation or report real elapsed hours without verified timing.

## Proposed ML and evaluation

1. Fit scaling and a CPU-friendly ExtraTrees next-observation predictor on a training portion of normal data. Build lag/rolling features from history only; predict the next values across the eight sensors. Use normalized prediction residuals as anomaly scores.
2. Compare persistence and rolling median/MAD baselines. Optional neural models require later evidence that this extra complexity helps.
3. Keep chronological train/calibration/normal-audit blocks with a gap at least as long as the feature window. Reserve whole disturbed experiments for validation and final test; never randomly split overlapping windows across partitions.
4. Freeze a small candidate grid, selected configuration, calibration thresholds, alert grouping and cooldown before final evaluation. Reset state between recordings. Verify no centered rolling, future interpolation, backward filling or test-fitted scaling.
5. Primary metrics: detected and missed events, delay to the first actual post-onset alert, false episodes divided by observed normal duration, and fraction of time in alert. Show numerators, durations and sample counts. Use physical time only after cadence/gaps are verified.
6. Report ordinary point-wise PR-AUC as a secondary view. Avoid point adjustment that retrospectively fills an entire labeled interval after one hit. [Kim et al., AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20680) documents inflated performance from this protocol.
7. CPU time, memory, missing-data behavior and run-level uncertainty are part of the comparison. Small single-testbed data cannot validate transfer to unseen equipment.

## Proposed implementation boundary after visual selection

Target root: D:/Projects/signal-watch/. Proposed stack: Python 3.11+, scikit-learn for reproducible CPU training, FastAPI with OpenAPI for local inference/replay/evaluation, React and TypeScript for the English web application, and Apache ECharts for linked plots. Store local experiment metadata in SQLite and exported result manifests in JSON. Versions and exact APIs must be checked against official documentation/Context7 during implementation, before dependency selection or code.

Proposed API resources: dataset/schema validation, training job and model artifact, replay session and next chunk, recorded alert episodes, frozen evaluation, exploratory comparison, and export/delete. Prefer same-origin local REST and bounded chunks/polling first; no external inference API is required. CSV import should map timestamp and numeric sensor fields, keep labels optional, and suppress unavailable quality metrics when labels are absent. Uploaded files stay local; storage/deletion behavior must be explicit in the implementation brief.

Core journey: choose bundled recording or validate a local CSV; select a saved model; replay with pause/step/speed controls; inspect an alert; compare methods; open audit labels and recorded metrics; export the result. Data gaps, warm-up, no alerts, model unavailable, invalid CSV and failed jobs need clear states. No GitHub Pages hosting claim for a Python backend.

## Three visual directions

Each concept is one 1440 x 1024 desktop web-app screen, not a landing page. Three independently generated images, English copy, natural viewport, no browser chrome. Use readable 14-16px body text, an 8px spacing rhythm, maximum two font families, small radii and restrained borders. No gradients/glass effects, promotional hero, fake 3D equipment or performance promises. Values and charts are illustrative and visibly labeled. Use elapsed recording time instead of inventing measurement dates. Current design-review date anchor: 2026-10-23.

- Replay Desk: warm-white/cobalt, signal-first layout with linked traces and a compact alert inspector. Emphasizes understanding the streaming decision.
- Event Review: charcoal/amber, event-first layout with an episode queue and one focused detail canvas. Emphasizes separating repeated alerts, misses and retrospective annotations.
- Alert Budget Lab: light ink/teal, comparison-first layout showing the validation-only trade-off between alert burden and missed events. Frozen audit results remain read-only.

These are alternative entry points into the same bounded product; they do not imply building three separate applications or every visible item as a new feature.

## Visual references and use decisions

| Reference | Evidence | Decision |
|---|---|---|
| Datadog official monitor documentation screenshot | Captured in the in-app browser; documentation, not a tested authenticated app | Adopt historical/evaluation separation and restrained chart context; transform layout and brand. |
| Lazyweb Sentry alerts screen, screens:efbf243f1f6969b4a12a3bea | Stock product screenshot inspected directly, including its empty state | Adopt quiet separators and a clearly described empty state; reject large unused chart area and excess navigation. |
| Lazyweb Better Stack incident screen, screens:ff61b46a1cbb27a3d38c194e | Stock product screenshot inspected directly | Adopt event-to-trace drill-down; reject the crowded filter stack and broad monitoring taxonomy for this small app. |

Reference screenshots under references/ are private research material, not new SignalWatch assets or proof that we tested those products. Do not publish third-party screenshots automatically. Stable Lazyweb search identifier: ad84e942-6241-4eca-9235-42ac56686f7c. Saved Product Design context was absent; the current user request and project rules supply the brief.

## Independent review and verification status

A read-only Model QA Specialist independently reviewed primary-source feasibility, leakage, metrics and dataset conditions. Its findings changed the preferred dataset from SMAP/MSL to SKAB and narrowed the novelty claim. No model results exist yet. The final concepts are only visual design artifacts, and no desktop/mobile application behavior has been verified.

Memory and agents: prior chat requirements and source-backed publication/evaluation conventions supply context; no new code graph exists. Root owns all brief and visual artifact files. Model QA Specialist owns an independent read-only feasibility verdict, with no edits or downloads of datasets/models. No implementation or publication agent is needed before user selection.
