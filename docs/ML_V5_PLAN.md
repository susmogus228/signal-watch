# V5 complementary-detector study

User request: continue correcting SignalWatch toward at least 25/28 event detections while reporting and reducing false alerts. Keep actual ML, existing sensor data and onset-based metrics. Prior V4 remains active until selection, audit and integration checks finish.

## Development evidence and sources

Normal-only V2 detects 20/21 development `other` segments and 0/8 valve segments. Weighted supervised V4 detects 17/21 `other` and 8/8 valve segments. This development complementarity motivates a bounded fusion study. We do not select the design using individual retrospective failures.

An exploratory calculation on the already saved scalar OOF arrays, using both existing thresholds, yields 28/29 events, 3 false starts and 37 normal-alert seconds. This is design feedback from reused development predictions, not new validation. The bounded aggregation study and its exact manifest are frozen separately before its candidate scoring.

[Ymir](https://arxiv.org/abs/2112.04704) studies supervised use of complementary anomaly representations. [M2AD](https://proceedings.mlr.press/v258/alnegheimish25a.html) studies aggregation and calibration of multivariate prediction residuals. These are conceptual references; this study does not reproduce their models or compare our event counts with their reported scores. [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html) documents the need for disjoint fitting and calibration data. We retain the existing normal calibration anchor and do not mix a new calibration architecture into this experiment.

## Frozen scope before scoring

Keep the same twelve development recordings, three grouped folds, source hashes and gap handling. Keep both selected branch configurations: V2 normal-only window=2/leaf=12/64 trees, and V4 weighted HGB with its frozen classifier threshold. V4 training weights and preprocessing are unchanged. Fit/scaling use training recordings only; the residual scale uses the original separate normal calibration anchor. Inference uses no labels, IDs, timestamps or row indices.

Evaluate four candidates: classifier-only fallback, classifier OR maximum channel residual, classifier OR mean of the two largest channel residuals, and classifier OR mean of all eight channel residuals. Residuals are actual normalized forecast errors. Reconstruct fold residuals and verify their maxima against saved V2 OOF scores; verify classifier OOF scores against saved V4 scores.

For each residual aggregation, evaluate the unchanged 101 normal-OOF quantiles from 0.90 to 1.00 plus nextafter(max,+inf), retaining finite positive thresholds. The classifier threshold stays fixed. The classifier-only candidate explicitly disables the residual branch. Fused score is max(classifier_score/classifier_threshold, residual_score/residual_threshold), with alert threshold 1. Both branches require two history observations. Require both scores finite; do not improve availability by substituting missing predictions.

Compute full fused time-series onset metrics. Do not add hit counts from separate models: an earlier residual alarm can suppress a later classifier onset, and that must count as a miss. No point adjustment, smoothing, hysteresis, cooldown, periodic resets or label-dependent inference. Baselines remain unchanged.

Feasible development points must have normal-alert fraction <=5% and false onsets <=2/normal hour. Rank by maximum detected segments, then lower normal-alert fraction, false-onset rate, candidate order and threshold order. Put classifier-only first so exact ties preserve it. If no point improves this rank over classifier-only, retain V4 and do not inspect another retrospective audit. Freeze exact folds, source/dependency hashes, parameters, aggregation rules, thresholds and ranking in plan.json before scoring.

## Selection, audit and interpretation

Only the selected development winner may be refit and audited once on the previously examined 28 annotations. Save per-recording, family, exposure, availability and common-baseline evidence. Do not choose fusion variants or thresholds by retrospective results. Repeated development selection remains optimistic; the retrospective set is not an untouched generalization test. The 25/28 target is not guaranteed.

Fused scores express relative alert levels, not calibrated fault probabilities. Sensor reference deviations remain observed context, not model attribution or physical diagnosis. If displayed, branch evidence identifies which detector crossed its threshold and does not establish a physical cause.

## Memory and agents

Memory: Codex memory for scope/verification decisions; fresh Codebase Memory list has no SignalWatch project, so source fallback without indexing. Actual root is D:/Projects/signal-watch.

Worker: AI Engineer owns new ml_hybrid.py, its tests, study CLI, hybrid-development artifacts and work/hybrid-diagnosis. Input: frozen branch configurations, source helpers and approved protocol; output: trained selected bundle, OOF traces and complete report. Root owns active artifacts, API integration, retrospective audit, UI and documentation. Model QA Specialist independently reviews leakage, onset integrity, selection and artifact reconciliation without editing files. No concurrent edits to shared surfaces.

## Verification and constraints

Use tests first for score fusion, onset suppression, warm-up/gaps, future-prefix invariance, label independence, fold isolation and unchanged baselines. Verify actual OOF scores and selected metrics, then integrate/reproduce through the existing staged local training workflow. Inspect affected UI metadata and desktop/mobile states. Preserve V1 through V4 evidence. No new packages, external inference, private-data collection, public deployment or Git push.


## Observed outcome

The frozen maximum-residual fusion wins development selection with 28/29 hits, 3 false onsets and 37 normal-alert seconds / 9,179 normal seconds. The selected model then detects 26/28 retrospective segments, meeting 25/28. It raises 56 false starts / 18,208 normal seconds (11.072/h), exceeding the 2/h development limit, with 272 normal-alert seconds (1.494%). Median detected delay is 18 seconds. Activation follows the frozen development rule; it is not an operational-readiness claim. The API reproduction job completed and reproduced all saved metrics/events before replacement.

The API/UI explicitly declare preloaded complete-record cadence. Prefix causality is conditional on that known configuration; unknown raw timestamp initialization was not established. Scientific sources and audit metrics remain frozen. No external publication occurred.
