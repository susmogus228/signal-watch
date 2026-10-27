# V10: learned trend representation and hard-normal weighting

Date: 2026-10-27. User instruction: improve the trained ML itself to detect gradual defects and reduce false alerts, rather than delaying alarms. Existing known-state labels, sensor schema and local Python/scikit-learn stack stay unchanged. No temporal confirmation, cooldown, score smoothing or decision backfill enters this study.

## Source-backed hypothesis

The existing HGB classifier uses 120 causal features, including means/std over 2/5/15/30 observations and deviations from a fixed initial two-row reference. It lacks explicit elapsed-time trend slope, direction consistency and linearity. The two-row innovation predictor reacts to short forecast excursions. The pinned data/SKAB-DATA-README.md describes development other/6 as linear rotor imbalance and other/9 as exponential rotor imbalance. These identify a predeclared progressive-behavior reporting subset, using their original anomaly labels; they do not create new gradual row labels or make every valve event gradual.

We test whether the classifier learns a useful separation between sustained changes and isolated normal excursions. A second arm emphasizes difficult normal training observations without changing labels. This is a hypothesis; the previously missed teaser intervention informed investigation and cannot serve as fresh independent validation.

## Controls and arms

- Product control: preserved V5 classifier/innovation OR fusion at its original cutoffs (28/29 development events, three false starts, 37 normal-alert seconds).
- Representation control A: preserved V5 120-feature classifier OOF alone at its original classifier cutoff, plus classifier-only probability/ranking diagnostics. A is not the full V5 fusion.
- B: newly fitted HGB with the same 120 base features plus 66 trend/coverage features; original equal-record/class-balanced fit weights.
- C: same 186 features, unweighted train-only preprocessing and HGB parameters as B; only training weights change using record-disjoint inner-OOF hard-normal mining.

B/A probability and ranking comparisons investigate the representation. C/B investigates weighting. Comparison of B or C with full V5 additionally removes the old hard OR rule, so it does not isolate the effect of features alone. New candidates produce a learned classifier score directly; the innovation predictor cannot force a candidate alarm against its learned decision. Retain the original fusion as a measured control.

## Exact causal representation

Keep the base feature builder and its two-row eligibility unchanged. Reset all history at existing gaps. For each eligible row t and each elapsed-time horizon W in [30,120] seconds, use only observations in [elapsed[t]-W,elapsed[t]), inside the current gap segment. Do not use the present value in these extra historical summaries; the unchanged base features already include the current observation. Partial history is allowed with no padding or added warm-up.

For each of eight channels compute four statistics: centered OLS slope per actual second; last-minus-first historical value; mean sign of successive historical increments (zero increments stay in the denominator); and OLS R-squared clamped to [0,1]. A constant series has slope/net/direction/R-squared zero. Fewer than two past observations give four zeros. Add one coverage value per horizon: min((last historical time-first historical time)/W,1), or zero with insufficient history. Total dimensions: 120+2*4*8+2 = **186**. Two-point R-squared may equal one; coverage discloses limited context to the learner.

No label, absolute clock, recording ID, annotation boundary or future observation enters features. Relative elapsed differences define windows and slope units. Prefix invariance is conditional on the project's fixed preloaded cadence/gap configuration; do not claim unknown-stream cadence initialization. Use finite float64 features and reject numerical overflow. Median/IQR preprocessing fits only the applicable model's training records, with the original 0.05*std and 1e-6 floors.

## Training and hard-normal mining

Use the original twelve development IDs and three whole-recording outer folds, verified against preserved manifests and actual CSV hashes. Each B fit uses the eight outer-training disturbed recordings and the unchanged fit-only normal anchor. Keep the original HGB parameters: 150 iterations, 15 leaves, minimum leaf20, learning rate0.05, L2=1, early_stopping=False, seed20261023, two CPU threads. No new dependency or larger estimator grid.

For each C outer fit, split only the eight outer-training disturbed IDs into three inner folds by sorted-ID index modulo3. Train an unmodified B classifier on inner-training IDs plus the normal anchor; fit its preprocessing there. Predict only its disjoint inner-held IDs. Outer-held IDs enter neither inner fitting, preprocessing, mining nor C fitting. Require both pooled training classes; fail rather than introducing test data or in-sample fallback.

For each disturbed outer-training record with n0 eligible label0 rows, mark exactly ceil(0.10*n0) highest inner-OOF class1 scores, with score descending then raw row index ascending for ties. These are hard normal observations, not necessarily misclassifications. Multiply their existing fit weight by3, then renormalize that recording's entire class0 weights to its original equal-record/class-balanced class0 mass. Keep its class1 mass unchanged. Never mine or upweight the always-fit normal anchor. Preserve all original labels, including ambiguous boundary normals. Save mining indices/scores, fit/mining provenance and per-record/class masses; report boundary proximity descriptively after selection, without excluding examples based on it.

Development uses exactly 15 HGB fits: B outer3, unmodified B inner-mining9, C outer3. Save every candidate OOF array and inner-mining array. No calibration or old-model probability adjustment substitutes for learning.

## Frozen development selection

Raw classifier predict_proba(class1) crosses a positive cutoff immediately; it is not a guaranteed physical-failure probability. Keep the same two-row/gap finite mask and score every original annotated interval; no masking creates a new onset. Use the unchanged onset_v2 evaluator and elapsed exposure.

For each B/C, scan the original normal-OOF quantiles linspace(0.90,1.0,101) plus nextafter(max normal,+infinity). Deduplicate finite positive thresholds; at most 204 points. Require >=26/29 events, fewer than three false episode starts, <=2 starts/normal hour, <=37 normal-alert seconds over exactly9,179 normal seconds, and retention of all original V5-detected intervals in other-6/other-9. The 26/29 floor is the existing ceil(29*25/28) project-target proxy. Rank feasible points by most events, fewer false starts, fewer normal-alert seconds, smaller median/mean detected-event delay, B before C and threshold order. Report all points and rejection reasons, per-record/family metrics, progressive-subset detections/delays and every lost/gained interval.

Record pooled/macro/per-record AP, Brier and log loss on classifier probabilities separately from operating-point results; constant/single-class subsets retain explicit denominators and undefined AP. These diagnostics do not alone establish alert quality. If no new point qualifies, keep V5, save a no-feasible report and perform no final-model refit or alternative held-out winner search in this protocol.

## One final candidate and reused diagnostic checks

If a candidate qualifies, freeze its identity/cutoff BEFORE any final fit or old-audit/teaser inference. Fit exactly that selected arm on all twelve development records plus the original normal anchor. B needs one final fit. C needs three full-development inner B mining fits and one C final fit. Thus the maximum whole study is 19 HGB fits. Freeze final artifact/source hashes and preserve unique versus repeated training-row counts.

Then run one fixed retrospective old-28 regression report and one reused-teaser diagnostic for this sole final candidate. Never compare alternative B/C final models or thresholds on these outcomes. The old28 corpus and teaser have both already been inspected and are NOT fresh/blind confirmation. The teaser uses the unchanged V9 frozen pivot, four source envelopes, uncertain transitions and assumed-background definitions. Report its gradual shaft intervention and background starts/duration alongside V5, without post-score unit changes or threshold adjustments.

The old benchmark checks whether the fixed candidate retains the user's >=25/28 target while reducing false burden; the intended rate remains<=2/h with no more than V5's272 normal-alert seconds. Report failures unchanged. The four-intervention teaser cannot prove field reliability or statistical superiority. Repeated development selection can overfit, and outcome-informed design weakens transfer claims even with a frozen final choice.

No active V5 replacement, API/frontend contract change, publication or global installation belongs to this research execution. A passing candidate requires a separately recorded release/integration check before replacing active artifacts. Preserve six active V5 files and all earlier frozen scientific sources throughout this study.

## Verification and outputs

TDD: analytic ramps versus isolated spikes, elapsed irregular cadence, constant channels, short partial windows, exact feature names/count, gap resets, prefix/label/ID/time-origin invariance, unchanged finite eligibility; mining record exclusion, stable top-count ties, anchor exclusion and record/class mass; fitted-estimator behavior and no-feasible guard. Independent Model QA checks the exact protocol/code/dependencies before fitting and reconstructs saved metrics/mining afterward. Run the full backend suite after changes; no unchanged frontend rebuild is required.

New files: signalwatch/ml_trends.py, tests/test_ml_trends.py, scripts/study_trends.py and artifacts/trend-development/. Root owns docs and any conditional final diagnostic runner/output in outputs/v10-diagnostics/. Preserve earlier ignored research inputs for local reproducibility; public packaging remains separate.

## Primary sources

[Canonical Interval Forest](https://arxiv.org/abs/2008.09172) and [catch22](https://link.springer.com/article/10.1007/s10618-019-00647-x) support evaluating interpretable interval representations; this study does not implement those classifiers or reproduce their benchmark gains. [Canevet and Fleuret, hard sample mining](https://proceedings.mlr.press/v39/canevet14a.pdf) motivates difficult-negative training; its object-detection results do not guarantee telemetry improvement, and this study implements explicit weighting rather than copying that mining algorithm. The installed scikit-learn1.9.1 [weighted HGB API](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/ensemble.rst) was checked through Context7. The pinned data/SKAB-DATA-README.md supplies the progressive-behavior experiment descriptions.
