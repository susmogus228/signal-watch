# V13: Pressure scaling correction and unchanged alarm-quality screen

Date: 2026-10-27. User explicitly requested correcting Pressure scaling, then checking false alerts and gradual-defect retention. This authorizes the existing GMM/HGB architecture and one bounded local experiment. Earlier frozen studies and active V5 remain unchanged.

## Single intervention

Preserve V12's normal-bank selection, bank medians, original three recording folds, twelve development records, original normal anchor [0,5172), labels, two-row eligibility, causal trend features and K1/K2/K4 learners. Preserve all estimator parameters, classifier preprocessing rules, weights, seeds, cutoffs and event definitions from the [V12 protocol](ML_V12_NORMAL_REGIME_PLAN.md).

For the Pressure sensor only, set its bank scale to max(training-bank IQR, training-bank population standard deviation, 1e-6). Compute standard deviation in float64 with ddof=0 from the identical balanced normal bank. Keep the other seven bank scales exactly max(IQR,1e-6), and keep all bank centers exactly median. Identify Pressure by the existing sensor key, assert its single correct column, and record the original scale, IQR, standard deviation, new scale and changed-column indices. A constant Pressure channel keeps the numerical floor; zero IQR alone does not imply constant observations.

The change limits extreme standardized Pressure values. It also changes effective Pressure covariance regularization in physical units: reg_covar * bank_scale^2. The experiment therefore tests a scaling/regularization intervention, not an isolated replacement of floating-point arithmetic. Do not change sklearn's density formula, covariance type, reg_covar, the other sensors or the twelve appended-feature scaling rules.

## Input preflight and numerical audit

Construct the exact V12 bank from eligible training-normal rows: equal m=min(512,min eligible normal count) per recording, indices I_r[floor(linspace(0,n_r-1,m))], no replacement. Exclude held-out records from bank fitting and scaling. Validate finite/positive scales and require maximum absolute standardized Pressure <=128 on that fit bank before calling either estimator. This bound exceeds sqrt(6656)+1 for the largest 13-record final bank; it is a fixed numerical sanity bound, not a learned alarm threshold. Invalid bank scaling invalidates the whole K arm before fitting and is reported without retries. Held-out ranges are diagnostic only; no clipping, per-test normalization or exclusion follows them.

Record maximum transformed magnitudes on bank, classifier-training rows and held-out rows, with per-sensor values. Keep the original 186 classifier center/scale entries exactly equal to preserved V10 B for matching folds. Keep the twelve appended columns' median/IQR/floor preprocessing rule unchanged, although their fitted values may change after the mixture changes.

For each fitted mixture, preserve weights, means, variances, convergence/iterations and warnings. At the same fixed parameters, compare installed expanded-quadratic negative log density with a centered-residual Gaussian calculation using stable log-sum-exp. Handle zero component weights as negative-infinite log weight without adding pseudo-mass. Save count/max/median/95th-percentile absolute differences and minimum variance relative to reg_covar. Audit the bank, all eligible classifier-training rows and all eligible held-out rows; preserve summaries on normal rows separately. Reuse computed features where practical and obtain no extra classifier predictions for the audit.

Predeclare numerical acceptance flags: finite audit values, maximum absolute NLL difference <=1e-7 and all fitted variances >=reg_covar-1e-9. These tolerances concern float64 conditioning; they do not round probabilities or change strict alarm comparisons. Report flags for every arm/fold. Keep them separate from the original quality screen and rank. A quality winner with failed numerical flags cannot authorize a final fit or diagnostics: preserve development_selected, set selected=null/status='numerical_check_rejected', and do not choose another arm. Apply the same fixed-parameter checks to the conditional final fit on its training rows; a failed final check rejects it without retry or model artifact. Do not tune tolerances after outcomes.

## Learning, controls and budget

Use exactly V12's GaussianMixture: K=1/2/4, diag covariance, reg_covar=0.001, tol=0.001, max_iter=200, n_init=1, init_params='kmeans', random_state=20261023, warm_start=False. Use V10 B's HGB150/15 leaves/min_leaf20/learning_rate0.05/L2=1/early_stoppingFalse/seed20261023 and equal-record/class-balanced supervised weights. No hard-normal mining, calibration, larger learner, delay rule, smoothing, cooldown or residual OR.

Nine GMM plus nine HGB development fit calls maximum; internal KMeans initialization counted separately. Only one qualifying, numerically accepted winner permits one additional GMM/HGB pair: at most twenty top-level estimator fit calls, at most two numeric CPU threads. Synthetic unit-test fits remain separate. Any nonconverged/non-finite mixture invalidates its entire K arm, discards partial OOF selection and skips remaining folds; failed attempts count. Unexpected implementation exceptions stop the one-shot execution with its marker intact.

Load saved V12 K1/K2/K4 OOF and fit evidence as same-K controls, alongside saved V10 B, V11 and V5 controls. Do not refit them. Require identical IDs, source rows, bank selections, all seven unchanged bank scales/centers and common score availability. On any fold whose Pressure scale is unchanged, require same-K OOF probability equality to V12 within absolute1e-12/relative0 and exact finite masks as an integrity check; save the maximum difference. A discrepancy stops execution for investigation rather than selecting a different arm. Save new full-length OOF probabilities and all fit metadata. Raw classifier decisions remain immediate probability > positive cutoff.

## Quality and matched comparisons

Use unchanged onset_v2 and exactly 9,179 observed normal seconds. Scan V12's 101 normal-score quantiles linspace(0.90,1.0,101) plus nextafter(max normal,+infinity), deduplicating positive finite values: at most306 points. Require >=26/29 events, fewer than3false starts, <=2/h, <=37 normal-alert seconds and both original progressive intervals other-6-event-0/other-9-event-0 retained. Do not change labels, annotation boundaries, reset rules, warm-up or exposure.

Rank feasible points by more events, fewer false starts, fewer normal-alert seconds, lower median/mean detected-event delay, then K1/K2/K4 and cutoff order. Freeze the exact pre-final point/OOF/candidate/plan identities before a final fit. If none qualifies, stop with selected=null and no final model or old-audit/teaser inference.

Additionally report the following descriptive comparisons against saved V12 using the same predeclared rule for both studies:

- Among points retaining both progressive intervals and >=26 total hits, choose smallest false count, then normal-alert time, most hits, delay and declared cutoff order. Report the complete metrics, per-record/family false starts and paired progressive outcomes. This is an explanatory frontier, not another selection rule.
- Among points satisfying the count/rate/time burden limits, report maximum recall and whether either progressive interval is lost, with deterministic ties by false count/time, delay and cutoff order.
- Report pooled/per-record probability metrics separately. A better AP or lower loss does not establish fewer alarms. Any raw-score/cutoff equality or non-equality is descriptive; do not infer unobserved tree feature usage.

Partial improvement must remain visible even when the full joint screen fails. Thresholds selected for each representation also change the operating point; do not attribute every alarm difference solely to floating-point conditioning.

## Artifacts, conditional diagnostics and ownership

AI Engineer owns only signalwatch/ml_pressure_scaled.py, tests/test_ml_pressure_scaled.py, scripts/study_pressure_scaled.py and artifacts/pressure-scaled-development/. Root owns this protocol, shared docs/evidence and any new conditional diagnostic runner/tests/output. Existing trend_diagnostics.py and its tests/script must not be edited or monkeypatched; explicitly include them in the V13 freeze in addition to the prior V12 dependencies. Reuse unchanged pure helpers where possible; do not refactor protected scientific files.

Preserve plan, execution marker, all per-arm OOF arrays/fit evidence, report, candidate_audits.json and selection_before_final_fit.json. Use the shared selection['oof_hashes'] contract. A sole final bundle has 198 feature names, arm K1/K2/K4, threshold, plan_sha256 and selection_before_final_fit_sha256; classifier_scores returns the existing dict['scores'] contract with the original NaN mask. Failed final convergence/numerics retains development_selected but selected=null, rejection reason/counters and no model artifact/SHA.

If a sole final winner exists, root prepares a separate frozen new runner and identity tests, obtains independent QA, then scores that model once on each old-audit record and once on the already-inspected teaser. No alternative model/cutoff or annotation correction follows those outcomes. Both remain reused diagnostics, not fresh validation. Preserve active V5 during this study; release integration is a separate check.

Freeze this protocol, new source/tests/script, all V12 scientific dependencies, complete V12 artifacts and prior scientific controls/data before the sole run. Preserve the active six V5 hashes. Independent Model QA checks code/actual freeze before fits, then reconstructs saved metrics, normal banks/scales and numerical audit without model fitting or new classifier inference. Root runs one final full backend suite after source completion and before the real study. New conditional runner code, if needed later, requires its own targeted/full checks.

## Sources and scientific limit

The [related-work review](RELATED_WORK_COMPARISON.md) and [V12 numerical finding](ML_V12_NORMAL_REGIME_RESEARCH.md) motivate this correction. Context7's scikit-learn1.9.1 preprocessing/common-pitfalls guidance confirms training-only scaling. Official StandardScaler documentation describes population standard deviation; our median/max-IQR-std rule is a project-specific safeguard, not StandardScaler itself. GaussianMixture's documented reg_covar acts in transformed coordinates. Some /1.9/ web extracts identify themselves as1.9.0; use the pinned1.9.1 source/docs and installed signatures for exact-version evidence. No dependency is upgraded or library patched. A numerically improved transform may still fail gradual detection or normal-mode transfer.
