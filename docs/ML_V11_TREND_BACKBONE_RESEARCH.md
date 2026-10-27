# V11: learner comparison on causal trend features

Date: 2026-10-27. Outcome: **no feasible candidate**. Six development fits completed in 166.185 seconds. No final model was fitted, no old-audit or teaser inference occurred, and V5 remains active.

## Change and controls

The [frozen protocol](ML_V11_TREND_BACKBONE_PLAN.md) compares deeper histogram gradient boosting (HGB31) and a 256-tree ExtraTrees classifier on V10's 186 causal features. The features include actual 30/120-second trends, partial-window coverage and the original 120 classifier features. Preprocessing, recording/class weights, grouped folds and immediate alarm decisions stay fixed. No hard-normal mining, temporal confirmation, smoothing or score calibration is added.

HGB31 changes leaf capacity from 15 to 31 relative to V10 B. ExtraTrees changes the learning algorithm. Each learner has three whole-recording held-out folds, with preprocessing fitted only on the corresponding training records. The normal anchor is fit-only. Saved V10 B, original 120-feature classifier and V5 fusion controls are reused without refitting.

## Scores improve, the alarm requirement does not

These classifier-only diagnostics use the same 13,425 scored development rows. AP measures ranking; Brier and log loss measure probability forecast error and do not establish calibration alone. These scores are not physical-failure probabilities.

| Classifier | Pooled AP, higher is better | Brier loss, lower is better | Log loss, lower is better |
|---|---:|---:|---:|
| Original A, 120 features | 0.887159 | 0.096255 | 0.481249 |
| V10 B, 186 features | 0.911362 | 0.092064 | 0.414039 |
| HGB31, 186 features | 0.925149 | 0.087186 | 0.400872 |
| ExtraTrees256, 186 features | 0.921981 | 0.088131 | 0.304504 |

All 204 predeclared operating points failed at least one primary requirement: at least 26/29 events, fewer than three false starts, at most two false starts per normal hour, at most 37 normal-alert seconds, and retention of both source-described progressive intervals. Exposure is 9,179 observed normal seconds. A new alert must actually start inside an annotated interval to count as a hit.

| Descriptive point from the frozen scan | Events hit | Progressive intervals retained | False starts | Normal-alert seconds |
|---|---:|---:|---:|---:|
| Active V5 development control | 28/29 | 2/2 | 3 | 37 |
| HGB31 at 0.9830233958441359 | 27/29 | 2/2 | 5 | 38 |
| ExtraTrees256 at 0.5637885882669705 | 27/29 | 2/2 | 11 | 138 |
| Best HGB31 recall under the count/time burden limits | 15/29 | Linear interval lost | 2 | 9 |
| Best ExtraTrees256 recall under those burden limits | 7/29 | Linear interval lost | 0 | 0 |

The two progressive-preserving rows show the smallest false-start count among each learner's scanned points with at least 26 hits. They are explanatory frontier points, not replacement selections. HGB31's five starts equal 1.961/h and satisfy the rate limit, but fail the stricter improvement over V5's three development starts; 38 seconds also exceeds 37. ExtraTrees fails both count/rate and duration limits. Lowering a cutoff can recover events while adding many more false starts, so maximum recall alone is not a useful success claim.

## Interpretation and next hypothesis

V10 already showed a shared-cutoff problem: normal scores in held-out valve records can exceed every anomalous score in the linear-imbalance record. V11 improves aggregate scores without resolving that joint constraint. More capacity and a different tree learner did not solve the tested transfer problem. This does not prove that every possible tree model must fail or that all false episodes are caused by operating-mode changes.

The [next proposal](ML_V12_NORMAL_REGIME_PROPOSAL.md) adds a learned representation of labelled normal regimes before classification. The user approved this architecture on 2026-10-27; its separate [protocol](ML_V12_NORMAL_REGIME_PLAN.md) governs implementation and execution. Scientific work on multimode monitoring motivates that hypothesis; it does not establish the cause of our remaining errors or promise a gain.

## Verification and evidence

Independent Model QA reconstructed all 204 candidates from saved OOF arrays, including metric aggregates, recording/family slices, event pairing, progressive outcomes and rejection reasons: zero mismatches. Pooled/macro/per-record probability diagnostics agree within absolute 1e-12. Fit provenance, all recording/class weight masses, train/held-out exclusions and all 186 scaler entries match the declared design. All 29 frozen dependencies and six active V5 files remain unchanged.

The full backend suite at completion of V10/V11 passes **160 tests**, with seven existing warnings, in 54.26 seconds, before the later V12 source extensions. ExtraTrees same-input score invariance uses absolute tolerance 1e-12 for parallel float64 summation; decisions still use strict raw comparisons without rounding. Synthetic tests cover the conditional final-model identity guards, but real final-model diagnostics were not exercised because neither study qualified.

- [Frozen study plan](../artifacts/trend-backbones-development/plan.json)
- [Execution marker](../artifacts/trend-backbones-development/execution_started.json)
- [Complete development report](../artifacts/trend-backbones-development/report.json)
- [All 204 candidate audits](../artifacts/trend-backbones-development/candidate_audits.json)
- [HGB31 OOF scores](../artifacts/trend-backbones-development/HGB31_oof_scores.npz)
- [ExtraTrees256 OOF scores](../artifacts/trend-backbones-development/ExtraTrees256_oof_scores.npz)
- [Selection before conditional final fit](../artifacts/trend-backbones-development/selection_before_final_fit.json)
- [Combined verification evidence](evidence/v10-v11-learned-model-check.json)

Report SHA-256: 9d12e0f90f8cf4164e74b259e9130a437b1081c9209fc8f892e44d819cec244e. Candidate-table SHA-256: 1124ec9677b1d99d3314c6d9c5dc9fdba670183de6794964929115577f93fc88.

scripts/study_trend_backbones.py refuses to repeat a completed frozen execution. Reproduction requires an isolated copy with pinned inputs and preserved ignored research artifacts. Reused development evidence is not fresh validation; no new release, UI state, deployment or field reliability is established. R17 and R20 quality goals remain open.
