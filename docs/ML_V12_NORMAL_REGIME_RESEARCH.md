# V12: learned normal regimes and progressive-defect detection

Date: 2026-10-27. Outcome: **no feasible candidate**. The approved bounded experiment is complete; the requested joint model-quality improvement is still unresolved. V5 remains active.

## Actual trained pipeline

The [frozen protocol](ML_V12_NORMAL_REGIME_PLAN.md) adds a learned normal representation before supervised classification. For each whole-recording development fold, a diagonal Gaussian mixture learns only labelled-normal training sensor vectors. Every training recording contributes exactly 512 selected observations, including the normal anchor: 4,608 bank rows per fold. Bank sampling, scaling and mixture fitting exclude held-out records.

Twelve density/distance/residual features join the unchanged 186 causal trend features. The same V10 B HGB configuration then learns from 198 columns with the original recording/class-balanced weights. K1/K2/K4 test one, two or four normal components. Inference uses fixed fitted parameters and an immediate classifier cutoff. No alarm delay, smoothing, live adaptation or new dependency is added.

All nine mixtures converged and all nine HGB fits completed: **18 top-level estimator fit calls**, with nine internal mixture KMeans initializations recorded separately. No arm was invalidated, no fit retried and no final pipeline fitted. The study took 247.300 seconds. All three 102-point scans are preserved, for 306 candidate operating points. The conditional two final fit calls were not used.

## Detection and false alerts

Acceptance requires at least 26/29 development events, fewer than three false starts, at most 2/h, at most 37 normal-alert seconds over 9,179 normal seconds, and retention of both source-described progressive intervals. Those intervals are linear other-6 and exponential other-9 under the original annotations.

| Descriptive point with both progressive intervals and at least 26 total hits | Events hit | False starts | Normal-alert seconds |
|---|---:|---:|---:|
| V5 development control | 28/29 | 3 | 37 |
| K1 at 0.9718653604686452 | 27/29 | 5 | 37 |
| K2 at 0.9555687264953208 | 27/29 | 6 | 37 |
| K4 at 0.8571184329230667 | 27/29 | 40 | 176 |

Each K row shows its smallest scanned false-start count under those event conditions. These are diagnostic frontier points, not selected replacements. All fail strict false-count improvement; K2/K4 also exceed the 2/h rate, and K4 exceeds normal-alert time.

Under the count/rate/time burden limits, K1 and K2 can reach 26/29 with one false start and 28 normal-alert seconds, but both lose the linear other-6 event. K4 reaches only 9/29 in that region and also loses it. Every one of the 306 points fails at least one primary condition, so selected=null. No alternative point or annotation convention is inserted afterward.

## Probability diagnostics and transfer limits

All rows below use the same 13,425 scored development observations. Higher AP and lower Brier/log loss are favorable, but they do not establish useful alarm behavior or physical-fault probabilities.

| Model | Pooled AP | Brier loss | Log loss |
|---|---:|---:|---:|
| V10 B, original trend classifier | 0.911362 | 0.092064 | 0.414039 |
| K1 | 0.911362 | 0.092064 | 0.414039 |
| K2 | 0.914870 | 0.090095 | 0.390281 |
| K4 | 0.901297 | 0.092949 | 0.440574 |

K1's saved OOF probabilities are exactly equal to V10 B in all twelve recordings, including availability. We do not infer which fitted tree splits used the appended features: outer classifier objects were not retained for that inspection. K2 slightly improves probability/ranking diagnostics but fails the joint alarm goal. The tested four-component configuration worsens these diagnostics and the relevant false-alert frontier. A numerical-conditioning finding below prevents attributing that degradation only to component count.

The predeclared descriptive density summaries expose a normal-representation limitation. In fold 0, K2's largest training-normal negative log density is 11.761, while labelled-normal other-3 has median 251.840 and maximum 271.736. K4's corresponding values are 11.868, 326.936 and 344.331. The transform therefore treats those held-out benchmark-normal observations as far from its training normal bank. Numerical convergence does not establish coverage of unseen normal records. These post-hoc observations do not prove a physical operating-mode change, a dangerous condition, label error or the cause of every classifier false start. They do not alter fitting or selection.

## Numerical-conditioning finding

Independent QA found a weakness in the chosen scaling rule: Pressure's normal-bank IQR is zero in folds 1/2, so its scale becomes 1e-6. Standardized magnitudes reach 1,639,639; other sensor channels stay below 3.45. This extreme range should have been checked before fitting. The V12 protocol and source remain frozen as evidence of that mistake; it affects a rejected research candidate, not active V5.

K4's minimum fitted Pressure variances are 0.0008779296875 and 0.0008931884765625, despite reg_covar=0.001. They are positive and satisfy the literal protocol, but indicate cancellation in the installed covariance calculation, which subtracts squared means from second moments. The installed diagonal Gaussian density also uses an expanded quadratic. QA reproduced the saved statistics with that exact arithmetic, then compared a centered-residual Gaussian calculation at the same saved parameters, without new classifier predictions.

For K4 folds 1/2, maximum held-out-normal absolute negative-log-density differences are 0.039292/0.045301; 95th percentiles are 0.013546/0.015750. Training-normal maxima are 0.045355/0.049531. NLL feature center/IQR values differ by up to 0.001539/0.005717. K1/K2 and K4 fold 0 agree below absolute 1e-12. The earlier fold-0 other-3 density mismatch therefore persists independently of this finding.

This fixed-parameter comparison does not quantify numerical error in the complete EM fitting trajectory or its contribution to forty false starts. Neither a corrected classifier nor a new operating point was fitted/scored. No result, label, parameter or saved OOF array was replaced. A future scaling correction needs its own frozen experiment and validation; the current run does not establish that correction's quality.

## Decision and verified boundary

Keep the negative result and active V5. No final estimator artifact, old-28 regression, reused-teaser inference, API activation or publication followed this study. The real final-model diagnostic path remains unexercised; its identity/availability guards have synthetic regression coverage.

The full backend suite passed **176 tests**, with seven existing warnings, in 69.30 seconds before the sole study run. Source/dependency hashes remained fixed during training. The warnings concern the existing Starlette/httpx deprecation and earlier calibration tests; V12 mixture fits emitted no convergence warnings. Software tests and converged fitting do not close R17/R20 model quality.

Independent saved-array/provenance QA is recorded in the [verification evidence](evidence/v12-normal-regime-check.json). It must be read separately from the quality rejection. The [version history](RESEARCH_LOG.md) retains V10/V11 and earlier failures. All comparisons reuse development records; no new independent-validation or operational-reliability claim follows.

## Artifacts and reproduction

- [Frozen plan](../artifacts/regime-development/plan.json) and [execution marker](../artifacts/regime-development/execution_started.json)
- [Complete report](../artifacts/regime-development/report.json), [all 306 candidate audits](../artifacts/regime-development/candidate_audits.json) and [pre-final selection](../artifacts/regime-development/selection_before_final_fit.json)
- [K1 OOF](../artifacts/regime-development/K1_oof_scores.npz), [K2 OOF](../artifacts/regime-development/K2_oof_scores.npz), [K4 OOF](../artifacts/regime-development/K4_oof_scores.npz)
- [K1 fit/density evidence](../artifacts/regime-development/K1_fit_evidence.json), [K2 evidence](../artifacts/regime-development/K2_fit_evidence.json), [K4 evidence](../artifacts/regime-development/K4_fit_evidence.json)

Report SHA-256: b3ee083b9d47a864838bf8f21603bd266fc473210624f02f53371e6878fcb327. Plan SHA-256: 0f81f004fe8acc4308b3398ff12c07d9844cf7a9d67e0d9aeea61e6356b58a3e. Candidate-table SHA-256: e33f2e72508c7f4ea5ea6c3a6bb271d7fd090878918ce70a6d66968df18e4e25.

scripts/study_regimes.py refuses to overwrite or repeat this execution. Reproduction requires an isolated copy with the pinned inputs, including the preserved ignored V10/V11 research controls. The source papers, inspected-content limits and exact-version API checks appear in the approved proposal and RESOURCES.md. No paper implementation or benchmark gains are claimed.
