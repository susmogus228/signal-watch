# V10 learned trend features and hard-normal training

Date: 2026-10-27. Outcome: **no feasible candidate** under the predeclared joint detection/burden/progressive-cohort conditions. We trained real new classifiers; this experiment uses no temporal confirmation or delayed alarm rule. V5 remains active.

## Actual model changes

The [frozen protocol](ML_V10_TREND_MODEL_PLAN.md) preserves the 120 original causal features and adds past-only 30/120-second slope, endpoint change, signed direction consistency, R-squared and history coverage. The total is 186 features, with the original two-row eligibility and gap resets. Partial history avoids a compulsory 120-second wait; it does not guarantee unchanged learned detection latency.

B fits the original 150-iteration/15-leaf HGB with this representation. C uses the same representation, preprocessing and HGB parameters but reallocates normal training weights toward the fixed top 10% high-score normal examples, identified through three whole-recording inner OOF folds. We multiply by 3 then restore each record/class's original weight mass. The fit-only normal anchor is never mined. These examples are suspicious to the learner, not necessarily misclassifications or physically verified safe operation.

The run made exactly 15 development HGB fits: three B, nine inner-mining B and three C. It took 293.926 seconds. No point passed all conditions, so the selected identity is null, with no final-model fit or old-28/teaser inference. We preserve all 204 candidate points, both outer OOF arrays, inner-mining arrays, fold/scaler/weight provenance and the negative result.

## Representation and weighting diagnostics

Classifier-only probability diagnostics use the same 13,425 eligible OOF rows and original labels. Lower Brier/log loss and higher AP are favorable; these do not directly measure alarm quality.

| Arm | Pooled AP | Equal-record macro AP | Pooled Brier | Pooled log loss |
|---|---:|---:|---:|---:|
| A, saved 120-feature HGB | 0.887159 | 0.921311 | 0.096255 | 0.481249 |
| B, newly trained 186-feature HGB | 0.911362 | 0.941118 | 0.092064 | 0.414039 |
| C, same 186 features with hard-normal weights | 0.907317 | 0.929641 | 0.095875 | 0.411636 |

B improves these pooled/macro representation diagnostics over A. C slightly lowers log loss relative to B but worsens its AP and Brier. Individually selected alarm cutoffs also change, so selected detection/false-start differences cannot be attributed only to features or weights. Full V5 adds a separate innovation OR branch; comparison with it also changes that system decision structure.

## Joint acceptance fails

We required >=26/29 events, fewer than 3 false starts, <=2 false starts/normal hour, <=37 normal-alert seconds over 9,179 seconds, and both source-described progressive intervals retained. data/SKAB-DATA-README.md identifies other-6 as linear and other-9 as exponential rotor imbalance. We use the original row annotations and do not invent a new gradual label column.

| Predeclared scan outcome | Events | False starts | Normal-alert seconds | Reason it cannot be accepted |
|---|---:|---:|---:|---|
| V5 product control | 28/29 | 3 | 37 | Reference, not a new learner |
| B at 0.9951986837854279 | 26/29 | 1 | 28 | Misses the linear other-6 interval |
| C best high-recall low-count/time region | 23/29 | 1 | 28 | Misses the overall recall floor and the linear interval |
| B at 0.9718653604686452, both progressive intervals retained | 27/29 | 5 | 37 | Does not reduce false starts below 3 |
| C at 0.9720286769889271, both progressive intervals retained | 27/29 | 5 | 37 | Does not reduce false starts below 3 |

These are descriptive points from the frozen scan, not alternative winners chosen afterward. No new threshold was inserted between grid points to rescue a cohort result. The conclusion applies to the tested candidate set, not a mathematical impossibility for all possible thresholds/models.

## Post-hoc failure diagnosis

B ranks the linear other-6 recording well, with normal/anomalous medians of 0.001278/0.963807 and maxima of 0.111801/0.994090. The ranges overlap, so this is not perfect class separation. The global low-false cutoff of 0.995199 lies above every anomalous score there. Held-out normal scores in other recordings reach 0.999284 in valve1-12, 0.979197 in valve1-9 and 0.977801 in valve2-0. At B's progressive-preserving point, all five false starts belong to valve1-9 (1), valve1-12 (3) and valve2-0 (1).

Strong within-record ranking is insufficient for a shared operating point across recordings. This suggests a cross-record confidence/normal-generalization problem, not a proved inability to represent a linear trend. It does not prove a specific capacity bottleneck or justify fitting a per-record threshold from test labels.

Across overlapping outer-training folds, the miner marks 1,760 normal-row appearances, not 1,760 unique independent observations. Of these, 23 fall within one second and 63 within five seconds of a label boundary. These are descriptive counts; we do not exclude/relabel them or claim they caused C's performance loss without another experiment.

## Decision and follow-up

Keep this experiment rejected and unchanged. A [separate bounded learner study](ML_V11_TREND_BACKBONE_PLAN.md) holds the 186-feature representation/base weights fixed and tests deeper HGB and ExtraTrees with the same primary conditions. That study adds model comparisons and can overfit the reused development set; it is not a fresh validation.

No active model replacement, new delay rule, external deployment or previously inspected test inference occurred in V10. A later final model, if one qualifies in another declared experiment, must retain its own identity and results separately from this negative run.

## Evidence and reproduction boundary

- [Frozen plan](../artifacts/trend-development/plan.json), [execution marker](../artifacts/trend-development/execution_started.json)
- [Complete study report](../artifacts/trend-development/report.json), [all 204 points](../artifacts/trend-development/candidate_audits.json)
- [B OOF](../artifacts/trend-development/B_oof_scores.npz), [C OOF](../artifacts/trend-development/C_oof_scores.npz), [inner mining OOF](../artifacts/trend-development/inner_mining_oof_scores.npz)
- [Selection frozen before the conditional final stage](../artifacts/trend-development/selection_before_final_fit.json)

Independent Model QA reconstructed all 204 candidates, proper losses, outer/inner partitions, hard-normal rankings/weight mass and unchanged active artifacts. No mismatches were found. All 17 frozen dependencies match. [Combined V10/V11 verification](evidence/v10-v11-learned-model-check.json) records the separate software-check and model-quality outcomes.

Report SHA-256: 96ea9680f3a00f910b33564b9a0c03b8e0c8bd963aad4149180444566bfc7939. Candidate table SHA-256: 7bb34371801abee2d8b997eaf508863f0ae00c2acee1c3fc411603cc87a76da8.

The CLI scripts/study_trends.py refuses to repeat a completed frozen run. Reproduce only in an isolated copy with the pinned inputs/dependencies and preserved source evidence. The primary papers, API checks and data boundaries appear in the protocol and RESOURCES.md; this work does not reproduce their published benchmark gains.
