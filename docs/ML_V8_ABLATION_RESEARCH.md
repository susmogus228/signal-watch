# V8 controlled component study

Date: 2026-10-23. Outcome: **no feasible new arm** under the frozen development rule. V5 remains active. This study diagnoses the V7 changes; it does not establish a replacement detector's generalization.

## Protocol and execution

We ran the [pre-fit protocol](ML_V8_ABLATION_PLAN.md) once: nine nested HGB/calibrator fits, five fixed arms, 35 classifier-cutoff scan points and five separate common-cutoff diagnostics. Each arm reuses the same V5 maximum innovation scores at cutoff 16.174573325357347. We did not refit a deployment model, score the old 28 events, change labels or activate a candidate.

Model QA checked the plan and 24 dependency hashes before fitting. Frozen plan SHA-256: e0a401b170779ec36b5e5005f833d5a7f7c0018d7c947fa70251affcc8870f81. The runner recorded source/environment/active-file hashes before fits. The run took 153.725 seconds and reproduced the saved V7 classifier OOF within absolute tolerance 1e-12 and zero relative tolerance. The six active V5 files and frozen scientific sources stayed unchanged.

## Probability diagnostics

These losses use classifier probabilities on the same eligible development rows, before fusion. Lower Brier/log loss is better; higher AP is better. Macro AP averages records equally; pooled AP ranks rows across folds. The task concerns the benchmark's annotated known anomalies.

| Arm | Classifier | Pooled Brier | Pooled log loss | Pooled AP | Macro-record AP |
|---|---|---:|---:|---:|---:|
| A | Saved V5 outer-full-training OOF | 0.096255 | 0.481249 | 0.887159 | 0.921311 |
| B | Fixed member 0, raw | 0.100259 | 0.487819 | 0.893829 | 0.926611 |
| C | Same member 0, sigmoid | 0.109576 | 0.478912 | 0.870751 | 0.926611 |
| D | Mean of three raw members | 0.094221 | 0.383431 | 0.907544 | 0.934026 |
| E | Mean of the same three calibrated members | 0.102837 | 0.342228 | 0.891573 | 0.937294 |

For B/C, all twelve per-record AP values are identical. Increasing sigmoid mappings preserve ordering within each record; different fold mappings alter pooled ranking and pooled-cutoff crossings. All nine fitted sigmoids increase with the raw score. Calibration lowers log loss but increases Brier loss in both B/C and D/E comparisons, so these results do not support a universal probability-quality improvement.

The mean bank improves macro AP over the predeclared member 0 in this development sample. D/E differ within records because averaging and member-specific nonlinear calibration do not commute. The two additional members and their training subsets form a particular bank, so we do not generalize that comparison to all ensembles or attribute A/B differences to calibration.

One fold/member calibrator produces 747 held-out values at least 1-1e-12 and eight additional duplicate-score rows relative to its raw output. Member 0 has no added saturation/ties in the B/C comparison. We retain these diagnostics instead of treating extreme calibrated probabilities as certain physical failures.

## Alarm operating points

Each row below uses 9,179 scored normal seconds. The original V5 operating point stays a separate control: **28/29 events, 3 false starts, 37 normal-alert seconds**. Arm A below also changes its classifier cutoff, so it is not that unchanged operating point.

| Arm | Normal-score quantile 0.999: hits / false starts / normal-alert seconds | Quantile 1.0: hits / false starts / normal-alert seconds |
|---|---|---|
| A | 27 / 5 / 18 | 23 / 3 / 9 |
| B | 28 / 5 / 19 | 24 / 3 / 9 |
| C | 28 / 5 / 19 | 25 / 3 / 9 |
| D | 27 / 6 / 18 | 27 / 3 / 9 |
| E | 28 / 5 / 19 | 26 / 3 / 9 |

These two predeclared quantiles illustrate the tradeoff; all 35 points remain available in the saved candidate report. Three false starts correspond to 1.176599 per normal hour. No scanned point has fewer than three false starts. D/E at the maximum-normal cutoff meet the recall and absolute burden limits but fail the predeclared requirement of fewer false starts than V5. We do not weaken that rule after observing results. All three selection outputs, best_all/best_control_A/best_new, are null.

At the common classifier cutoff 0.8580258155093349, A/B/C/D/E yield 28/27/26/28/28 hits and 44/18/5/7/5 false starts. This is a sensitivity check on different probability scales, not a fair equal-burden ranking. Separately tuned normal-quantile cutoffs also change decision rules, so their alarm metrics cannot isolate calibration alone.

Arm E at that cutoff uses exactly the saved V7 classifier OOF but the lower fixed V5 residual cutoff. It gives 28/29, 5 false starts and 19 normal-alert seconds. The original V7 development point used residual cutoff 61.83420703195029 and gave 27/29, 2 false starts and 10 normal-alert seconds. On these development records, that residual cutoff change traded one detected event for three fewer false starts and nine fewer normal-alert seconds. It does not explain every retrospective error or prove how a new recording will behave.

## Conclusions and next boundary

The data separate several effects that V7 changed together. Single-member calibration does not improve within-record discrimination here; a three-member bank changes ranking; threshold choices trade short-event sensitivity against alert burden. Proper probability losses and pooled ranking can disagree with per-record or alarm outcomes. These observations explain why one attractive summary metric cannot choose an operational detector.

The fixed innovation branch can mask classifier changes, and all scanned fusion points retain at least three false starts. This observed scan minimum is not a proof that another decision rule cannot reduce alerts. The study stores classifier-only metrics, per-record/family exposure and paired crossing/onset differences to support the next causal diagnostic without relabelling errors.

R17 remains open. Before another replacement claim, define a separate experiment that targets the remaining false-episode mechanism and confirm the frozen choice on independent recordings. The quarantined SKAB teaser needs sensor mapping and transition-label rules before inference; it has only four same-testbed events. No new architecture, adaptation mechanism or external data transfer entered this run.

## Evidence

- [Frozen plan](../artifacts/ablation-development/plan.json)
- [Pre-fit manifest](../artifacts/ablation-development/prefit_manifest.json)
- [Complete report](../artifacts/ablation-development/report.json)
- [All 35 scan points](../artifacts/ablation-development/candidate_audits.json)
- [Raw member OOF bank](../artifacts/ablation-development/raw_member_oof_scores.npz)
- [Calibrated member OOF bank](../artifacts/ablation-development/calibrated_member_oof_scores.npz)
- [Version history](RESEARCH_LOG.md)

Report SHA-256: 675d51d09b127699381cf16ba9fc8f1befb63bdc099091942cf3c3076ba2efab. Candidate table SHA-256: edb3c13b48065a9f6c30c067eb274d1de22a13711489b5caa28ed1c8009410c5. We retain experimental correctness and model-quality acceptance as separate verdicts.

Independent Model QA recomputed proper losses, arm arrays and all 40 fused/classifier-only comparisons from saved OOF arrays, including per-record/family metrics and onset events. It found no mismatches and accepted experimental correctness; false-alert quality remains insufficient. The full backend suite passed 116 tests. [Verification record](evidence/v8-component-study-check.json).

For reproduction, scripts/study_ablation.py freezes a plan before fitting and refuses to overwrite a completed study. Use a separate clean project copy with the same source/dependency snapshots and no existing artifacts/ablation-development/ outputs. Preserve the original run. The saved environment and OOF arrays allow metric reconciliation without model inference. We did not test reproduction on another machine in this turn.
