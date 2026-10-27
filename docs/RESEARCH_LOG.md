# SignalWatch research history

Snapshot 2026-10-27: **new training and cutoff tuning are paused**. V5 remains active after V13; the false-alert quality goal stays open. Read the [results report](RESULTS.md) for charts and [figure/data index](research/README.md) for inspectable exports. The history below records completed work, not a promise to execute the next model.

We retain unsuccessful experiments to explain which changes helped development selection and which failed to transfer. This chronology reconstructs saved plans, reports and source archives. We do not describe these engineering iterations as a published paper, autonomous evolution or automatic learning from user corrections.

## Version evolution

| Version | Hypothesis and implementation change | Development result | Retrospective result | Decision and evidence |
|---|---|---|---|---|
| V1 | Learn absolute next sensor values on a normal recording; use forecast error for anomaly detection. | Original selection; later diagnosis found mode mismatch and limited event observability with 30-row history. | 5/28 event starts; 6.863 false starts/normal hour; 97.632% normal time in alert. | Retain source and corrected onset evaluation in artifacts/v1/. High alert occupancy and missed event starts rule out operational usefulness. |
| V2 | Predict normal-relative innovations with shorter causal history, grouped recording folds and alert-burden constraints. | 20/29 events. | 11/28; 8.502/h; 0.972% normal time in alert. | Improved normal-alert occupancy, but development missed both valve families. artifacts/v2-development/, artifacts/v2/. |
| V3 | Use supervised known-state HGB classification after approval to learn from labelled normal and anomalous training intervals. | 24/29. | 23/28; 7.909/h; 1.483%. | Genuine supervised ML; the target covers annotated experimental anomalies, not unknown physical faults. artifacts/supervised-development/, artifacts/supervised/. |
| V4 | Give recordings equal fit mass and balance classes to limit domination by long runs. | 25/29. | 22/28; 8.502/h; 1.346%. | Development gain did not transfer to retrospective recall. Preserve artifacts/v4/ and supervised-weighted study reports. |
| V5, active | Fuse complementary supervised and normal-only innovation scores. Keep causal gap-aware inference. | 28/29; 3 false starts; 37 normal-alert seconds / 9,179 normal seconds. | 26/28; 56 false starts / 18,208 normal seconds = 11.072056/h; 272 seconds = 1.493849%. | Meets the requested recall minimum of 25/28; fails the 2/h alert-rate goal. Preserve artifacts/v5/; active release remains this version. |
| V6, rejected | Require observed elapsed persistence before a raw alert becomes a confirmed alert. | Waits 0/1/2/5/10 seconds yield 28/26/25/20/13 hits out of 29 and 3/1/1/0/0 false starts. | No audit or deployment refit. | No positive wait retained all existing development hits, as the protocol required. Keep the negative result; see ML_V6_ALARM_PLAN.md. |
| V7, rejected | Use three record-disjoint HGB/sigmoid pairs, average probabilities and retune classifier/residual cutoffs. Several factors change together. | 27/29; 2 false starts; 10 normal-alert seconds / 9,179 seconds. | 23/28; 81 false starts = 16.014938/h; 386 seconds = 2.119947%. | The single frozen audit failed recall and burden acceptance. V7 never replaced V5. See ML_V7_RESEARCH.md and ML_V7_DEGRADATION_REVIEW.md. |
| V8, development ablation | Separate raw versus calibrated scores and fixed member versus averaging; hold the residual cutoff fixed. Preserve raw member OOF predictions. | Nine fits, 35 scan points plus 5 fixed-cutoff diagnostics; no new arm meets all predeclared requirements. | No old-28 scoring or activation. | Within-record single-member AP stays equal after sigmoid; probability losses disagree; no scanned point reduces false count below V5. See ML_V8_ABLATION_RESEARCH.md. |
| V9, branch-selective policy | Confirm innovation for one observed elapsed second; preserve immediate classifier and both trained ML components. | 27/29, 1 false start and 34 normal-alert seconds versus V5's 28/29, 3 and 37; one short hit lost. | No old-28 scoring. On a separate four-intervention teaser, both detect 3/4; assumed-background starts decrease 10 to 9 and alert time 29 to 19 seconds. | Partial burden benefit with sensitivity/delay costs; retain as research candidate and keep V5 active. See ML_V9_BRANCH_POLICY_RESEARCH.md. |
| V10, learned trend features and hard-normal weighting | Train 186-feature HGB; compare base weights with inner-record-disjoint normal mining. Immediate decisions. | 15 fits, 204 points, no feasible candidate. Base-weight arm: 26/29, 1 false start, 28 seconds loses linear imbalance; retaining both progressive intervals at 27/29 gives 5 starts and 37 seconds. | No final fit or old-audit/teaser inference. | Reject under frozen joint criteria; preserve both OOF banks, inner mining and every point. See ML_V10_TREND_MODEL_RESEARCH.md. |
| V11, trend learner comparison | Hold V10 representation/preprocessing/base weights fixed; test HGB with 31 leaves and ExtraTrees256. | 6 fits, 204 points, no feasible candidate. With both progressive intervals and at least 26 hits, lowest false counts are HGB31: 27/29, 5 starts, 38 seconds; ExtraTrees: 27/29, 11 starts, 138 seconds. | No final fit or old-audit/teaser inference. | Better pooled AP/proper losses do not satisfy the alarm goal. V5 stays active. See ML_V11_TREND_BACKBONE_RESEARCH.md. |
| V12, learned normal-regime features | Learn K1/K2/K4 diagonal mixtures on balanced training-only normal banks; append twelve features to the original trend HGB. Immediate decisions. | 9 converged GMM + 9 HGB fits, 306 points, no feasible candidate. K2 with both progressive intervals: 27/29, 6 starts, 37 seconds; its 26/29, 1-start point loses linear imbalance. | No final fit or old-audit/teaser inference. | K1 reproduces V10 B OOF; K2 slightly improves proper losses, K4 worsens the relevant frontier. Normal-density mismatch persists on a held-out benchmark-normal record. See ML_V12_NORMAL_REGIME_RESEARCH.md. |
| V13, training-bank Pressure scale correction | Use max(IQR,population std,1e-6) for Pressure alone; preserve the V12 protocol otherwise. | 9 converged GMM + 9 HGB fits, 306 points; numerical audits pass, no quality winner. At 27/29 with both gradual intervals, K1/K2/K4 starts change 5/6/40 to 5/17/21. | No final fit or old-audit/teaser inference. | K4 reduces false burden but linear delay rises 3 to 46 seconds. K2 worsens. V5 stays active; see ML_V13_PRESSURE_SCALING_RESEARCH.md. |

The table reports different versions on the same previously inspected retrospective corpus. It does not establish an unbiased ranking of generalization. Rounded values follow ML_EVALUATION.md; original JSON reports retain full precision. Event counts describe annotation intervals split at gaps, not independent machines or physical incidents.

## V7 diagnosis, 2026-10-23

The innovation predictions stayed equal to V5, while the selected residual cutoff increased from 16.174573325357347 to 61.83420703195029. Two newly missed events previously depended on the innovation branch. The third depended on the classifier. Classifier-only false starts increased from 31 to 63; three recordings account for 78 of the 81 V7 false starts.

These measurements support a threshold and transfer investigation. They do not isolate calibration as the cause because training subsets and averaging also changed. The study stores all 137 V5/V7 false episodes and three raw trace cases in outputs/v7-diagnosis/.

We keep benchmark label disagreements as false alerts under the frozen metric. Sensor deviations alone cannot establish a physical hazard or an annotation mistake. The dataset lacks the operating commands, safety limits and independent incident adjudication needed for those conclusions.

## Evaluation practices retained

We freeze data partitions, candidate sets, selection rules and source hashes before each fit/scan. Whole-recording folds separate held-out rows from model fitting, preprocessing and calibration. We count actual alert episode starts and elapsed normal exposure, including normal tail time after true starts. We preserve missed events, warm-up gaps and negative results.

Repeated development reuse can overfit model selection. The old 28-event corpus had already been inspected, so later runs on it remain retrospective. We require separate confirmation before claiming that a new detector transfers. Model correctness tests and successful local runs do not satisfy R17's unresolved alert-quality goal.

## Supporting documents

- [Versioned evaluation and metric definitions](ML_EVALUATION.md)
- [Current requirements](REQUIREMENTS.md)
- [V7 source-backed degradation analysis](ML_V7_DEGRADATION_REVIEW.md)
- [V8 frozen-method protocol](ML_V8_ABLATION_PLAN.md)
- [V8 component study and negative selection result](ML_V8_ABLATION_RESEARCH.md)
- [V9 transient diagnosis, fixed policy and separate-record comparison](ML_V9_BRANCH_POLICY_RESEARCH.md)
- [V10 learned representation/training experiment](ML_V10_TREND_MODEL_RESEARCH.md)
- [V11 fixed-representation learner comparison](ML_V11_TREND_BACKBONE_RESEARCH.md)
- [V12 learned normal-regime features and coverage diagnosis](ML_V12_NORMAL_REGIME_RESEARCH.md)
- [V13 Pressure correction, paired burden and gradual-event delays](ML_V13_PRESSURE_SCALING_RESEARCH.md)

## Historical independent-source feasibility, 2026-10-23

The SKAB authors describe the [teaser](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab-teaser) as a separate collection from the same testbed with four anomalies. A public unauthenticated Kaggle API request returned version 3 and GNU Affero General Public License 3.0, which differs from the pinned main repository's data notice. We downloaded a local quarantined archive for compatibility inspection only; no model inference or training used it.

Archive SHA-256: c444f45acc329999d09bdce230a7a58256e67e5d829dafed2bc4dad4645f8801, 3,013,056 bytes. The CSV is 2,668,370 bytes with header datetime/id/value/index, so it needs a verified long-to-wide sensor mapping. The archive also contains testbed.png. Metadata and archive remain in work/v8-independent-source/; they are not part of the training manifest. The author's data card gives transition intervals rather than a ready-made anomaly/changepoint column. A separate annotation/onset specification and compatibility review must precede inference. Four same-stand events provide a limited transfer check and cannot confirm a 25/28 acceptance target.

## Separate-record comparison, 2026-10-27

The later V9 protocol froze the eight-sensor pivot, author-clock envelopes, uncertain transitions and assumed-background convention before its first teaser model inference. One unchanged V5 classifier/innovation score pass supports both V5 and V9 decisions. Both detect three of four source interventions; V9 reduces assumed-background starts from ten to nine and duration from 29 to 19 seconds. The dataset remains outside training and threshold selection. This supersedes the earlier inspection-only state without turning the four same-testbed interventions into field validation. V5 remains active; [full V9 outcome](ML_V9_BRANCH_POLICY_RESEARCH.md).

## Learned-model follow-ups, 2026-10-27

V10/V11 perform 21 new development fits without adding alarm delay. Whole-recording held-out predictions improve in ranking/probability losses, but the tested shared cutoffs cannot retain both source-described progressive intervals while reducing V5's development false count and preserving overall recall/time limits. Independent QA reconstructs both 204-point scans and confirms no feasible candidate. All conditional final stages are correctly skipped; repeated development reuse remains a selection-overfit risk.

The normal-score extremes in held-out records motivate the [V12 normal-regime proposal](ML_V12_NORMAL_REGIME_PROPOSAL.md). The user separately approved the new learned stage; its own protocol governs execution. Neither the hypothesis nor source papers establish physical operating-mode drift as the cause of every false alert.

The approved V12 run completed eighteen top-level fits with all mixtures converged. None of its 306 points passes. K2 slightly improves pooled probability losses, but its progressive-preserving frontier has six false starts instead of V5's three. A labelled-normal held-out record has much higher negative log density than every training-normal observation in fold 0, showing a coverage limit of the tested transform. No label is changed and no physical hazard is inferred. [Full V12 outcome](ML_V12_NORMAL_REGIME_RESEARCH.md). Across V10-V12, 39 top-level fits and 714 operating-point evaluations provide no qualifying replacement; these are repeated selection comparisons, not independent model-quality trials.

## Related-work comparison, 2026-10-27

At the user's request, we reviewed eight papers and five technical primary sources and compared them with preserved V5-V12 evidence. The [comparison matrix](RELATED_WORK_COMPARISON.md) covers evaluation credit, normal-regime mismatch, probability versus alarm quality, gradual evidence, validation reuse and numerical preprocessing. It includes a project-report paragraph and prioritized future hypotheses. The [source ledger](evidence/related-work-sources-2026-10-27.json) distinguishes abstracts, selected methods and official implementation guidance.

This documentation review changes no experiment result or active model. It provides a qualitative related-work comparison, not a common-protocol external benchmark or proof of physical causes. R17/R20 remain unresolved; source review does not create a quality gain.

## Pressure correction, 2026-10-27

Following that review, V13 corrects the training-only Pressure scale before repeating the fixed K1/K2/K4 development study. Maximum bank magnitude falls from about 1.64 million to 6.28; all 279 numerical audits pass and the unchanged fold-0 control matches V12 exactly. The new transform also changes physical regularization and initialization geometry, so it is not an isolated fixed-parameter arithmetic repair.

The 306-point screen produces no winner. Progressive-preserving K4 improves 40 to 21 false starts and 176 to 111 normal-alert seconds, but linear-defect delay worsens 3 to 46 seconds. K2 worsens 6 to 17 starts; K1 remains at five. Both gradual intervals remain detected at these 27/29 descriptive frontiers. The low-burden region still loses the linear interval. This completes R22's correction/comparison without closing R17/R20. [Full paired evidence](ML_V13_PRESSURE_SCALING_RESEARCH.md).
