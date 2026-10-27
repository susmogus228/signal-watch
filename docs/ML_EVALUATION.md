# Versioned ML studies and retrospective evaluation

## Results

| Version | Development selection | Retrospective hits / 28 | False onsets / normal hour | Normal time in alert |
|---|---|---:|---:|---:|
| V1 absolute next-value predictor | Original selection | 5 | 6.863 | 97.632% |
| V2 normal-only innovation | Grouped OOF 20/29 | 11 | 8.502 | 0.972% |
| V3 supervised classifier | Grouped OOF 24/29 | 23 | 7.909 | 1.483% |
| V4 weighted supervised, archived | Grouped OOF 25/29 | 22 | 8.502 | 1.346% |
| V5 complementary fusion, active | Grouped OOF 28/29 | 26 | 11.072 | 1.494% |
| V6 confirmation, rejected | Identity retained; positive waits lose hits | Not audited | Not audited | Not audited |
| V7 calibrated ensemble, rejected | Nested grouped OOF 27/29 | 23 | 16.015 | 2.120% |
| V8 controlled component study | No new arm meets all frozen development conditions | Not scored | Not scored | Not scored |
| V9 fixed branch-selective policy, research candidate | 27/29, 1 false start, 34 normal-alert seconds | Not scored | Not scored | Not scored |
| V10 trend features/hard-normal weighting, rejected | 204 points; no joint-screen winner | Not scored | Not scored | Not scored |
| V11 alternative trend learners, rejected | 204 points; no joint-screen winner | Not scored | Not scored | Not scored |
| V12 learned normal regimes, rejected | 306 points; no joint-screen winner | Not scored | Not scored | Not scored |
| V13 Pressure scaling correction, rejected for activation | 306 points; numerical checks pass, no joint-screen winner | Not scored | Not scored | Not scored |

**V5 meets the requested minimum of 25/28 events, but fails the 2/h alert-rate limit.** Its frozen development selection determines activation; retrospective counts do not choose the candidate.

V5: 56 false starts / 18,208 scored normal seconds = 11.072056/h; 272 normal-alert seconds / 18,208 = 1.493849%. Median detected-event delay is 18 seconds, mean 25.115 seconds; two missed segments remain separate. Relative to V4, four more segments are detected, false starts increase by 13 (+30.2%), and normal-alert time increases by 27 seconds. V4 had 22/28, 43 false starts and a 29-second median delay. Software checks and the recall target do not establish operational reliability.

V5 family counts: other 12/14 with 24 false starts over 7,029 normal seconds; valve1 11/11 with 29 over 7,691; valve2 3/3 with zero over 1,554; normal audit has 3 over 1,934. Exposure sums to 18,208 seconds and false starts to 56.

## Diagnosis and experiments

The [research history](RESEARCH_LOG.md) records the version sequence, hypotheses and decisions. The [V8 ablation protocol](ML_V8_ABLATION_PLAN.md) separates component diagnostics from threshold selection and prohibits another old-28 selection scan.

V8 completed nine nested fits, 35 cutoff points and five fixed-cutoff diagnostics. Single-member raw/calibrated AP is equal within each of the twelve development records; log loss improves while Brier loss worsens after calibration. No scanned fusion point has fewer than the V5 reference's three false starts, so best_all/best_control_A/best_new are null. This is a negative development selection result, not a new release or independent validation. [Component results](ML_V8_ABLATION_RESEARCH.md).

V9 changes only innovation-branch alarm confirmation at a fixed one-second wait. The development comparison gives 27/29 and one false start instead of 28/29 and three, with one short event lost and a largest common-event delay increase of 21 seconds. A single frozen separate-record comparison on the SKAB teaser gives 3/4 source interventions for both V5/V9, with assumed-background starts 10/9 over 5,678 seconds and alert time 29/19 seconds. Source envelopes and background assumptions differ from binary gap-split annotation truth; these are limited transfer diagnostics. V9 remains inactive and R17 remains open. [Protocol and results](ML_V9_BRANCH_POLICY_RESEARCH.md).

V1 normal training flow median is 124.668 L/min; normal rows in development experiments have median 32.9574 L/min. Acceleration and temperature regimes also differ. An absolute-level tree predictor mistakes normal operating regimes for anomalies. Thirty-row warm-up leaves 16/29 development segments unscorable. The original selection objective lacks a normal-alert-time constraint.

Those development findings motivated normal-relative prediction, shorter causal history and burden limits. Normal-only V2 misses both valve families in development. The user then approved supervised known-state classification. The bounded weighting follow-up gives long and short recordings equal training mass; it improves development hits from 24/29 to 25/29, without the same improvement on retrospective records.

V5 exploits complementary development failures: normal-only V2 finds 20/21 other segments and no valve segments; V4 finds 17/21 other and all 8 valve segments. Four frozen fusion candidates use three whole-recording OOF folds, the fixed classifier threshold and normal-only residual quantiles. The selected maximum-residual fusion reaches 28/29, with 3 false starts and 37 normal-alert seconds over 9,179 normal seconds. See ML_V5_PLAN.md and ML_V5_RESEARCH.md for provenance and boundaries.

## False-alert follow-up outcomes

V6's one frozen scan used elapsed waits 0/1/2/5/10 seconds. Development hits fell 28/26/25/20/13, while false starts fell 3/1/1/0/0. Its acceptance rule required every previously detected event to remain detected; no positive wait qualified. No retrospective scoring or refit followed. [Protocol and table](ML_V6_ALARM_PLAN.md).

V7 tested three sigmoid-calibrated HGB members with nested record-disjoint fitting/calibration, retaining the normal-only predictor. Its fixed 147-point grid had six qualifying points. The declared ranking selected max residual aggregation at classifier threshold 0.8580258155093349 and residual threshold 61.83420703195029. Development gave 27/29, 2 false starts (0.784/h), 10 normal-alert seconds (0.109%) over 9,179 seconds; median delay 3 s. One previous hit, other-12-event-16, was lost. No event was gained.

The winner's single retrospective audit failed: **23/28, 81 false starts (16.015/h), 386 normal-alert seconds (2.120%) over 18,208 normal seconds**. V5 remains 26/28, 56 and 272 seconds. Lower median detected-event delay, 3 versus 18 seconds, does not compensate for five misses and greater false burden. Baseline results and exposure stayed equal. The candidate failed the >=25/28, fewer-false-starts and no-higher-normal-alert-time gate; active files remain unchanged. No alternative audit scan or API training activation occurred. The generalization gap remains unresolved. [Method](ML_V7_RESEARCH.md), [frozen development evidence](../artifacts/calibrated-development/report.json), [single audit](../artifacts/calibrated/retrospective_audit.json).

## Learned trend/classifier follow-ups

V10 trains 186-feature HGB models using causal actual-time trends and compares base weights with training-only hard-normal mining. Fifteen fits and 204 fixed operating points produce no feasible candidate. At 26/29 and one false start, the linear progressive interval is lost; retaining both progressive intervals at 27/29 requires five false starts and 37 normal-alert seconds. [V10 method and diagnosis](ML_V10_TREND_MODEL_RESEARCH.md).

V11 holds those features/base weights fixed and trains HGB31 and ExtraTrees256 across the same three whole-recording folds: six fits and 204 fixed points. Pooled AP rises to 0.925149/0.921981 versus the original classifier's 0.887159, but no point satisfies the joint acceptance screen. With both progressive intervals and at least 26 total hits, the smallest false counts are 5 for HGB31 (27/29, 38 normal-alert seconds) and 11 for ExtraTrees256 (27/29, 138 seconds). [V11 method and results](ML_V11_TREND_BACKBONE_RESEARCH.md).

These studies use immediate classifier decisions, with no temporal confirmation. They require >=26/29, fewer than three false starts, <=2/h, <=37 normal-alert seconds over 9,179 normal seconds and retention of other-6/other-9. Scores improve without proving a better alarm system. No winner is refitted, no old-28/teaser inference follows, and V5 stays active. The quality goals remain unresolved.

V12 adds twelve learned normal-regime features to those 186 columns. K1/K2/K4 use nine converged diagonal GMM and nine HGB fits. All 306 points fail the same screen: progressive-preserving 27/29 points have 5/6/40 false starts and 37/37/176 normal-alert seconds. Low-burden K1/K2 can give 26/29 with one start and 28 seconds but lose linear other-6. K1 OOF equals V10 B; K2 slightly improves proper losses, K4 worsens them. Convergence does not establish coverage of unseen benchmark-normal records. No final fit, old-audit/teaser scoring or activation occurs. [V12 evidence and diagnosis](ML_V12_NORMAL_REGIME_RESEARCH.md).

V13 changes only training-bank Pressure scaling, using max(IQR,population std,1e-6). All numerical checks pass, with maximum fixed-parameter NLL disagreement 3.64e-12. The same-K progressive-preserving frontiers retain 27/29: K1 false starts remain 5, K2 rise 6 to 17, K4 fall 40 to 21. K4's linear delay rises from 3 to 46 seconds, despite no added confirmation rule. All 306 points fail the joint screen; low-burden K1/K2 retain 26/29 with one start but miss the linear interval. No final fit/audit/activation follows. These are reused development comparisons at each arm's own cutoff. [Pressure results and limitations](ML_V13_PRESSURE_SCALING_RESEARCH.md).

## Protocol

We keep the original twelve disturbed development recordings and twenty-two retrospective disturbed recordings plus the normal audit block. Each development fold holds out whole recordings. Preprocessing and model fit use training recordings only. Supervised training uses their normal and anomalous labels. Inference uses no labels, IDs, timestamps or row indices as features.

Each plan freezes candidates, folds, threshold quantiles, constraints, tie-breaking and no-feasible behavior before scoring. V2 through V5 evaluate 101 normal-score quantiles from 0.90 to 1 plus a value above the maximum. V6 freezes five elapsed waits; V7 freezes six normal-score quantiles plus a value above the maximum for each branch and three residual aggregations. Development requires <=5% normal-alert time and <=2 false onsets per normal hour. We rank feasible points by event hits, normal-alert fraction and false-start rate. The weighted-versus-original-supervised choice rule was fixed before weighted results. We use no smoothing, periodic alert resets or point adjustment.

The audit recordings had already been examined in V1. New scores on them are retrospective, not a fresh blind test. Repeated selection also reuses development data and can overfit it. Further audit-driven training needs new independent evaluation data.

## Definitions and availability

An event is consecutive anomaly=1 rows, split at timestamp gaps. A hit requires an actual alert episode START inside that interval. A pre-existing alert does not count. False starts begin on normal rows, even if the episode later overlaps an annotation. Normal-alert duration includes carryover from a true onset. A gap exceeds three times initial cadence; we exclude missing time from exposure and reset history. Left-endpoint intervals determine exposure; the terminal observation contributes zero seconds. Warm-up scores remain unavailable.

With two-row history, 29/29 development segments are observable versus V1's 13/29, and 28/28 retrospective segments versus 25/28. Reports retain point/time metrics on common thirty-row eligibility. We never mask alerts to manufacture new onsets. Increased observability contributes to the improvement.

## Saved evidence

- Original failure/source: artifacts/v1/ and preservation_manifest.json.
- Normal-only: artifacts/v2-development/ and artifacts/v2/retrospective_audit.json.
- First supervised: artifacts/supervised-development/ and artifacts/supervised/retrospective_audit.json.
- Weighted: artifacts/supervised-weighted-development/ and artifacts/supervised-weighted/retrospective_audit.json.
- Complementary fusion: artifacts/hybrid-development/ and artifacts/hybrid/retrospective_audit.json; original audited child/report retained in artifacts/hybrid/.
- Archived V4 release/source: artifacts/v4/preservation_manifest.json.
- Active report/model: artifacts/frozen_evaluation.json, metadata.json, model.joblib.
- Choice rule and original audit/model identity: artifacts/release_manifest.json.

The archived overlap report documents an earlier metric error. Its 25/28 credited pre-existing alarms; corrected V1 is 5/28. We retain the corrected onset rule throughout these studies.

## References

The [related-work comparison](RELATED_WORK_COMPARISON.md) maps external research to our exact metrics and failures. SKAB/NAB windows, point-adjusted F1, VUS-PR, empirical false-episode rates and no-change expected runtime answer different questions. No external leaderboard number is a direct counterpart of our onset_v2 recall or 2/h operating target. The comparison preserves the original metric definitions and all negative results.

[SKAB source](https://github.com/waico/SKAB), [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold), [Kim et al. on point-adjustment inflation](https://ojs.aaai.org/index.php/AAAI/article/view/20680), and [gradient boosting](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting).
