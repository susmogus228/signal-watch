# V9 branch-selective confirmation protocol

Date: 2026-10-27. Scope approved by the user: diagnose false starts, choose one justified change, freeze its test and compare on a separate recording. This is an alarm-rule experiment on the existing trained ML components, not a newly trained classifier or automatic learning.

## Evidence and single change

The preserved V5 development OOF reproduces 28/29 event starts, 3 false starts and 37 normal-alert seconds over 9,179 normal seconds. All three false-start episodes last one observed row, start only in the innovation branch and occur away from gaps/warm-up. Causal prehistories and same-record diagnostic percentiles suggest brief vibration forecast excursions, not established normal-mode drift. Post-hoc label-based diagnostic statistics must not enter inference.

V6 confirmed the entire fused signal and lost two development events at a one-second wait. One of those events has an immediate classifier hit. The proposed single change confirms only the innovation branch; the classifier keeps its immediate decision. All component fits, feature definitions, cutoffs and maximum residual aggregation remain V5's originals.

Apply the existing apply_confirmation to residual/max-cutoff ratios with exactly **1.0 elapsed second**. Start pending on the first observed ratio strictly above 1. Confirm at the actual current observation after at least one second with every intervening observed ratio above 1. Below/equal threshold, non-finite score or gap resets pending. Do not infer continuous observations inside missing time, backdate an onset, introduce cooldown, or keep firing after a crossing ends. Clip pending residual decision ratios to 1 as the existing policy does. Fuse the confirmed residual ratio and unchanged classifier ratio by maximum, retaining V5's requirement that BOTH branches be finite.

Retain the raw branch scores, raw fused score, residual pending/confirmed state and final decision separately. A missing value in either branch resets residual pending state; apply confirmation only on jointly finite observations. Re-evaluate literal fused alert episodes; classifier and residual episodes can merge or split, so subtracting three false starts is not a valid metric calculation.

## Development screen, frozen before policy execution

One candidate, one fixed duration. No duration/threshold/aggregation grid and no model fitting. Load only the twelve development recordings and preserved normal anchors if needed, with source hashes/folds from the V5 manifest; use preserved classifier/residual OOF arrays. Reproduce the original V5 metrics first. Do not load or score the old 28-event corpus.

Require at least **26/29** development events, strictly fewer than 3 false starts, at most 2 false starts/normal hour, and at most 37 normal-alert seconds / 9,179 normal seconds. Preserve finite eligibility and normal exposure. Save every lost/gained event, paired delays, per-record/family metrics and raw/policy traces. No feasible candidate means retain V5 and do not try another parameter in this run.

The 26/29 minimum follows ceil(29*25/28), the existing development proxy for the user's >=25/28 recall target. V6 imposed the stronger experiment-specific condition of retaining every V5 development hit. We do not reuse that condition in this newly scoped recall/burden tradeoff test and do not rewrite the rejected V6 result. Diagnosis already predicts loss of other-12-event-16: it has a single eligible innovation pulse at the last row before a gap, with no classifier hit. Report this cost even if the candidate passes the project-target proxy. Development-driven choice does not prove generalization.

## Separate recording protocol

Use only the quarantined author-owned SKAB teaser version 3, archive SHA-256 c444f45acc329999d09bdce230a7a58256e67e5d829dafed2bc4dad4645f8801. Verify inventory/hash and pivot long rows by exact timestamp/sensor key to the existing eight sensor keys. Reject duplicate timestamp/sensor pairs, missing/extra sensors, non-finite values and invalid timestamps. Sort the grouped timestamps chronologically without interpolation/resampling. Preserve cadence/gap rules from the existing parser. The archive's GNU Affero General Public License 3.0 notice remains separate from project source licensing.

The author data card gives four timed interventions and no binary label column. Freeze these naive source-clock intervals on 2019-07-08 before any model inference:

| ID | Appearance starts | Appearance complete | Disappearance starts | Disappearance complete |
|---|---|---|---|---|
| shaft-instant | 18:39:22 | 18:39:22 | 18:42:32 | 18:42:32 |
| shaft-gradual | 18:44:36 | 18:45:49 | 18:46:51 | 18:46:51 |
| valve-first | 19:06:57 | 19:07:37 | 19:10:45 | 19:11:31 |
| valve-second | 19:14:40 | 19:16:24 | 19:19:15 | 19:21:16 |

For source-envelope event detection, count a new alert episode start within [appearance starts, disappearance complete); preserve pre-existing-alarm status separately. Report latency from both appearance start and completion, so gradual transitions do not imply a precise known onset. Stable anomaly-core rows lie in [appearance complete, disappearance starts); gradual transition rows remain uncertain for point metrics. Rows outside the four envelopes are **assumed background from the card**, not independently adjudicated safe normality. Call rates there unannotated-background alert starts/hour, not verified physical false-fault rates. Do not silently trim startup or reassign labels using model predictions.

After the one development candidate is frozen, compare the archived fully fitted V5 model and the same raw branch outputs with the selected policy on the complete teaser. No fitting, calibration or threshold adjustment uses teaser observations or labels. If development rejects the candidate, score only V5 as a limited transfer diagnostic; do not select another policy with teaser feedback. Freeze the candidate report/model/source identities before this inference and preserve per-event outcomes and full traces. Four same-testbed interventions cannot establish the original 25/28 target, field reliability, a precise false-alert rate or statistical superiority. Apparent shifts in operating levels/units also require disclosure, not post-hoc unit correction.

## Acceptance boundary and checks

Development success identifies a research candidate only. Independent comparison describes transfer and costs; activation/API retraining is outside this diagnostic run. Keep six active V5 artifacts and existing scientific sources unchanged. Do not push/deploy, alter global dependencies or modify frontend/API contracts.

TDD covers selective immediate classifier behavior, residual elapsed support, gap/NaN/reset, prefix/label independence, finite-mask invariance, short-event loss and candidate rejection; teaser tests cover exact pivot, invalid data and timestamp-boundary annotations. Independent Model QA must check the protocol before policy execution and reconcile final saved arrays/metrics. Run the full backend suite after source changes. Output: artifacts/branch-policy-development/, outputs/v9-diagnosis/, outputs/teaser-transfer/ and English research history/docs.

## Primary sources

[Prometheus alerting rules](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/) documents pending-to-firing elapsed support. We reuse our own existing implementation, without a Prometheus dependency or promised detection guarantee. [Author's teaser card](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab-teaser) supplies source intervals. [CANDI](https://ojs.aaai.org/index.php/AAAI/article/view/38524) and [multimode monitoring](https://doi.org/10.1016/j.ifacol.2019.06.134) motivate the mode-shift question; the three observed development false starts do not establish that mechanism, and this experiment adds neither adaptation nor clustering.
