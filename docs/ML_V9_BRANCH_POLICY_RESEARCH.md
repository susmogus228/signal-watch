# V9: transient-residual diagnosis and branch-selective confirmation

Date: 2026-10-27. Result: a measurable development tradeoff and a small benefit on a separately collected recording. **The overall false-alert quality goal remains open. V5 stays active.**

We changed the alarm rule over the existing trained HistGradientBoosting classifier and ExtraTrees innovation predictor. We trained no new model. V9 is a research candidate; it does not add online learning, mode clustering or automatic adaptation.

## Diagnosis before the change

The saved V5 development predictions reproduce 28/29 event starts, three false starts and 37 normal-alert seconds over 9,179 normal seconds. All three false-start episodes occupy one observation. They start only in the innovation branch, with the classifier far below its cutoff:

| Recording / row | Elapsed time | Classifier / cutoff | Maximum innovation / cutoff | Largest channel |
|---|---:|---:|---:|---|
| other-6 / 570 | 597 s | 0.002274 | 1.437083 | Accelerometer1RMS |
| other-9 / 216 | 226 s | 0.003492 | 1.127457 | Accelerometer1RMS |
| other-9 / 218 | 228 s | 0.013314 | 1.033003 | Accelerometer1RMS |

The previous 30 seconds contain labelled normal rows; none of the starts is at a gap or warm-up boundary. Accelerometer changes rank around the 99.6th to 99.9th percentile of same-record normal changes while preceding flow levels stay around 126 to 128 L/min. The evidence motivates a short vibration-residual excursion test; it does not establish normal-mode drift, a physical hazard or an incorrect annotation. Matched controls intentionally exclude alerting rows, so their low scores provide no independent causal proof.

Only three of the 37 normal-alert seconds belong to these false-start episodes. The remaining 34 seconds are normal tails of episodes that started inside annotations. Removing the three original starts therefore cannot by itself remove all normal-alert time. The other-6 false start precedes an annotation by three seconds; we keep its frozen false classification.

## One fixed policy, without a search

The [pre-execution protocol](ML_V9_BRANCH_POLICY_PLAN.md) confirms only the innovation branch after one elapsed second of consecutively observed above-cutoff scores. The classifier still fires immediately. Gaps, a below/equal-cutoff score or a missing value in either branch reset pending evidence. We retain joint finite eligibility and count the actual confirmation timestamp, without backdating or cooldown. Raw ML scores, pending/confirmed state and decision scores remain separate.

V6 applied a wait to the whole fused signal and lost two development hits. In one of those intervals, an immediate classifier crossing can retain detection. The other, other-12-event-16, has one eligible innovation pulse at the last row of a three-row annotated interval and no classifier crossing. Its loss was known before executing V9 and appears in the protocol.

We tested one candidate with a fixed one-second wait, unchanged V5 cutoffs and no model fitting. The development screen requires at least 26/29 events, fewer than three false starts, at most 2 starts/normal hour and no more than 37 normal-alert seconds. The minimum follows the existing ceil(29*25/28) project-target proxy. V6's stronger experiment-specific requirement to retain every hit remains part of its rejected historical result; V9 does not rewrite it.

## Development result

| Measure | V5 raw control | V9 policy |
|---|---:|---:|
| Detected gap-split annotated intervals | 28/29 | 27/29 |
| False episode starts | 3 | 1 |
| False starts / observed normal hour | 1.176599 | 0.392200 |
| Normal-alert seconds / 9,179 normal seconds | 37 | 34 |
| Median delay among detected intervals | 2 s | 3 s |
| Mean delay among detected intervals | 12.857 s | 14.630 s |

The candidate passes the declared development screen. It loses only other-12-event-16 and gains no event. It retains other-12-event-6 through the immediate classifier. Delay summaries use different detected-event counts; they do not describe a uniform one-second delay.

All three original false starts disappear, but one new false start appears in other-6 at row 975, elapsed 1,021 seconds. The raw innovation run starts on the last anomalous row 974; confirmation occurs on a normal row, so the three-row confirmed episode is a false start under the unchanged onset metric. We retain that result rather than backdating confirmation into the anomaly. For another common detected interval, other-12-event-0, delay increases from 55 to 76 seconds because the initial crossing does not survive confirmation.

We do not infer operational reliability from this development result. It reuses the same development material that motivated the rule.

## Separate-record transfer check

After freezing the candidate and source/implementation hashes, we ran one classifier prediction pass and one innovation prediction pass on the [author-owned SKAB teaser version 3](https://www.kaggle.com/datasets/yuriykatser/skoltech-anomaly-benchmark-skab-teaser). Both methods use those same raw branch predictions. No fitting, threshold selection, unit correction, startup trimming or annotation revision uses teaser outcomes.

The strict pivot maps 51,240 long-format rows into 6,405 complete eight-sensor observations, with five gaps under the existing one-second cadence rule. The source recording dates to 2019-07-08 and predates the main development recordings. It is a separate collection from the same testbed, not a prospective test, new site or machine.

The data card describes four interventions and gradual appearance/disappearance intervals, with no binary label column. We freeze source envelopes from earliest appearance to latest recovery, keep transition rows uncertain for point diagnostics and count four intervention outcomes independently of gap-split annotation intervals. Outside those envelopes we assume background from the card, without claiming independently verified safe normality.

| Measure | V5 | V9 |
|---|---:|---:|
| Source interventions with a new alert onset | 3/4 | 3/4 |
| Starts in assumed background | 10 | 9 |
| Starts / assumed-background hour | 6.340261 | 5.706235 |
| Alert seconds in assumed background | 29 | 19 |
| Scored assumed-background exposure | 5,678 s | 5,678 s |

Both methods miss the gradual shaft-defect intervention. Both detect the first shaft intervention and two valve interventions. The first shaft delay increases from two to three seconds; valve delays remain 29 and 36 seconds from the earliest appearance times. Gradual-transition latency also includes a separate reference to appearance completion in the complete report.

The maximum raw fused ratio in the missed gradual-shaft envelope is 0.9769985, below the alert boundary of 1. This is a missing raw crossing, not a pre-existing-alarm accounting artifact; confirmation cannot create a detection there. We do not lower a threshold after observing it.

V9 retains all three V5 intervention detections on this recording and reduces background alert starts and duration. It still exceeds the project's 2/h reference and misses one intervention. Four interventions and assumed-background labels cannot establish field false-alarm rates, statistical superiority or the original 25/28 target. This is limited evidence of partial transfer, not a release acceptance result.

## Decision and verification

Keep V5 active. Preserve V9 as an explicit recall/delay/burden tradeoff candidate. R17 remains unresolved. Further improvement needs a separate protocol for the remaining error mechanism, including gradual changes; this run adds no extra wait, model architecture or threshold selected after the teaser.

The full backend suite passed **134 tests** with seven existing warnings in 97.36 seconds. New tests cover selective confirmation, causal prefix/gap behavior, missing-branch resets, strict pivot failures, source-clock annotations, pre-existing alarms and refusal to repeat a completed transfer run. Independent Model QA reconciles the saved decisions, event starts, delays, exposure and source identities. The current UI/API behavior and six active V5 artifacts remain unchanged.

The new development report measures the learned detector and its policy only. Its unused robust/persistence arrays are unavailable placeholders; baseline_overlap=false and empty contributing-sensor fields in that report do not establish baseline failures or sensor attribution. V5's original measured baseline report remains separate.

## Reproduction and evidence

- [Diagnosis](../outputs/v9-diagnosis/diagnostics.json), [readable summary](../outputs/v9-diagnosis/summary.md)
- [Frozen development plan](../artifacts/branch-policy-development/plan.json), [one candidate](../artifacts/branch-policy-development/report.json)
- [Frozen transfer plan](../outputs/teaser-transfer/plan.json), [transfer outcomes](../outputs/teaser-transfer/report.json), [raw/decision traces](../outputs/teaser-transfer/traces.npz)
- [Version history](RESEARCH_LOG.md)

Development report SHA-256: d2386ddb8f77308e3138f8174bfba297509e172fa738c3286c6b64718ed1e0df. Transfer report SHA-256: 30b4336a68541a3d48a26942db2181ea5fe477c7e25f5a1d648ceaeace32756e.

The development CLI is scripts/study_branch_policy.py; the transfer CLI is scripts/check_teaser_transfer.py. Both preserve one-shot evidence and reject repeated execution against a completed run. Use an isolated fresh project copy and the pinned inputs to reproduce. Existing .gitignore excludes work/ and outputs/; a future public research package must deliberately include or provide the diagnosis script/evidence and separately licensed teaser input before claiming fresh-checkout reproduction. This turn verifies local execution, not another machine or publication.

[Prometheus pending-to-firing semantics](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/) informs the temporal rule. We use our existing confirmation helper without a Prometheus dependency. The frozen plan records the mode-shift papers considered and the limits of applying their conclusions here.
