# SignalWatch model card

Snapshot 2026-10-27: new research training/cutoff tuning is paused at the user's request. V5 remains the active artifact; V13 is a rejected research follow-up. [Results and limitations](RESULTS.md) collect the measured outcomes and [version history](RESEARCH_LOG.md).

## Active V5 and intended use

Use SignalWatch to review recorded sensor experiments. V5 combines known-state classification and normal-behavior prediction. The retrospective result is **26/28**, with **11.072 false onsets/normal hour** and **1.494% normal time in alert**. The 25/28 recall goal is met; the 2/h alert-rate limit remains unmet. False starts increase from 43 to 56 (+30.2%) versus V4. Field reliability and unknown-fault generalization require further evidence.

## Classifier branch

- HistGradientBoostingClassifier: 150 iterations, 15 maximum leaves, min_samples_leaf=20, learning_rate=0.05, L2=1, seed=20261023. Internal early stopping is disabled.
- 18,595 examples: 13,915 normal and 4,680 anomalous rows from the normal anchor and twelve development recordings.
- 120 causal features: current eight-channel values, previous-step changes, past means and standard deviations over available prefixes of 2/5/15/30 rows, current-minus-past means, and deviations from the first-two-observation reference after each gap.
- Two history observations before learned scoring; both unchanged baselines retain thirty. The initial reference can contain an anomaly.
- Training-fold median/IQR normalization with standard-deviation and numerical floors. Held-out recording values and labels do not enter fitting.
- Record-equal, class-balanced training weights: w_i=N/(R*K_r*n_rc). Each record has equal mass, shared by its represented classes. Evaluation uses ordinary unweighted observations.
- Threshold 0.9954052245087474. The classifier score is uncalibrated; it does not measure physical-failure probability.

Exact parameters, feature names, wrapper/child hashes and provenance appear in [metadata.json](../artifacts/metadata.json). Only trusted bundled joblib files with verified hashes are supported.

## Normal-behavior branch and fusion

The second branch is an ExtraTreesRegressor with 64 trees, min_samples_leaf=12 and a two-observation causal window: 32 predictor features alongside 120 classifier features, 152 in total. It fits 13,891 normal examples, a subset overlapping the 18,595 classifier examples; these counts must not be added. Channel residual scales are estimated on the separate normal calibration anchor, without development-held-out or retrospective records. The maximum of eight normalized residuals is compared with 16.174573325357347. The fused score is max(classifier_score/0.9954052245087474, residual_score/16.174573325357347), with alert threshold 1 and both branches available. This is a relative alert level, not a calibrated probability.

## Selection and reproduction

We froze six normal-only innovation candidates, four supervised classifiers, then four weighted supervised classifiers. Each uses three whole-recording development folds. Selection uses grouped OOF scores with pooled limits of 5% normal time in alert and 2 false onsets/normal hour. Feasible points rank by event hits, lower normal-alert fraction, lower false-onset rate and declared order. The release rule was fixed before weighted results; audit counts do not select the active version.

V5 freezes four candidates: classifier-only control, or fusion with maximum, top-two mean, or all-channel mean residuals. Residual OOF channels are reconstructed from the original three normal-only fold fits and checked against archived V2 maximum scores. Classifier OOF scores come from the hash-verified V4 archive. Only a strict development improvement replaces the control. Maximum residual wins the declared rank: 28/29 events, 3 false starts / 9,179 normal seconds (1.177/h), and 37 normal-alert seconds (0.403%). These reused selection scores are not an unbiased final estimate.

Budget exploration selects among 307 saved OOF operating points, verifies their SHA against active metadata, and reapplies feasibility for the requested budget. It does not refit or rescore retrospective records. The frozen scientific plan hashes six scientific source files and all study dependencies.

Train locally runs the selected study in a separate work folder. Before replacement it checks the plan, selected point, aggregate and family metrics, and each recording's metrics/events. Failure leaves active inference and development exploration intact. Original V1 and V4 artifacts and source remain under artifacts/v1 and artifacts/v4 with preservation hashes. The originally audited V5 child and development report are separately retained under artifacts/hybrid; retraining may change joblib serialization but must reproduce every saved audit metric and event before activation.

## Sensor interpretation

Sensor buttons rank absolute deviations from the initial two-observation reference, normalized by training sensor IQR. They describe observations, not classifier attribution or physical causes. Positive-threshold plots show score/threshold. For a zero threshold, the API declares an offset display transform (score+1)/(threshold+1); metrics still use raw score > raw threshold.

## Learning from errors

We use complementary trained branches, gradient boosting and error-driven iterative development. Within boosting, successive trees reduce ensemble training loss. Between releases, we inspect development errors and train a new frozen study from scratch. Automatic feedback learning, teacher-student transfer, and evolutionary mutation/crossover are not implemented. Training on audit errors requires a new independent evaluation set.

## Recorded replay timing boundary

Causal feature/score checks assume preloaded recording cadence and gap configuration. Cadence uses the first 30 timestamp intervals of the complete CSV before replay begins. The API and UI declare this configuration; initialization from unknown raw timestamps is not tested. Identical early sensor/time prefixes can have different early gaps if their preloaded cadence differs. This recorded-file limitation does not change the frozen audit metrics.

## Rejected false-alert follow-ups

The active artifact is still V5. V6 tested observed elapsed confirmation at 0/1/2/5/10 seconds. Positive waits lost previously detected development event IDs; no new retrospective audit or activation followed.

V7 genuinely trained three HGB/sigmoid pairs with disjoint fitting/calibration recordings and a fixed normal-only predictor. Nested whole-recording evaluation excluded outer-held-out records from both fitting and calibration. The deployment ensemble uses its three calibrated pairs without a common classifier refit. Equal-record calibration weights preserve each recording's observed class prevalence; calibrated outputs do not establish deployment fault probabilities.

The fixed 147-point development grid selected maximum residual aggregation, classifier threshold 0.8580258155093349 and residual threshold 61.83420703195029: 27/29 hits, 2 false starts, 10 normal-alert seconds / 9,179 seconds. It lost other-12-event-16 versus V5. The single frozen retrospective candidate scored 23/28, 81 false starts and 386 normal-alert seconds / 18,208 seconds. All three acceptance dimensions regressed; the release gate rejected it. It was not activated or repeated through the training API, and no alternative was selected using that audit.

V7 preserves 18,595 unique classifier observations, 13,425 unique calibration observations drawn from that same set and 13,891 overlapping normal-predictor observations. Three classifiers and three sigmoid calibrators do not triple observations or the 152-feature vocabulary. Detailed provenance and negative results: [V7 study](ML_V7_RESEARCH.md), [development report](../artifacts/calibrated-development/report.json), [retrospective audit](../artifacts/calibrated/retrospective_audit.json).

## Limits

Later [V8-V13 research](RESEARCH_LOG.md) retains the active V5 artifact. V10/V11 train trend/hard-normal/different-learner variants; V12 adds learned normal-regime features before HGB. V13 corrects Pressure scaling: numerical checks pass and K4's progressive-preserving false count falls 40 to 21, but its linear delay rises 3 to 46 seconds and K2 worsens. None passes its joint event/progressive/false-alert screen, so none becomes a final model or replaces this card's active architecture. [V13 outcome](ML_V13_PRESSURE_SCALING_RESEARCH.md) separates numerical correction from alarm quality. V9 remains a separate policy research candidate with documented event/delay costs.

All recordings come from one physical testbed. Gap-split segments share conditions and are not independent failures. V5 observes 28/28 retrospective segments versus V1's 25/28; shorter warm-up contributes to the difference. Reports retain common-eligibility point/time metrics. Retrospective family hits are other 12/14, valve1 11/11 and valve2 3/3. False-start rates are 12.292/h, 13.574/h and 0/h respectively; the separate normal audit has 3 false starts (5.584/h). Median detected delay is 18 seconds; two misses remain. Branch context reports actual threshold crossings inside the visible interval, never physical cause or a guaranteed event attribution.
