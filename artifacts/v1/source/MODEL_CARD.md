# SignalWatch model card

## Intended use and verdict

SignalWatch is a local experiment-review tool comparing a trained detector with simple baselines. **The frozen learned operating point fails the practical alert-burden check.** It is in alert for 98.42% of scored time, including 97.63% of scored normal time. Only 5/28 annotated segments have a new alert onset inside them. The held-out normal audit alone spends 78.48% of scored time in alert. Use this release to inspect and reproduce the failure, not to monitor real equipment.

## Actual trained model

- scikit-learn ExtraTreesRegressor, multi-output regression for eight next-observation sensor values.
- 64 trees, min_samples_leaf=4, random_state=20261023, max_features=1.0, n_jobs=2.
- 5,142 supervised training windows from 5,172 normal-source rows.
- 30-sample historical context, 32 features: previous values, historical mean, historical population standard deviation and historical endpoint difference for each sensor.
- Sensor centering and scaling use only the normal training block. The scale is training IQR with a training-standard-deviation floor. No test-set extrema or labels are used.
- Training prediction residual medians determine channel residual scales, floored at 0.03 in scaled sensor units. This in-sample scale can be optimistic; independent calibration is essential.
- Learned score: maximum absolute channel prediction residual divided by its training residual scale. It is not a probability.
- Current model artifact SHA-256: 1c8554a484148129b281e7aa6657d078658d778b3cff6edbf3f4f16594a36cc2.

Exact versions, fitted parameters, timing and thresholds are in [metadata.json](../artifacts/metadata.json). The model is serialized with joblib. Only the application's own artifact path is accepted, and its SHA-256 is verified before loading. Loading an arbitrary uploaded pickle/joblib file is unsupported.

The Train locally browser workflow completed on 2026-10-23. It reproduced the selected candidate, thresholds, final aggregate metrics, per-recording events/metrics and group slices. Artifact bytes differ from the first training run; this is semantic reproduction, not a claim of byte-identical model serialization. Training metadata includes a new time and duration. The regenerated selection report also includes additional burden/precision fields. The split manifest and archived legacy report remain byte-identical. See [verification](VERIFICATION.md).

## Baselines

Robust statistics compares the present value with the median of the preceding 30 observations. The denominator is 1.4826 times historical MAD with a training-only floor. Persistence compares the present observation with the previous one using a training-only median absolute change scale. Both aggregate channels by maximum and reset at recording/gap boundaries. They do not use labels or future observations.

Each detector is independently calibrated on the same separate normal block against a requested empirical false-episode budget. Shared UI plots divide each score by that detector's own threshold; score ratios across methods are not probability comparisons.

## Selection and evidence boundaries

Two candidates were predefined: 64 trees with minimum leaf size 4 or 12. Training uses only the normal training block. Thresholds use only calibration. Selection uses the 12 whole validation runs, maximizing event recall minus 0.02 times false episodes per observed normal hour. The selected configuration and calibration evidence were written to selection_manifest.json before final-test scoring. No final-test tuning was performed.

The initial overlap-based event matching credited a pre-existing alarm when an annotation began. Version 2 corrects this measurement error: a detection requires a new alert episode to start inside the annotation. The original overlap report is preserved with its hash as frozen_evaluation_legacy_overlap.json. Corrected event recall is 5/28, rather than the superseded overlap count of 25/28. The trained artifact, thresholds, split and original selection are unchanged. Reproduction retains legacy overlap metrics solely inside the original candidate-selection objective; default API events, validation exploration and the final report use corrected onset metrics.

## Interpretation and limitations

The model assigns high residual scores to most held-out normal observations. This directly demonstrates an unsuitable operating point. A change in operating conditions relative to the single normal training recording is a plausible explanation, not a proven causal diagnosis. Ranking quality and threshold quality are different: a reasonably ranked score can still produce unusable alerts.

The robust baseline has a much lower normal-alert-time burden but misses many annotated segments. No detector wins every trade-off. Calibration contains only 0.666 observed scored hours, so one episode corresponds to roughly 1.50 episodes/hour; a stated budget is not a future guarantee. A nearly continuous alert also makes false-episode count alone misleading, which is why normal-alert time and point precision are mandatory alongside it.

No confidence interval implying independent failure events is reported: fragments and experiments from the same testbed are correlated. No live-production, new-equipment, demographic-fairness, root-cause or safety-critical claim is supported. Improving the model after this audit requires a new explicitly versioned development study and a genuinely untouched test set; reusing these test outcomes for tuning would make them development evidence.
