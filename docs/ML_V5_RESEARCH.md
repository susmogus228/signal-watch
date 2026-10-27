# V5 hybrid research and development-only diagnosis

## Question and evidence boundary

Can the complementary normal-innovation and weighted known-state classifiers improve event coverage at the existing alert burden? This note inspects implementation, frozen development OOF arrays and primary scikit-learn documentation. It does not inspect old test recordings or retrospective audit results. The initial fixed-OR calculation is exploratory development motivation, not an independent validation result or a selected release.

## Existing complementary evidence

The innovation branch uses window=2, 64 ExtraTrees and minimum leaf=12. Its development threshold is 16.174573325357347. The weighted HGB classifier has 150 iterations, 15 leaves, minimum leaf=20, learning rate=0.05 and L2=1, with record-equal class-balanced sample weights. Its fixed threshold is 0.9954052245087474.

The existing hash-verified OOF arrays cover the same twelve development recordings and the same three held-out-recording folds. Both branches have identical finite-score eligibility after two past observations and after gaps. A direct fixed OR, evaluated as full sequences rather than summing event counts, gives:

| Development quantity | Fixed OR |
|---|---|
| Detected annotated segments | 28/29 |
| Missed segments | 1 |
| False onsets / observed normal time | 3 / 9,179 seconds = 1.1766/hour |
| Normal time in alert | 37 / 9,179 seconds = 0.4031% |
| Literal point precision | 98.6374% |
| Median actual onset delay | 2 seconds |
| Family coverage | Other 20/21; Valve1 6/6; Valve2 2/2 |

The added detections relative to the classifier are other-6 and two annotated fragments in other-12. They are not three independent machines or newly collected failures. The actual OR onset hit set equals the union of individual hit sets in this calculation; no episode-merging losses occurred here. The implementation still has to test that failure mode because it can occur on other traces. The complete calculation is saved in work/hybrid-diagnosis/fixed_or_diagnostic.json.

The Other family retains 3.477 false onsets/hour despite passing the aggregate constraint. A summary average must not hide this slice.

## Implementation findings

signalwatch.ml_v2.innovation_scores already returns actual normalized per-channel residuals in contributions, shape n x8. However, selected_oof_scores.npz stores only the maximum scalar per row. Mean-of-top-two and mean-of-eight residuals cannot be recovered from a maximum. Reconstruct the three fixed normal-branch fold models, retain their OOF per-channel errors, and verify their maxima against the frozen scalar arrays. Using the final model trained on all twelve development recordings to construct OOF contributions would leak training information.

Only three aggregation variants are needed: maximum, mean of the largest two channels, and mean of all eight channels. They share the exact predictor fit, channel normalization, history and labels. No additional tree search, sensor selection, temporal smoothing or event-rule change is justified in this study.

For positive branch thresholds, fused score=max(classifier_score/classifier_threshold, aggregated_residual/residual_threshold), with alert threshold 1, is exactly a pointwise OR. Both input scores must be finite. It is an alert-level ratio, not a calibrated failure probability. Keeping raw classifier and residual branches enables trace reconciliation and future-prefix tests.

## Calibration and transfer limits

An OR can never reduce the classifier's pointwise false-alert time: every classifier-positive normal sample remains positive in the hybrid. Residual aggregation may reduce extra alarms contributed by that branch, but cannot repair an already unsuitable classifier operating point. Episode counts and detected-onset counts are not monotone under OR because episodes can merge. Report both physical normal-alert time and actual onset metrics.

The current OOF models use nine training groups each, including the normal anchor; final refits use thirteen groups. Normal-branch per-channel residual normalizers also change: recorded fold/final ratios range as low as 0.880 for accelerometer1 and as high as 1.342 for flow rate. This demonstrates a changed scoring pipeline, not a measured failure on an unseen recording. A threshold selected from OOF models is not guaranteed to retain the same normal burden after a final refit.

[cross_val_predict](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.cross_val_predict.html) documents that each prediction comes from a model trained outside that sample's held-out fold, and warns against treating pooled predictions as a generic generalization estimate for nondecomposable metrics. Here the OOF arrays select thresholds and configurations, so they are explicitly development evidence.

[Threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html) warns against fitting a classifier and choosing its decision threshold on the same observations. [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html) describes both fold-model ensembles and the alternative final refit, and requires disjoint fitting and calibration data for a prefit estimator. These sources motivate preserving calibration boundaries; they do not establish that our anomaly thresholds are probabilistically calibrated.

A future separate experiment could hold a model fixed, calibrate on recording-disjoint normal material, and deploy that exact model without another refit. That changes the training/calibration architecture and must be separately frozen. It is not mixed into the bounded V5 aggregation study.

## Recommended bounded study and API

Follow ML_V5_PLAN.md: classifier-only fallback plus the three residual aggregations. Keep the classifier threshold fixed. For each residual aggregation, select its threshold only from the fixed 101 normal OOF quantiles plus a value above the maximum; evaluate constraints on the complete fused traces. Rank feasible points by hits, normal-alert fraction, false-onset rate and stable candidate/threshold order. Exact ties preserve classifier-only. No development improvement means retain the current release and skip another retrospective audit.

New module APIs: score_recording(child_bundle,recording,base_scored), load_development_bundle(root), compare_development_budget(root,bundle,budget) and run_study(root). Preserve actual baseline arrays. Store per-channel OOF residuals, classifier OOF scores, branch thresholds and dependency hashes. The final bundle records both branch models and their separate fitting counts; unique training rows are classifier-eligible rows, not the sum of overlapping branch counts.

Scoring must never read labels, recording ID, filename, timestamps or elapsed position. Explanatory channel values remain observed reference deviations, not feature attribution or inferred mechanical causes. Only a selected, frozen development winner may undergo one later retrospective audit; the old 28 target remains unproven by this note.

## Completed bounded development study

After root and independent protocol preflight, the implementation froze plan SHA-256 0fc5da6ef7a9e2badb6f0d8fdc40f5c47861dca215ff4a12645b9d26a8c5dfdc before its candidate scan. The three reconstructed fold predictors reproduced every stored V2 OOF maximum within absolute tolerance 1e-8, with identical finite eligibility. The classifier scores were copied byte-for-byte from the preserved V4 development archive.

The selected maximum aggregation has 28/29 OOF development detections, 3 false onsets per 9,179 observed normal seconds, 37 normal alert seconds and median delay 2 seconds. Mean-of-top-two gives the same hit count and normal burden; the declared candidate-order tie-break retains maximum. Mean-of-all-eight gives 27/29 hits with zero false onsets and therefore loses the predeclared maximum-hit ranking. None of these alternatives was selected using retrospective data.

The final refit uses 18,595 unique classifier-eligible observations; its predictor uses a 13,891-observation subset. These counts overlap and are not added. The two branches expose 152 named feature entries. Selected model SHA-256: 05e4679c7f66df7eff8dfa348eb2146972f5424c92713a25117d65444caa5085. Report SHA-256: a41a182cda1fd06c964c0ea4054a9f7fd041ba67cc48ffdddf82a8af9a47beb0.

Eight tests pass, including an example where an earlier residual alarm suppresses a later classifier onset, finite-score gating, gap/prefix/label independence, baseline preservation, source/fold integrity, actual fitted branch reconciliation, and budget exploration reproducing saved OOF metrics without changing active or prior artifacts. This remains development evidence; root owns any later single retrospective audit and release decision.
