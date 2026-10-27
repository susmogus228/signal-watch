# Frozen evaluation and reproducibility

## Result

The trained model is real, but the learned detector is not a successful operational monitor at its frozen threshold. **97.63% of observed normal time is in alert.** The corrected onset-based event recall is also low. SignalWatch retains that failure and the competing baseline results instead of reporting only a favorable overlap percentage.

| Method | Detected / segments | Missed | False episodes / normal hour | All scored time in alert | Normal time in alert | Literal point precision | Mean detected delay |
|---|---|---|---|---|---|---|---|
| ExtraTrees residual | 5 / 28 | 23 | 6.863 | 98.42% | 97.63% | 33.66% | 31.80 s |
| Rolling median/MAD | 4 / 28 | 24 | 0.624 | 0.097% | 0.040% | 73.08% | 62.75 s |
| Persistence | 8 / 28 | 20 | 17.885 | 1.208% | 0.659% | 63.70% | 57.00 s |

These are 22 disturbed test recordings plus a separate held-out normal block. Scored normal exposure is 17,311 seconds, or 4.809 hours; total scored observed exposure is 25,906 seconds. The false-episode onset numerators are 33, 3 and 86 respectively. Mean ordinary per-recording average precision is 0.8211, 0.3719 and 0.3905 respectively. Single-class recordings are excluded from this secondary average, and it is not pooled PR area. It does not override the alarm-burden failure.

The full finite JSON report, per-recording events, exact definitions, thresholds, model hash and group slices are in [frozen_evaluation.json](../artifacts/frozen_evaluation.json). Validation-only threshold exploration does not overwrite it.

## Frozen definitions

- An annotated segment is consecutive anomaly=1 samples, split at a timestamp gap. A detected segment requires a new alert episode START within its boundaries. An alarm already active before onset does not detect the new event. No point adjustment fills the rest of the segment.
- An alert episode is a consecutive sequence of above-threshold finite scores. There is no smoothing, merge, cooldown or multi-sample trigger. A gap ends an episode and resets history. An episode touching the visible prefix endpoint is ongoing.
- A false episode starts outside an annotated anomalous interval. Later overlap with an annotation does not change that onset into a valid detection. Its normal alert duration also contributes to normal_alert_fraction.
- Delay is the elapsed timestamp difference from annotated onset to the first new alert-episode onset within the annotation. Misses have no fabricated delay. Mean/median delay includes detected segments only.
- Exposure is the actual interval from a sample to the next sample, assigned to the left sample's label and alert state. Gap-crossing intervals and unscored warm-up contribute no eligible time. The terminal sample contributes zero seconds. No missing interval is interpolated.
- False episodes/hour uses scored observed normal seconds as denominator. Total observed normal seconds before warm-up exclusion are reported separately.
- Point precision and average precision use literal finite scored sample labels; no retrospective filling is applied. Time fractions use timestamp durations, not sample percentages.

## Calibration

The requested frozen budget is 2 false episodes per scored normal hour. A fixed set of 101 score quantiles from 0.90 to 1.00 plus a threshold above the maximum is evaluated explicitly, because episode count is not monotone in threshold. The lowest threshold among that fixed set satisfying the empirical rate is selected independently per method. All three selected operating points have zero calibration episodes in 2,399 scored seconds. This short exposure cannot establish a reliable future false-alert rate. The report shows the 1.50 episodes/hour single-episode resolution and actual test burden.

## Run-family slices

Family slices are saved for other, valve1, valve2 and anomaly-free material. Learned normal-alert fraction is 100% in each disturbed-run family and 78.48% in the separate normal audit. This is coverage evidence for one testbed, not demographic fairness or independent-equipment validation.

## Measurement correction history

The original overlap-based event report is retained as [frozen_evaluation_legacy_overlap.json](../artifacts/frozen_evaluation_legacy_overlap.json), SHA-256 4f8ba3e8a9fc8031e9d9d78cede65b03e9730efa084f863b5febfbc8ee420699. It wrongly credited already active alarms as fresh detections and excluded episodes beginning before an anomaly from the false count. Schema 2 fixes that logic while preserving the exact model, thresholds, split and original validation selection. Candidate selection reproduction still explicitly uses the historical overlap objective; it does not select a new winner after seeing final-test errors. Only event matching/counts/delays change; scores, time burden, precision and average precision remain identical. The corrected final learned count is 5/28, not the superseded 25/28.

## Reproduce

From the project root, after installing the project-local dependencies:

```powershell
.venv/Scripts/python.exe scripts/prepare_data.py
.venv/Scripts/python.exe scripts/train.py
.venv/Scripts/python.exe -m pytest tests/test_dataset.py tests/test_ml.py tests/test_evaluation.py -q
```

The existing split/config manifest must match before training can proceed. New uploads are never used. Fixed model randomness and source hashes support reproduction; wall-clock duration and UTC metadata creation time naturally vary. The final test is never used to select parameters. A fresh training job reproduces the predefined study, not an optimization against its frozen report.

## Verification performed

Twelve domain tests pass: strict ingestion and mapped headers, binary labels, finite numbers, row/byte limits, timestamp order, causal cadence under changed future timing, score-prefix invariance using both a controlled predictor and the real fitted artifact, label-feature separation, gap warm-up reset, empty/short prefixes, episode-onset delay and physical exposure, label-free metrics, continuous-alert burden, a pre-existing alarm followed by a genuinely new onset, and validation comparison preserving frozen bytes. Hash verification rejects changed bundled data or model files. API, persistence and browser integration are verified separately by their owners.
