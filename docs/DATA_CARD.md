# SKAB data card

## Source and permitted scope

SignalWatch bundles the public SKAB water-circulation testbed snapshot [b2c0d46c2971dcbfe71e26087b6d231998bb91c2](https://github.com/waico/SKAB/tree/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data). The upstream GPL-3.0 notice is retained as [SKAB-LICENSE](../data/SKAB-LICENSE); the original repository and experiment descriptions are retained in data/. No project-source license is selected by this data notice. These are physical testbed recordings, not generated sensor curves or production equipment data. No participant data was collected.

The [machine-readable manifest](../data/manifest.json) records every relative path, source URL, SHA-256, byte count, rows, nominal cadence, timestamp span, labels and gaps. scripts/prepare_data.py downloads only these pinned official resources. Existing raw files are retained; application loading verifies their recorded SHA-256 before use.

## Inspection facts

- 35 CSV files, 4,392,581 original CSV bytes, 46,806 rows.
- One anomaly-free recording has 9,405 rows. The other 34 files are disturbed recordings with anomaly and changepoint labels.
- Eight finite numeric sensor channels: two RMS accelerometers (g), motor current (A), pressure (bar), motor and liquid temperatures (degrees Celsius), voltage (V), and RMS flow rate (L/min).
- All recordings establish a one-second nominal cadence from their first 30 timestamp intervals. Some actual intervals are two seconds; time is never replaced by row count.
- 40 intervals exceed three times nominal cadence. The largest is 247 seconds. These gaps reset all detector histories and are excluded from observed exposure.
- No missing/nonfinite sensor values or duplicate/nonincreasing timestamps were accepted in the pinned files.
- The upstream textual folder example is not treated as an exact inventory; the pinned Git tree and downloaded CSVs determine the manifest.

## Original V1 split and current roles

The split is defined before fitting/scoring in [split_manifest.json](../artifacts/split_manifest.json). A recorded experiment never appears in both disturbed-run validation and final test.

| Material | Half-open indices or rule | Purpose |
|---|---|---|
| Normal training | [0, 5172) | Scaling, predictor fit and training-only residual scales |
| Purge | [5172, 5202) | 30 samples excluded |
| Normal calibration | [5202, 7524) | Threshold selection, no predictor fitting |
| Purge | [7524, 7554) | 30 samples excluded |
| Normal audit | [7554, 9405) | Held-out normal exposure, virtual ID normal_audit |
| 12 disturbed validation runs | Numeric recording ID divisible by 3, within each source group | Choose one of two predefined model candidates |
| 22 disturbed test runs | Other numeric IDs | Frozen audit only |

V1 and both unchanged baselines use thirty history observations. V5 starts learned scoring after two observations per gap segment. The twelve disturbed development runs now provide supervised training and grouped OOF selection, disjoint from the retrospective audit. The calibration block has 2,322 rows spanning 2,431 seconds: 2,211 one-second intervals plus 110 two-second intervals. Its scored exposure is 2,399 seconds after warm-up. A gap-aware label interval may split one physical experiment into multiple evaluable segments; the final 22 disturbed runs contain 28 such segments. Segment count is not a count of independent machines or independent failures.

## Local imports

CSV imports remain local. An import must contain a consistent ISO timestamp column and all eight numeric channels, either named exactly or explicitly mapped. Optional anomaly/changepoint labels must be binary. Limits: UTF-8, 5 MiB, 100 to 100,000 rows, unique headers, finite numbers, and strictly increasing timestamps. Labels are never inference features. Supervised fit uses labels from the twelve bundled development recordings; uploaded labels serve audit only. Missing labels mean unknown detection quality; the application does not invent normal labels. Cadence is preloaded from the first 30 timestamp intervals of the complete CSV before recorded replay. An unseen timestamp inside that calibration block can change an early gap decision. Prefix invariance therefore assumes this known recording cadence; initialization of an unknown raw stream is not evaluated here. Uploaded values are not included in the bundled training study.

## Limits

This is one testbed with narrow operating conditions. Chronological normal blocks are not different equipment. Run-disjoint evaluation reduces window overlap leakage but does not establish equipment transfer. No demographic attributes are present; demographic fairness is not measurable here. Group slices by experiment family are reported to expose coverage differences. Anomaly annotations describe dataset context; model residuals cannot establish a physical failure cause.
