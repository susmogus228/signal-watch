# Supervised detector study

User approval, 2026-10-23: compare a supervised detector if normal-only v2 does not achieve the required quality. The task changes to recognizing known annotated abnormal states; this does not establish unknown-fault detection. Keep the original event definitions, raw records and baselines.

## Protocol before scoring

Use exactly the same 12 development recordings, three recording-held-out folds and normal-train anchor as the frozen normal-only v2 plan. Fit supervised preprocessing/model on fold training recordings only. Development labels can train the classifier and select its threshold. Every held-out recording remains outside model/scaler fitting. Do not use the old test to choose features, candidates or thresholds.

Use causal sensor features only: current channel values, previous differences, historical rolling mean/standard deviation at 2/5/15/30 past samples using the available prefix after gaps, current-minus-past means and a two-sample initial reference deviation. No filename, ID, label, timestamp, elapsed time or row index feature. Require two historical samples before classification, reset rolling history at gaps. Document reference behavior exactly. Scores are uncalibrated classifier scores, not failure probabilities.

Freeze four candidates before scoring: ExtraTreesClassifier with 128 trees and minimum leaf 4 or 12, and HistGradientBoostingClassifier with max_iter=150, max_leaf_nodes=15 or 31, min_samples_leaf=20, learning_rate=0.05, l2_regularization=1 and early_stopping=False. Use deterministic randomness and bounded CPU threads. No random internal validation of overlapping windows.

Keep the fixed normal quantiles 0.90..1.00 (101 points) plus above-max threshold from v2. Select candidate and threshold on grouped OOF development results: maximize onset_v2 recall subject to <=5% normal time in alert and <=2 false onsets per observed normal hour; ties lower normal-alert fraction, lower false-onset rate, candidate order, threshold order. Preserve original onsets; no smoothing, cooldown, periodic pulsing or point adjustment. Freeze exact folds, candidate parameters, feature schema and tie-breaks in a study plan JSON before scoring.

Save per-recording and family slices, exposure/numerators and common-baseline availability. Fit the chosen candidate on development only, freeze it, then root may perform one retrospective audit on the old 28 annotations. Clearly distinguish development selection, retrospective audit and any future independent data.

## Ownership, paths and checks

Keep the existing local Python/scikit-learn stack and API shape. AI Engineer owns new signalwatch/ml_supervised.py, tests/test_ml_supervised.py, scripts/study_supervised.py and artifacts/supervised-development/. Root owns active integration, retrospective audit and documentation. Model QA Specialist performs independent read-only causality/split/metric review. No concurrent edits to shared surfaces.

TDD checks: future-prefix invariance, label/ID/time independence, gap/reset/two-sample eligibility, training-group isolation, class probability range, feasibility selection and unchanged baseline arrays. Preserve v1 and normal-only v2; no active artifact replacement until root reconciles results. No installation, private-data transfer, deployment or Git push.

Target: at least 25/28 existing annotated segments, while reporting false alerts and time burden. The target may remain unmet. No manufacturing of labels or claimed independent-test accuracy is allowed.


## Bounded weighting follow-up

A second supervised study changes training sample weights only. Its four configurations and exact formulas are frozen separately before scoring in `artifacts/supervised-weighted-development/plan.json`. Both procedures retain 120 causal features, the same grouped development folds, onset matching and alert-burden limits. The release choice rule is fixed before weighted results: maximize selected grouped-OOF event hits, then minimize normal-alert fraction, then false onsets/hour. Exact ties retain the first supervised study. Previously examined test metrics do not enter this choice. This remains repeated development selection and does not create an independent test.
