# V11: bounded learner comparison on the frozen trend representation

Date: 2026-10-27. User scope: improve actual trained ML for progressive defects and false alerts. Preserve the completed negative V10 experiment. No change to labels, cutoffs after evaluation, temporal confirmation, smoothing or V5 active artifacts.

## Reason for a separate follow-up

V10's 15-fit experiment improved classifier AP/proper losses, but no point retained the progressive cohort with fewer than three false starts and the required overall recall/burden limits. At B's 26/29, one-false-start point, the linear other-6 interval is missed. A progressive-preserving 27/29 point has five false starts, all in held-out valve recordings. Within other-6, B separates normal and anomalous scores; cross-record high-confidence normal errors force the shared low-false cutoff above its anomaly scores. This is post-hoc development diagnosis, not a proof that capacity or a different learner will solve transfer.

The within-record distinction concerns ranking and medians, not perfect class separation: B's other-6 normal/anomalous medians are0.001278/0.963807, while the observed score ranges overlap. We therefore keep the exact representation/training data and test two existing scikit-learn tree learners with different biases. This is a new, predeclared experiment, not an amendment that turns V10 into a passing run. Repeated development use and multiple model comparisons increase selection-overfit risk.

## Exact models and computation

Reuse the frozen V10 causal feature builder: 186 columns, past-only actual30/120-second slope/net/direction/R-squared plus two coverage values and original120basefeatures. Keep the same two-row eligibility, partial history, constant-channel conventions, gap resets and train-only unweighted median/IQR preprocessing. Keep V10 B's equal-record/class-balanced training sample weights and the fit-only normal anchor. No hard-normal mining in this study; V10 C did not improve its relevant frontier.

Two fixed learners, seed20261023 and at most two CPU threads:

1. HGB31: HistGradientBoostingClassifier, max_iter150, max_leaf_nodes31, min_samples_leaf20, learning_rate0.05, l2_regularization1, early_stopping=False. The only estimator change from V10 B is leaf capacity15 to31.
2. ExtraTrees256: ExtraTreesClassifier, n_estimators256, min_samples_leaf4, min_samples_split2, max_features1.0, criterion='gini', max_depth=None, bootstrap=False, class_weight=None, n_jobs2, random_state20261023. Apply the same explicit training sample weights and preserve the locked version's other defaults in fit evidence. This learner already exists in the project; no new dependency or stack is introduced.

Use exactly the original three outer recording-disjoint folds, eight disturbed training IDs plus the normal anchor per fit. Outer-held records enter neither fitting, preprocessing nor training weights. Six development fits total. Save both OOF arrays, fit counts/rows/scalers and source/dependency identities. Reuse saved V10 B186, original A120 and V5 fusion controls without refitting them. HGB31 versus V10 B investigates leaf capacity; ExtraTrees256 versus B also changes the learning algorithm. Selected alarm outcomes additionally include development operating-point selection.

## Unchanged primary screen and ranking

Immediately alert when the raw classifier class1 probability exceeds its selected positive cutoff. No innovation OR branch, V9 policy, cooldown, backfill or forced window wait. Probability scores are not calibrated physical-fault probabilities.

Use the same predefined normal-OOF quantiles linspace(0.90,1.0,101) plus nextafter(max normal,+infinity), unique positive finite cutoffs; at most204points. Retain the unchanged onset_v2 evaluator and exactly9,179 observed normal seconds. Require >=26/29 events, fewer than3false starts, <=2false starts/normal hour, <=37normal-alert seconds and all V5-detected intervals in source-described linear/exponential other-6/other-9 retained. No reinterpretation of labels or relaxation after outcomes.

Rank feasible points by most detected events, fewer false starts, fewer normal-alert seconds, lower median/mean detected-event delay, HGB31 before ExtraTrees256 and threshold order. Save every candidate/rejection, per-record/family/progressive metrics and lost/gained events. Retain proper-score and ranking diagnostics separately. If none qualifies, report no feasible candidate and do not refit or score an alternative final model on old28/teaser.

## Conditional final model and diagnostics

For one qualifying winner only, freeze the exact learner/cutoff/OOF/candidate identity before fitting all twelve development records plus the original anchor. One additional final fit; maximum seven fits in this study. Freeze model SHA and source identities before inference. The V10-style final diagnostic runner must verify file SHA, semantic learner/cutoff, plan/selection identities and feature count before any prediction.

Run only this final model once on the old28 corpus and once on the already inspected teaser, as explicit reused regression/diagnostic checks. No alternative learner, threshold or annotation convention follows their outcomes. Compare old recall with >=25/28, false-start rate with<=2/h and normal-alert seconds with<=272. Teaser uses the unchanged four source envelopes, uncertain transitions and assumed-background definitions. Both corpora are already seen; neither is new independent validation. Keep V5 active until a separate release/integration check; do not claim field reliability from a few same-testbed recordings.

## Ownership and checks

New worker files: signalwatch/ml_trend_backbones.py, tests/test_ml_trend_backbones.py, scripts/study_trend_backbones.py, artifacts/trend-backbones-development/. Root owns this protocol, documentation and the conditional diagnostic runner/output. Earlier V10/V9 and other scientific source files stay unchanged.

TDD verifies exact learner configuration, record exclusion, real fitted-estimator prediction/availability, same preprocessing/weight mass and no-feasible/progressive/ranking behavior. Model QA checks protocol/code/dependency freeze before fitting, then independently reconciles saved arrays/metrics/selection without retraining. Run one final full backend suite after current changes. No frontend/API edits, new global packages, Git push or deployment.

ExtraTrees prediction with two worker threads can change floating-point addition order by about1e-16. Same-input probability invariance tests use absolute tolerance1e-12 and zero relative tolerance; feature arrays and finite masks remain exact. Alarm comparisons remain strictly above the measured cutoff, without rounding or a decision tolerance. Values on numerical threshold/tie boundaries can therefore be sensitive to floating-point reduction order; retain this reproducibility limit.

## Sources

Reuse V10's primary interval-feature and hard-example research with its documented limits. For these estimators, check the installed scikit-learn1.9.1 [ensemble documentation](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/ensemble.rst); do not attribute published benchmark gains or a new algorithm to this project. data/SKAB-DATA-README.md defines the progressive cohort. V10's complete preserved report documents the diagnosis motivating this bounded follow-up.
