# ML v2 development protocol

User request, 2026-10-23: improve the detector and aim for at least 25 of the existing 28 annotated segments. This is an empirical target, not a guaranteed result. Preserve onset-based event matching and control false alarms.

## Diagnosis and sources

The independent train/calibration/development-only inspection found a normal-regime mismatch: training median flow 124.668 L/min versus 32.9574 on normal development material. The v1 model predicts absolute levels; its residual scales come from in-sample errors. All 8,392 scored normal development samples exceed the learned threshold. A second structural limit is the 30-sample warm-up: 16 of 29 development annotations contain no scored sample after gaps.

Sources checked before implementation:

- Official [SKAB repository](https://github.com/waico/SKAB): measured sensor data, distinct experiments, anomaly/changepoint tasks and benchmark protocols. The published leaderboard uses different metrics and cannot establish our 25/28 result.
- [Scikit-learn grouped evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#group-k-fold): hold out entire groups; fit preprocessing on training material. Context7 documentation checked for installed scikit-learn 1.9.1.
- [Kim et al., AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20680): point adjustment can inflate detection results, including random scores. Keep literal alert onsets and burden alongside recall.

## Selected approach and alternatives

Use a normal-only ExtraTrees innovation predictor across several development operating regimes. Predict the present sensor values relative to past context; fit translation-invariant features. This keeps a normal-behavior prediction task and the existing local Python/scikit-learn stack. A deeper absolute-level forest would retain the regime problem. A supervised normal/anomalous classifier is a separate potential study, not this implementation.

Files: new signalwatch/ml_v2.py and scripts/study_v2.py; new tests/test_ml_v2.py. Root owns dispatch/integration changes in ml.py, API metadata and documentation. Development output goes to artifacts/v2-development/ and scratch to work/v2/. Preserve original artifacts and source snapshots under artifacts/v1/. No remote publication or global installation.

## Frozen selection boundary

- Development material: original normal training block and the 12 original validation recordings. Development labels may identify all-normal training windows and evaluate candidates; they are not inference features.
- Hold out entire recording groups for out-of-fold development scoring. Do not randomly mix overlapping windows across folds. Every training target and its complete historical window must be normal and gap-free.
- Normal centering/scaling comes from fold training material. Use the independent original normal calibration block for residual normalization, not in-sample prediction errors.
- Predeclare the bounded grid before running it: history 2, 5 or 10; 64 trees; minimum leaf 4 or 12; deterministic seed. Use identity smoothing (1 sample). The development plan JSON must freeze exact recording IDs in three folds, the threshold quantile grid, tie-breaks and no-feasible-candidate behavior before first scoring. No arbitrary periodic alert resets or pulses.
- Select thresholds and candidate on development out-of-fold results. Require normal-alert fraction <= 5% and false onset rate <= 2 per observed normal hour. Maximize onset_v2 event recall among feasible candidates, then lower burden, then declared order. If none passes, report failure.
- Preserve the existing baselines and their 30-sample context. Report each method's actual scored exposure and warm-up; do not hide the difference in score availability.
- Report per-recording/family burden, numerators and exposure. Include a common-availability comparison and the number of events made scorable by shorter warm-up. Preserve original alert onsets when masking availability; do not turn a mask boundary into a new detection.
- Do not inspect the old test results to select a candidate or threshold. Score the chosen model on the old test only after selection. These 28 annotations have already been examined in v1, so label the result retrospective audit. It is not a new blind test.

The benchmark cannot prove new-equipment or production usefulness. No new private data, external inference, participant collection or deployment enters this study.

## Memory and agents

Codex memory supplies evidence/scope rules, not current metrics. Fresh Codebase Memory listing has no SignalWatch index; inspect source, no indexing. AI Engineer owns the new isolated ML module, study script, tests and development artifacts. Root owns v1 preservation, integration, shared documentation and API/UI consistency. Model QA Specialist independently reviews causality, split isolation, onset matching, selection and reported results after the implementation. No reviewer edits implementation files.

## Implementation and verification

- [x] Archive and hash v1 model/report/manifests/source. Preservation manifest: artifacts/v1/preservation_manifest.json.
- [ ] Write failing tests for prefix, gap, label and translation invariance; implement v2 and run the predeclared development study.
- [ ] Review candidate feasibility and select without old-test results. Independently review model/data/evaluation boundaries.
- [ ] Score the selected candidate on the existing 28 annotations once; report event hits, delay, normal-alert fraction and false-onset rate. Keep unmet targets explicit.
- [ ] Integrate the justified model result, training/replay/budget semantics and English cards/UI. Preserve v1 comparison/history.
- [ ] Run backend and relevant frontend checks, actual API/browser replay, model metadata and report reconciliation. Record completion and limits in a v2 verification report.

## Outcome

The normal-only study finished: grouped OOF 20/29, retrospective 11/28. It fails the target and retrospective false-onset limit. V1 remains archived. Subsequent user-approved supervised studies and the active V4 result appear in ML_EVALUATION.md; 25/28 remains unmet.
