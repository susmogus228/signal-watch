# V12: frozen-design normal-regime feature study

Date: 2026-10-27. User explicitly approved a learned normal-regime stage plus trend classification, up to twenty estimator fit calls, with existing data/dependencies and no alarm delay. This protocol refines the [approved proposal](ML_V12_NORMAL_REGIME_PROPOSAL.md) before any actual study fit.

## Question, controls and boundaries

Test whether learned distances from labelled normal regimes improve the joint progressive-event and false-alert screen. Use Python 3.11/scikit-learn 1.9.1, the original twelve development recordings, three original whole-recording folds and the original fit-only normal anchor. Preserve V10/V11 sources/artifacts and all active V5 files. No live model adaptation, data collection, new packages, API/frontend edits, deployment or Git push.

Compare K1, K2 and K4: GaussianMixture with one, two or four diagonal components, then V10 B's original 15-leaf HGB. K1 controls for a single normal population. Saved V10 B, V11 and full V5 controls require no new fitting. This changes the representation by adding a supervised-normal transform; it does not reproduce a published paper algorithm or isolate every consequence of bank balancing/extra features.

## Training-only normal bank and mixture

Use only each outer training recording's eligible anomaly=0 rows, including the original normal anchor slice [0,5172). Eligibility is exactly V10's two-row gap-segment mask. Read the eight current sensor values at those raw row indices. No held-out record, timestamp, ID or row position is an input feature. Normal is the benchmark label, not independent proof of physical safety.

Let I_r be each training record's sorted eligible-normal raw indices and n_r their lengths. Set m=min(512,min_r n_r). If a training record has no eligible normals or m<2, invalidate that whole K arm and record the reason; do not omit a record. For each record take I_r[floor(linspace(0,n_r-1,m))], including both endpoints. Require exactly m distinct raw indices per record, no replacement. Record all selected indices, per-record counts and source hashes. The normal anchor has exactly the same bank count as each disturbed training record.

From this balanced bank only, compute per-channel center=median and scale=max(Q75-Q25,1e-6). Standardize the eight columns. Fit GaussianMixture with n_components=K, covariance_type='diag', tol=0.001, reg_covar=0.001, max_iter=200, n_init=1, init_params='kmeans', random_state=20261023, warm_start=False; retain other installed defaults in evidence. No sample_weight is passed: the installed fit API does not accept it. Limit numeric CPU threads to two.

Record converged_, n_iter_, lower_bound_, component weights/means/variances and exact source/preprocessor identities. Any nonconverged or non-finite outer mixture invalidates the entire K arm, not just one fold. Do not screen partial OOF arrays, drop bad folds, retry, increase iterations or replace parameters. A failed invocation counts within the bound; skip that arm's remaining fits after invalidation. Unexpected implementation errors stop the one-shot execution for investigation, without deleting its marker or silently restarting it.

## Exactly 198 classifier features

Keep V10's 186-feature causal representation unchanged. Append twelve features from the current standardized sensor vector z, fitted diagonal component means mu_k/variances v_k and posterior q_k:

1. -GaussianMixture.score_samples(z), the negative log density.
2. min_k D_k, where r_kc=(z_c-mu_kc)/sqrt(v_kc) and D_k=sum_c r_kc^2.
3. sum_k q_k D_k.
4. max_k q_k.
5. Eight signed residual features sum_k q_k r_kc, in the original sensor order.

These definitions are invariant to permutation of mixture components. No component index is an input feature. Regularized variances must be finite and positive; no adaptive clipping or zero replacement of invalid values. Keep full-length score availability exactly equal to the original eligible mask. Partial 30/120-second histories and causal gap resets remain unchanged.

The original 186 column center/scale values must exactly match V10 B for the same training IDs. For the twelve appended columns only, compute center=median and scale=max(Q75-Q25,1e-6) over all eligible classifier training rows. Concatenate both preprocessing vectors. All mixture/scaler parameters are fixed at inference; no test-record normalization or future-row fitting occurs.

Fit HGB exactly as V10 B: max_iter=150, max_leaf_nodes=15, min_samples_leaf=20, learning_rate=0.05, l2_regularization=1, early_stopping=False, random_state=20261023. Use the unchanged equal-record/class-balanced supervised sample weights. No hard-normal mining, additional fit weights, score calibration or novelty-dependent correction.

The normal-bank transform is fitted on classifier-training normals; that is a legitimate training-only pipeline, not held-out label leakage. It can nevertheless encode in-sample density fit versus new-record novelty. Save train-normal and held-out-normal negative-density/minimum-distance count/min/Q25/median/Q75/max summaries as descriptive diagnostics. Those summaries add no selection criterion or fitted correction.

## Bounded fitting, decisions and selection

Three K arms times three outer folds yield at most nine mixture and nine classifier fit calls. Internal mixture initialization is recorded separately. A single qualifying final winner permits one more mixture and classifier fit: at most twenty top-level estimator fit calls. Save attempted/completed/skipped counts and invalid-arm reasons. Synthetic unit-test estimator calls are separate from this study count.

Immediate decision: classifier probability strictly above a positive selected cutoff. No waiting, smoothing, cooldown, backfill, V9 policy or residual OR branch. Probability is not a calibrated physical-fault probability.

For each complete valid arm, scan the same linspace(0.90,1.0,101) quantiles of normal OOF probabilities plus nextafter(maximum normal,+infinity); deduplicate positive finite cutoffs. At most 306 points total. Reuse the unchanged onset_v2 evaluator and require exactly 9,179 observed normal seconds. Require >=26/29 events, fewer than three false starts, <=2 false starts per normal hour, <=37 normal-alert seconds, and retention of both V5-detected source-described progressive intervals other-6-event-0 and other-9-event-0. No label changes, new windows or annotation exclusions.

Rank feasible points by more events, fewer false starts, fewer normal-alert seconds, lower median/mean detected-event delay, K1/K2/K4 order and threshold order. Preserve every point, rejection, per-record/family/progressive outcome, lost/gained event pair and probability diagnostic. Repeated development comparison increases selection-overfit risk; no fresh-validation claim follows.

If no arm qualifies, selected=null, no final estimator, no old-audit/teaser inference and no activation. Do not fall back to a descriptive frontier point. If one qualifies, freeze its exact K/cutoff/OOF/candidate/plan identities before one final mixture and HGB fit using all twelve development records plus the original anchor. Final nonconvergence rejects the candidate without retries or another winner. Preserve the pre-final point as development_selected, set selected=null/status='final_fit_rejected'/final_model_available=False, retain the rejection evidence and counters, and write no model artifact or model SHA. The pre-final selection file remains unchanged.

## Conditional diagnostics and artifact contract

Use artifacts/regime-development/ for plan, execution marker, report, all candidate audits, per-arm OOF arrays, pre-final selection and conditional selected_bundle.joblib. Bundle identity includes arm K1/K2/K4, threshold, feature_names with exactly 198 entries, plan_sha256 and selection_before_final_fit_sha256, plus all fitted transforms/estimators. classifier_scores(bundle,recording) returns the same dictionary contract as V10, with ['scores'] as the full-length classifier probability vector and NaNs at unavailable rows.

Only a sole qualifying final candidate may enter the existing conditional diagnostic runner, after its own source/model/dependency freeze and independent QA. Score it once on each old-audit recording and once on the already-inspected teaser; no further candidate/cutoff/model correction follows those outcomes. Both are reused retrospective diagnostics. Preserve original normal exposure, source-envelope conventions and assumed-background limitations. Active V5 remains unchanged during the study. A release replacement needs a separately recorded integration/release check.

## Ownership, freeze and verification

AI Engineer owns only signalwatch/ml_regimes.py, tests/test_ml_regimes.py, scripts/study_regimes.py and artifacts/regime-development/. Root owns protocols, shared documentation, conditional diagnostic helper/tests/outputs and evidence. No editing of frozen V10/V11 scientific files or existing scientific model/evaluator/data implementations.

TDD covers exact sampling and empty/too-small banks, training exclusion, component-permutation features, degenerate channels/non-finite mixtures, convergence invalidation, 198 feature/scaler dimensions, genuine fitted predictions, prefix/gap/label/ID invariance, immediate crossing behavior, no-feasible selection and correct ranking. Root extends final-model identity/path dispatch tests before any conditional inference. Model QA reviews protocol/code then verifies actual frozen dependencies before the sole fit run, and reconstructs saved selection/provenance afterward without fitting or scoring models. Run one full backend suite after source changes.

The study plan hashes this protocol, all source/tests/scripts and their scientific dependencies, data manifest/source notices, original controls and all preserved V10/V11 plan/report/OOF/candidate/selection artifacts. Store the actual freeze before fitting; verify all dependencies and six active V5 files at completion. Tests prove implementation behavior, not the model-quality requirement.

Primary-source and exact-version API checks appear in the approved proposal and RESOURCES.md. Neither normal-mode physics nor a numerical gain is assumed from those sources.
