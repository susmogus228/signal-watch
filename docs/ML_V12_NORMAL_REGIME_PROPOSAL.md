# V12 proposal: learned normal regimes plus trend classification

Date: 2026-10-27. **Architecture approved by the user on 2026-10-27; not a measured result.** The user approved the normal-regime model plus trend classifier, up to twenty estimator fit calls, without delay or new dependencies. The exact [execution protocol](ML_V12_NORMAL_REGIME_PLAN.md) incorporates independent review clarifications before freezing. No data collection, paid services or external inference are proposed.

## Problem and hypothesis

V10/V11 improve development ranking/probability losses, but no candidate preserves both progressive intervals and the overall recall/burden limits. Normal scores in some held-out valve records exceed anomalous scores in the linear-imbalance recording. That supports investigating cross-record representation; it does not prove physical mode changes caused every false alert.

Hypothesis: distance from several learned normal regimes gives the classifier more useful information than absolute sensor levels and time trends alone. Normal means anomaly=0 under the existing benchmark labels, not independently verified equipment safety. An unseen healthy mode can still trigger an alert; a fault close to a known mode can still be missed.

## Proposed bounded design

Use the existing Python/scikit-learn 1.9.1 stack. In each original whole-recording development fold, learn GaussianMixture models from training-only normal sensor vectors, then append their normal-regime distance features to the frozen 186-feature trend representation. Fit the original V10 B HGB configuration on those features and the unchanged recording/class-balanced labels. No waiting, smoothing, temporal alarm confirmation, per-record cutoff or use of held-out labels at inference.

Compare K=1, 2 and 4 diagonal Gaussian components. K=1 is the single-normal-population control. Fix reg_covar=0.001, tol=0.001, max_iter=200, n_init=1, init_params='kmeans', random_state=20261023 and warm_start=False. A nonconverged model makes its arm invalid, without retrying new parameters. Preserve convergence/iteration/fit metadata. The mixture is trained offline; no live adaptation or automatic incorporation of anomalous observations is allowed.

GaussianMixture.fit has no sample_weight argument in the installed version. Balance its normal bank explicitly: take m=min(512, the smallest eligible normal count across training recordings), deterministically and evenly spaced by raw row order from each record, without replacement. Include the fit-only normal anchor. Freeze the index-selection rule and sensor scaling before execution. Normalize eight sensor columns using that bank's median/IQR, with an explicit 1e-6 scale floor. Keep every source index and source hash in the report. No held-out record enters mixture fitting, scaling or bank selection.

Append twelve same-width features for every K: negative log density, minimum squared component Mahalanobis distance, posterior-weighted squared distance, maximum component posterior, and eight posterior-weighted signed standardized residuals. Retain the original 186 causal features, including trend and history coverage. Standardize appended features only from classifier training rows, with an explicit scale floor. Mixture and classifier transforms at inference use only the current/past sensor data and fixed fitted parameters; labels, record IDs and future rows are excluded.

The classifier remains HGB with 150 iterations, 15 leaves, minimum leaf 20, learning rate 0.05, L2=1, seed 20261023 and early_stopping=False. Use the unchanged V10 B preprocessing on its original columns and equal-record/class-balanced supervised weights. This tests learned normal-regime features; it does not independently isolate every effect of bank balancing or extra features.

Three arms times three outer folds yield nine mixture and nine classifier fit calls, with mixture initialization performed internally. Only a qualifying winner permits one additional mixture and one classifier fit on all development data: at most twenty top-level estimator fit calls. Restrict CPU threads to two. No open-ended hyperparameter search or outcome-driven rerun.

## Acceptance, diagnosis and stopping

Freeze source/dependency hashes, folds, component counts, fit parameters, feature definitions and bank indices/rule before execution. Use the same 101 normal-OOF quantiles plus the above-maximum cutoff, at most 306 distinct candidate points. Require at least 26/29 events, fewer than three false starts, at most 2/h, at most 37 normal-alert seconds over the same 9,179 normal seconds, and retention of both original progressive intervals. Rank feasible points by hits, false starts, alert time, median/mean delay, then K and threshold order. Preserve all rejected points and report baseline controls from saved V10/V11/V5 arrays.

If no arm qualifies, stop with no final model and retain V5. If one qualifies, freeze its identity before the two final fit calls and run only that winner once through existing conditional old-audit/teaser diagnostics. Both corpora have already been inspected; these remain reused diagnostics, not fresh validation. Active replacement requires its own integration/release check. No relaxation, relabelling or threshold rescue follows outcomes.

Synthetic tests must cover train/held-out exclusion, exact bank balancing, degenerate channels/mixtures, finite features, prefix/gap/label/ID invariance, immediate decisions, saved identity guards and no-feasible behavior. Independent Model QA reviews the frozen design before fitting, then rebuilds candidates and provenance from saved arrays without refitting. Root owns protocol/docs/diagnostics; the AI Engineer owns only new model/script/tests/artifacts. One final backend suite follows accepted source changes. No API/frontend changes, publication or global installation are included in this research proposal.

## Sources and limits

- [Yu and Qin, 2008](https://aiche.onlinelibrary.wiley.com/doi/10.1002/aic.11515): publisher abstract describes multimode monitoring with finite Gaussian mixtures and Bayesian inference. We inspected the abstract, not the complete paywalled method; the proposed fixed-K feature pipeline does not reproduce its Figueiredo-Jain procedure or published results.
- [Tan et al., 2020](https://www.sciencedirect.com/science/article/pii/S0959152419304482): the accessible article motivates distinguishing multiple healthy modes from faults; its online update procedure requires confirmation of a new healthy mode. This proposal implements neither its DP-GMM/KPCA method nor automatic online updates.
- [scikit-learn 1.9.1 GaussianMixture](https://scikit-learn.org/1.9/modules/generated/sklearn.mixture.GaussianMixture.html): constructor, fit, predict_proba and score_samples checked against installed signatures. Context7 was consulted but returned non-specific mixture examples; the exact versioned official API and installed signatures are the fallback evidence.

These sources justify a testable hypothesis, not an expected numerical improvement. Existing SKAB provenance/licensing and same-testbed limitations remain applicable. No paper text, code or figures are copied.
