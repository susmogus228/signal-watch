# V8 controlled development ablation

Status: methodological protocol accepted by pre-fit Model QA review, before fits. Owner approval: 2026-10-23, continue controlled component comparison and preserve version evolution. The runner must freeze this document and implementation hashes before fitting. This study diagnoses V7; it does not activate a replacement model.

## Question and controls

V7 changed the classifier training subsets, sigmoid calibration, probability averaging and residual cutoff together. Its retrospective degradation cannot identify the effect of any one component. We compare five predeclared arms:

| Arm | Classifier score | Training relationship |
|---|---|---|
| A | Saved V5 full-fit OOF classifier | Reference; outer training uses all eight development training recordings plus the normal anchor. |
| B | Raw HGB probability, inner member 0 | Same fitted estimator and preprocessing as C. |
| C | Sigmoid-calibrated probability, inner member 0 | Calibration uses record-disjoint inner calibration recordings. |
| D | Mean of three raw member probabilities | Same fitted bank as E; includes B. |
| E | Mean of three sigmoid-calibrated probabilities | Reproduces the V7 classifier OOF, before fusion thresholds. |

B/C and D/E identify calibration effects on the same fitted estimators. B/D and C/E compare a fixed single member with the fixed three-member bank; they do not estimate the benefit over every possible single member. A/B also changes training size and composition, so we do not attribute that difference to calibration. No member selection follows results. Arm A uses saved outer-training OOF predictions, not the deployed all-development model. Keep the original V5 operating point (28/29, 3 false starts, 37 normal-alert seconds) as a separate unchanged reference; A's new cutoff scan is not that operating point.

## Frozen data and computation

Reuse the twelve development recordings, three whole-recording outer folds and fit-only normal anchor from the hash-verified V5/V7 manifests. Within each outer training set, sort IDs lexicographically and assign inner calibration member j by index modulo 3. Fit the other IDs plus the anchor. Outer-held records enter neither preprocessing, HGB fitting nor calibration. Use the existing 120 causal classifier features, HGB hyperparameters and equal-record/class-balanced fit weights. Calibration uses equal-record weights without class balancing.

Fit exactly nine classifier/calibrator pairs. Save raw and calibrated member OOF arrays, fold provenance, environment versions and dependency hashes. No full-data refit follows this study. Check E against saved V7 classifier OOF within an explicitly recorded numerical tolerance; stop on a mismatch. Preserve NaN warm-up and gap resets. Reuse only the hash-verified V5 development residual channels and classifier OOF control; no retrospective arrays enter computation.

## Probability and alarm comparisons

For each classifier arm, report unweighted binary Brier loss, log loss and average precision on eligible OOF rows. Report pooled and equal-record macro values and per-record results; record prevalence and sample counts. Brier and log loss evaluate probability forecasts and do not measure calibration alone. Do not report class-balanced weighted loss as a natural-prevalence probability diagnostic. Use binary labels [0,1] explicitly for log loss on single-class subsets. Record each sigmoid's orientation and saturation/ties. An increasing sigmoid preserves within-member ordering, so B/C should not gain discrimination from calibration alone. The mappings differ between outer folds, so pooled ranking and pooled normal quantiles can change even when within-record ordering is preserved. Include per-record AP and paired alert/onset differences at each shared quantile, not only each arm's selected winner. Explain deviations using fold mappings, finite quantile interpolation, ties and saturation; do not assert automatic equality of pooled-threshold crossings. Averaging separately calibrated members can change ranking because calibration and averaging do not commute.

All alarm arms use the unchanged maximum innovation channel and the V5 residual threshold **16.174573325357347**. Fusion remains max(classifier/classifier cutoff, residual/residual cutoff), requiring both branches finite; fire strictly above 1. No temporal confirmation, smoothing, cooldown, relabelling, backfill or point adjustment.

Use two distinct comparisons:

1. Fixed classifier cutoff 0.8580258155093349, the saved V7 cutoff, for all arms. This is a probability-scale sensitivity check, not a fair equal-burden comparison or a causal accuracy estimate.
2. A bounded operating-point scan for each arm: normal OOF quantiles 0.95, 0.975, 0.99, 0.995, 0.999, 1.0 and nextafter(max normal,+infinity). Deduplicate finite positive cutoffs. At most 35 arm/cutoff points; no residual cutoff search. This estimates development operating-point tradeoffs; separately tuned cutoffs prevent attribution of alarm changes to calibration alone.

Evaluate actual episode starts inside annotated events, false starts per scored normal hour, normal-alert seconds/fraction and detected-event delay using the unchanged evaluator. Report every recording and source family, normal exposure, worst-family false-start rate and normal-alert fraction. Small family exposures do not support reliable operational rate guarantees. The common residual branch can mask classifier differences; always retain classifier-only diagnostics.

## Ranking and interpretation

Development screening requires at least 26/29 detected events, at most 2 false starts per normal hour, at most 5% normal-alert time, fewer false starts than saved V5 and no more normal-alert seconds than saved V5. Rank feasible points by most events, fewest false starts, least normal-alert time, least worst-family false-start rate, median delay, mean delay, arm order A/B/C/D/E and threshold order. Save best_all across A-E, best_control_A and best_new across B-E separately, each null if none qualifies. Include A as a control, not an independently new model; its ranking cannot hide a feasible new arm. If no new arm qualifies, report no feasible new arm. If a new arm qualifies, call it a development candidate only. Do not change this rule after scores. Log-loss clipping follows the installed scikit-learn float64 machine-epsilon behavior; record the version. Fixed-cutoff comparisons add five diagnostic points, outside the at-most-35 selection scan.

No old-28 inference, alternative retrospective threshold scan, final deployment fit or active-file replacement. Existing old-28 results remain retrospective diagnostic evidence. Repeated development use also permits selection overfit. A development win cannot establish transfer or satisfy the unresolved false-alert acceptance requirement.

## Independent-data feasibility

The authors describe a separately collected four-anomaly SKAB teaser from the same testbed. Check public access, exact version, schema, annotation semantics and its own license before proposing confirmation. It is not a new machine/site and four events cannot validate a 25/28 target. Keep any downloaded data quarantined outside training and do not score it in this protocol. A separate frozen confirmation specification must identify one final model and thresholds before inference.

## Verification and outputs

- Tests: pairing/raw extraction, fold exclusions, unchanged calibration base model, probability losses including single-class records, finite eligibility, deterministic thresholds/ranking and no-feasible handling.
- Pre-fit read-only Model QA check; freeze protocol/source hashes before actual fits.
- New research module and CLI only; preserve scientific source files and six active V5 artifact hashes.
- Save all development points, per-record/family metrics and raw/calibrated OOF scores under artifacts/ablation-development/.
- Explain supported contrasts and confounders in docs/ML_V8_ABLATION_RESEARCH.md; append evidence-backed history to docs/RESEARCH_LOG.md.

## Sources

[scikit-learn calibration for version 1.9.1](https://github.com/scikit-learn/scikit-learn/blob/1.9.1/doc/modules/calibration.rst), [Ovadia et al. on uncertainty under dataset shift](https://proceedings.neurips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf), [SKAB pinned source and teaser description](https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/README.md). The V7 degradation review records the broader false-alert literature and its assumptions.
