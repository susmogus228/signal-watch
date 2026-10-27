# V13: Pressure correction, false alerts and gradual-defect retention

Date: 2026-10-27. **The numerical Pressure problem is corrected in this experiment. The joint alarm-quality goal is not met.** K4 shows partial burden improvement with a substantial linear-defect delay cost; K2 worsens. Active V5 remains unchanged.

## What changed

The user requested a Pressure fix before another quality comparison. The [frozen protocol](ML_V13_PRESSURE_SCALING_PLAN.md) changes only the normal-bank Pressure scale to max(IQR, population standard deviation, 1e-6), using the same training-only normal bank. Bank medians, seven other sensor scales, original 186 classifier preprocessing entries, appended-feature preprocessing rules, 198 features, data/labels/folds, K1/K2/K4 estimators, sample weights and alarm rules remain fixed.

This is a learned GMM/HGB pipeline with immediate classifier decisions. It adds no confirmation wait, clipping, test-record normalization or new dependency. Scaling also changes physical covariance regularization, KMeans initialization geometry and the possible EM trajectory. A resulting alarm difference cannot be attributed solely to floating-point arithmetic.

One run completed in 202.825 seconds: nine converged Gaussian-mixture fits and nine completed HGB fits, eighteen top-level calls. All arms passed numerical checks. All 306 predeclared operating points failed at least one quality condition, so no final fit/model, old-audit/teaser inference or activation followed.

## Numerical result and unchanged control

| Outer fold | V12 Pressure scale | V13 Pressure scale | V13 bank maximum absolute Pressure z |
|---|---:|---:|---:|
| 0 | 0.327927 | 0.327927 | 5.000012 |
| 1 | 0.000001 | 0.25716489305517487 | 5.100661 |
| 2 | 0.000001 | 0.2610529761145108 | 6.280867 |

The old maximum of about 1,639,639 becomes 6.280867 on these normal banks. Every bank still contains 4,608 selected training-normal rows. The other seven scales and all bank centers are identical to V12. Fold 0 uses an unchanged Pressure scale: all three same-K OOF arrays on its held-out records are exactly equal to V12, stronger than the declared absolute 1e-12 check.

All 279 saved bank/training/held-out/per-record numerical audits pass. The largest fixed-parameter centered-versus-expanded NLL difference is **3.637978807091713e-12**, below 1e-7. Minimum fitted variance is **0.0010045758539011574**, above reg_covar=0.001. V12's two K4 held-out-normal maxima were about 0.039292/0.045301. The audits compare the arithmetic at each experiment's own fitted parameters; they do not measure full EM error or create matched classifier predictions under corrected arithmetic alone.

## False alerts with both gradual intervals retained

The comparison rule was fixed before training: among each arm's scanned points retaining both progressive intervals and at least 26/29 total events, choose the smallest false-start count, then normal-alert time, most hits, delay and cutoff order. These are descriptive frontier points, not replacement selections. Every row uses the same 9,179 observed normal seconds.

| Arm | V12 events | V13 events | False starts V12 to V13 | Normal-alert seconds V12 to V13 |
|---|---:|---:|---:|---:|
| K1 | 27/29 | 27/29 | 5 to 5 | 37 to 37 |
| K2 | 27/29 | 27/29 | 6 to 17 | 37 to 75 |
| K4 | 27/29 | 27/29 | 40 to 21 | 176 to 111 |

K4 reduces false starts by nineteen and normal-alert time by sixty-five seconds on this reused development comparison. Most of the count reduction is in other-9 (20 to 3); valve1-9 decreases 7 to 5 while valve1-12 remains 11. K2 worsens, and K1 remains unchanged. The best progressive-preserving false count among these arms is still five; the requirement is fewer than three.

Both progressive intervals remain detected at these frontier points, but detection timing changes:

| Arm | Linear other-6 delay, V12 to V13 | Exponential other-9 delay, V12 to V13 |
|---|---:|---:|
| K1 | 46 to 46 seconds | 7 to 7 seconds |
| K2 | 46 to 26 seconds | 7 to 7 seconds |
| K4 | 3 to 46 seconds | 7 to 6 seconds |

K4's linear onset is 43 seconds later. The code adds no timer; its learned signal/operating-point combination crosses later. This cost must accompany the lower false count. Median delay alone would hide it: K4's median over detected events decreases from three to two seconds, even as that linear case becomes later.

The cutoffs also change: K2's frontier goes from 0.9555687264953208 to 0.9176793138426952; K4's from 0.8571184329230667 to 0.8395438345167259. For K2, other-9 OOF probabilities remain exactly equal to V12, but its false starts at the chosen frontier rise from zero to five because the shared cutoff is lower. We therefore compare complete operating points, not equal numeric thresholds or isolated arithmetic effects.

## Low-burden region and full acceptance

| Arm | Best V13 recall under count/rate/time burden limits | False starts | Normal-alert seconds | Gradual limitation |
|---|---:|---:|---:|---|
| K1 | 26/29 | 1 | 28 | Linear other-6 missed |
| K2 | 26/29 | 1 | 28 | Linear other-6 missed |
| K4 | 9/29 | 1 | 18 | Linear other-6 missed |

The full screen requires >=26/29, fewer than three false starts, <=2/h, <=37 normal-alert seconds and retention of both original gradual intervals. None of the 306 points passes. We neither weaken the screen nor select K4 because one comparison improved. Active V5's separate development control remains 28/29, three false starts and 37 normal-alert seconds; its retrospective false-alert problem also remains unresolved.

## Probability diagnostics are separate

| Arm | Pooled AP V12 to V13 | Brier V12 to V13 | Log loss V12 to V13 |
|---|---:|---:|---:|
| K1 | 0.911362 to 0.911362 | 0.092064 to 0.092064 | 0.414039 to 0.414039 |
| K2 | 0.914870 to 0.913281 | 0.090095 to 0.093192 | 0.390281 to 0.397139 |
| K4 | 0.901297 to 0.896499 | 0.092949 to 0.089432 | 0.440574 to 0.433464 |

K4 improves two probability losses and its relevant false-alert frontier while pooled AP decreases. These measures answer different questions. K1's equality does not identify which fitted tree splits used appended features; outer model objects were not retained for that analysis. None of these scores is a physical-failure probability.

## Verification and decision

The full backend suite passes **192 tests**, with seven existing warnings, in 54.63 seconds before the sole study. Sixteen new tests cover Pressure tails/constant channels, unchanged columns, fitted predictions on synthetic fixtures, causal/label boundaries, numerical guards, selection and final rejection. Frozen source hashes stay unchanged afterward. No API/frontend source changed, so browser/build checks were not repeated.

Independent Model QA reconstructs saved outcomes and numerical/provenance evidence without new model fitting or classifier predictions. The [verification evidence](evidence/v13-pressure-scaling-check.json) records that review separately from numerical success and quality failure. All 75 study dependencies, eleven V12 artifacts and six active V5 files remain preserved. The conditional final-model path is unexercised because no candidate qualifies.

R22's correction and comparison are complete; R17/R20 joint model quality remains open. The result supports using the numerically corrected transform for future agreed studies, not replacing V5. It does not establish physical fault causes, field reliability or independent generalization: these development records have been reused, and the gradual cohort contains only two source-described intervals.

## Preserved artifacts

- [Frozen plan](../artifacts/pressure-scaled-development/plan.json), [execution marker](../artifacts/pressure-scaled-development/execution_started.json)
- [Complete report](../artifacts/pressure-scaled-development/report.json), [all 306 points](../artifacts/pressure-scaled-development/candidate_audits.json), [pre-final selection](../artifacts/pressure-scaled-development/selection_before_final_fit.json)
- [K1 OOF](../artifacts/pressure-scaled-development/K1_oof_scores.npz), [K2 OOF](../artifacts/pressure-scaled-development/K2_oof_scores.npz), [K4 OOF](../artifacts/pressure-scaled-development/K4_oof_scores.npz)
- [K1 fit/audit evidence](../artifacts/pressure-scaled-development/K1_fit_evidence.json), [K2 evidence](../artifacts/pressure-scaled-development/K2_fit_evidence.json), [K4 evidence](../artifacts/pressure-scaled-development/K4_fit_evidence.json)

Report SHA-256: 6fe202065d97dc5eb9c332749e757ca7fc993b840547ed7c2ee6bfe79e5b5b2d. Candidate SHA-256: 5b44c46b76f0aa3f993402c753c642044f6031768f8e2ebb973b1dee9cc9bb6c.

scripts/study_pressure_scaled.py refuses to repeat this execution. Reproduction requires an isolated copy with the preserved source/data/control chain. Exact-version source checks and scope limits appear in the protocol and RESOURCES.md. No library code is patched and no external algorithm gains are claimed.
