# V7 degradation: fixed-model diagnosis and false-alert research

Date: 2026-10-23. Scope: diagnose the already trained V5 and rejected V7. V5 remains active. This investigation fits no model, tries no alternative threshold, changes no annotation and activates no release. The old audit is already examined material; this analysis provides explanations rather than independent validation.

## Findings supported by the saved models

We reconstructed both fixed models on all 23 audit recordings. Their aggregate, group and per-recording metrics/events reproduce the saved reports. Both have 18,208 scored normal seconds. Raw innovation scores agree within 1e-8 across all recordings; the predictor did not become numerically worse.

| Audit measure | V5 | V7 |
|---|---:|---:|
| Detected annotation segments | 26/28 | 23/28 |
| False starts | 56 | 81 |
| Normal seconds in alert | 272 | 386 |
| False starts / observed normal hour | 11.072 | 16.015 |
| Median delay among detected segments | 18 s | 3 s |
| Mean per-recording ordinary average precision | 0.780 | 0.824 |

V7 does not worsen every metric. Its average precision and detected-event delay improve, while the agreed recall/false-burden outcome worsens. Delay excludes missed events. Average precision measures score ranking across thresholds, not the quality of the selected alert rule.

### 1. A stricter predictor threshold removes two detections

The frozen innovation threshold rose from 16.174573325357347 to 61.83420703195029, about 3.82 times. The new value is the maximum normal development OOF residual. The underlying audit residuals stayed the same. Here "raw residual" means the dimensionless standardized innovation score before division by the alert threshold, not a sensor error in physical units.

| Lost interval | V5 first detecting branch | Maximum raw residual inside interval | V7 classifier maximum / its threshold | V7 residual maximum / its threshold |
|---|---|---:|---:|---:|
| other-10-event-0 | Innovation | 45.8525 | 0.3621 | 0.7415 |
| other-11-event-0 | Innovation | 16.9956 | 0.9360 | 0.2749 |
| other-4-event-0 | Classifier | 7.1002 | 0.8291 | 0.1148 |

The first two residuals exceed V5's threshold but never reach V7's. V7's classifier does not rescue them. In the third interval, V5's classifier barely crossed its own threshold; neither V7 branch crosses. V7 has no already-active alarm at these interval starts and no crossing inside them. These are missed intervals under the frozen rule, not artifacts of giving an earlier alarm no onset credit. No new gains offset them.

### 2. Classifier crossings account for most added burden

| Branches above threshold at a false episode start | V5 | V7 |
|---|---:|---:|
| Classifier only | 31 | 63 |
| Innovation only | 13 | 7 |
| Both | 12 | 11 |

Classifier-only normal-alert time rises from 95 to 282 seconds. The categories describe fixed threshold crossings; they are not an independent causal ablation. Numerical classifier thresholds, 0.9954 and 0.8580, cannot be compared as if they belonged to the same score function.

The V7 classifier exceeds its fixed threshold on 9/8,745 development OOF normal rows (0.103%) versus 315/17,340 audit normal rows (1.817%), about 17.65 times the row fraction. These denominators count rows, not seconds or alert episodes. Predictor normal exceedances change from zero OOF rows to 85 audit rows despite the higher residual threshold; the largest audit normal residual is 374.026. An empirical development maximum does not bound future residuals.

### 3. Errors concentrate in specific recordings

| Recording | V7 false starts | V7 normal-alert seconds |
|---|---:|---:|
| other-13 | 24 | 161 |
| valve1-2 | 30 | 125 |
| valve1-1 | 24 | 83 |
| All other recordings | 3 | 17 |

These three recordings contain 78/81 false starts and 369/386 normal-alert seconds. The valve1 family alone rises from 29 to 54 false starts; the separately declared normal audit improves from three starts/three seconds to zero. A uniform deterioration on all normal data is therefore the wrong explanation.

### 4. Most false starts are not annotation-boundary disagreements

- Only 3/81 V7 false starts are within five seconds of an annotation boundary. The five-second distance is descriptive, not a forgiveness window.
- None of its false-start episodes later overlaps an anomaly annotation.
- There are 21 singleton episodes and 46 episodes with at most two observed rows. The longest false episode lasts 46 observed seconds; 55 starts follow a previous false start within ten seconds.
- Fourteen starts occur at the first eligible row after segment warm-up. Gap/reset handling deserves targeted study, but does not explain all 81.
- False-start episodes contain 377 normal-alert seconds. The reported 386 includes another nine normal seconds at the tails of episodes that started inside true annotations. The two quantities have different definitions.

Repeated crossings create fragmentation; a longer wait alone is not established as a solution. V6 already showed that blanket confirmation can remove short true detections along with false starts. Episode merging can reduce notification count without reducing wrong sensor classifications; both quantities would need reporting.

## Likely contributors and remaining uncertainty

The measured normal-score tail does not transfer from nested OOF to the deployed ensemble/audit. Both recordings and fitted models differ between these scopes. Small calibration groups, recording heterogeneity and repeated development selection are consistent with this failure. Twelve development recordings and about 2.55 normal exposure hours do not become thousands of independent experiments merely because they yield many sensor rows.

There are large normal operating differences: the normal anchor's flow median is about 124.67 L/min, while normal portions of several valve recordings are around 32 L/min. Those lower-flow modes also occur in development, so anchor distance alone does not prove that audit data are outside the entire training support. Median/IQR feature distances describe a reference difference; they do not establish model attribution or a physical fault.

We changed classifier fitting subsets, ensemble averaging, calibration and both branch thresholds together. This two-model comparison cannot isolate the causal effect of sigmoid calibration. Raw nested-fold HGB predictions were not saved, so the report's raw-versus-calibrated member comparisons use deployed pairs on the same audit rows. They do not supply the missing nested ablation.

## Do these alarms indicate real danger?

Three claims must remain separate:

| Claim | Evidence available here |
|---|---|
| Sensors deviate from a reference or a model | Measured values, residuals and fixed threshold crossings |
| The detector disagrees with SKAB's anomaly annotation | Benchmark true/false status under the unchanged onset rule |
| Equipment has a harmful condition requiring action | Not established: operating commands, setpoints, maintenance findings, safety limits and independent adjudication are not provided |

The [official SKAB descriptions](https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/data/README.md) describe controlled valve/flow, imbalance, temperature and cavitation experiments, and a separate normal-mode recording. The binary anomaly label is not a calibrated danger probability. A normal-labelled deviation can be a harmless operating transition, an incomplete annotation or an unexplained change; confidence alone cannot select between those interpretations. Three V5 alerts in the normal-mode audit are benchmark false alarms, and V7 removes them. The remaining disagreements retain their false labels; this review does not declare them hidden true failures.

The 28 gap-split annotated segments are not 28 independent machines or physical incidents. The [SKAB changepoint leaderboard](https://github.com/waico/SKAB/blob/b2c0d46c2971dcbfe71e26087b6d231998bb91c2/README.md) uses a separate right-sided detection window. We do not import it into the frozen onset protocol after seeing errors.

## Scientific methods relevant to the problem

| Primary paper | Practical lesson | Boundary for SignalWatch |
|---|---|---|
| [Ovadia et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/9547-can-you-trust-your-models-uncertainty-evaluating-predictive-uncertainty-under-dataset-shift.pdf) | In-distribution post-hoc calibration need not survive distribution shift. | Supports a transfer diagnostic; neural-network results do not prove calibration alone caused this tree-model regression. |
| [Cobb et al., AISTATS 2022](https://proceedings.mlr.press/v151/cobb22a.html) | Calibrate sequential false detection behavior, not only individual scores. | Its independent reference/pre-change formulation does not yield a direct guarantee on these correlated alert episodes/hour. |
| [Siffer et al., KDD 2017, SPOT/DSPOT](https://www.amossys.fr/wp-content/uploads/anomaly-detection-with-evt-1.pdf) | Model score tails with extreme-value theory; account for drift in a local model. | Requires checking tail fit, contamination and dependence. The paper's i.i.d. formulation is not a universal guarantee for industrial traces. |
| [Gibbs and Candes, NeurIPS 2021](https://arxiv.org/abs/2106.00170) | Adaptive conformal inference can track long-run prediction-set coverage with outcome feedback under shift. | Coverage is not a bound on false episode starts, harm or delay. A predictor observes future sensor outcomes; it does not observe physical fault truth. |
| [Kim et al., AAAI 2026, CANDI](https://ojs.aaai.org/index.php/AAAI/article/view/38524) | Curate possible false positives and adapt to normality changes while preserving the backbone. | Reconstruction/latent adaptation needs another architecture; automatic adaptation can absorb real faults and needs a new evaluation. |
| [Kim et al., AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20680) | Point adjustment can inflate reported detection quality. | Preserve unadjusted point metrics, onset timing and false-alert duration; do not repair scores by relabelling disagreements. |

We checked primary abstracts and relevant methods/assumptions, not a replication of these papers. No paper implementation, figure or benchmark gain is claimed for SignalWatch.

## Recommended next experiment, not executed

1. Keep V5 as the reference and define the operational decision: known experimental anomaly, unfamiliar operating mode or confirmed fault. The current data directly support only the first task and measured deviations.
2. Run one-factor development ablations: same fixed training partitions with calibration-only, averaging-only and the unchanged control. Measure calibration with proper probability scores separately from event recall, false starts, normal-alert time and delay. Include per-recording/family limits rather than relying on a pooled total alone.
3. Test normal-mode-aware score calibration or an unfamiliar-mode abstention branch before a larger classifier. Existing raw current-value features and normal mode diversity make this a better grounded hypothesis than adding trees indiscriminately. Mode identities must come from causal signals/configuration, never audit labels.
4. Treat sequential confirmation/hysteresis and statistical tail calibration as separate experiments. Freeze reset, delay and notification rules on development; preserve the raw ML trace and short-event sensitivity.
5. Freeze the choice, then use new independent recordings or previously untouched operating experiments for confirmation. Reusing these 28 segments for another winner selection would not establish generalization.

These steps are proposals, not a promise of zero false alerts or a new approved architecture. No further training has started.

## Reproducible evidence

- [Complete diagnostics and all 137 false episodes](../outputs/v7-diagnosis/diagnostics.json)
- [Diagnostic script](../work/v7-diagnosis/analyze.py)
- [Visual trace report](../outputs/v7-diagnosis/report.html)
- [False-episode CSV](../outputs/v7-diagnosis/false-episodes.csv)
- [Selected V7 audit](../artifacts/calibrated/retrospective_audit.json)
- [V5 reference](../artifacts/v5/frozen_evaluation.json)

The visual cases use all raw rows from deterministically selected recordings: highest V7 normal-alert time, highest V7 false-start count and the normal audit control. Source hashes accompany the arrays. Full diagnostic JSON SHA-256: 338c7ce31a4e8ad19a6acf4012cbd036300348d34e7f8b17128c92671800e9c3.
