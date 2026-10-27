# V6 bounded alarm-confirmation study

## Approved scope and diagnosis

The user approved ML plus temporal alarm stability on 2026-10-23. Root: D:/Projects/signal-watch. Keep the genuine trained V5 classifier/predictor, configurations and thresholds. V6 adds a separately evaluated decision policy, not a new learned architecture or automatic feedback learning. V5 is preserved under artifacts/v5. No dependencies, external data, Git push or deployment.

Saved V5 has 56 false starts: 29 singleton, 36 at most two rows, 45 at most five rows. All 43 V4 false starts remain; fusion adds thirteen. These previously examined summaries informed the design, not parameter selection. Evidence: work/false-alert-diagnosis/saved_episode_lengths.json.

## Frozen policy

Candidates: consecutive observed positive raw fused scores spanning [0,1,2,5,10] elapsed seconds. This does not establish the analog signal between observations. Zero is the exact identity control. Positive means finite and strictly above threshold1. Pending starts at the first positive observation. Activate only at an observed positive row whose elapsed time minus pending start meets the duration. No backfill. Stay active only while raw positive. Nonpositive/nonfinite scores and timestamp gaps reset state. No delayed recovery, cooldown, periodic resets, labels or retrospective episode-length filtering.

Retain raw relative score, confirmation state and raw episode start. Pending rows remain finite no-alert: min(raw_score,1), never NaN. Active rows expose their actual raw score. Preserve warm-up NaNs and baseline arrays. The confirmed onset is activation time. Show both pending start and activation if they straddle an annotation. Existing onset_v2 evaluates emitted alarms; an already confirmed episode never becomes a new hit. Report raw and confirmed metrics separately; gated-score AP is not unchanged raw-model discrimination.

Both annotation-boundary directions require tests: a normal raw start can confirm inside an anomaly, or an anomalous raw start can confirm on normal data. Status follows actual confirmation, never raw origin. Normal-alert time is a subset of raw alert time; false-start counts need not decrease monotonically with waiting.

## Selection and acceptance

Use only hash-verified preserved V5 OOF scores and the twelve original development recordings. Freeze source/dependency hashes, grid and rank before scanning. Reused OOF remains selection evidence. Final refit uses exactly V5 configurations and normal/development anchors; in-sample refit scores never choose the policy.

Feasibility: retain EVERY event detected by the raw OOF control; <=2 false onsets/normal hour; <=5% normal time in alert; median detected delay <=control median+10 seconds. Preserve scored normal exposure exactly. Rank by fewer false starts, lower normal-alert fraction, lower median and mean delay, then candidate order. Promote only a strict burden improvement over identity. Otherwise retain V5 and skip retrospective scoring. Report fold/family slices and paired events.

After freezing the winner, audit it once on the old28: success requires >=25/28, fewer than56 false starts and no increase in normal-alert seconds. The operational objective remains <=2/h; failure stays explicit. If acceptance fails, retain V5 and do not try another policy on this audit. No guaranteed target or new blind-test claim. Further architecture changes need a separately agreed experiment.

## Integration and reproduction

Study_version6 keeps three public method keys; learned denotes the confirmed decision score. Expose raw learned scores and policy state separately. Preserve raw model metrics; explain the distinction in API/UI/model card. Keep the selected black/amber layout. The training API must refit both fixed branches, reproduce selection and all saved per-recording/event results in staging before replacement. Copying pretrained weights alone is not training. Preserve original audited artifacts when serialization changes.

## Memory and agents

Codex memory supplies requirements/evidence boundaries. A fresh Codebase Memory list contains no SignalWatch graph; source fallback, no indexing. Root owns archives, plan, active integration, API/frontend and docs. AI Engineer may own new policy/study/tests/CLI only. Model QA Specialist reviews onset semantics, leakage, exposure, selection and acceptance read-only, without fitting candidates or auditing variants. Review timebox8 minutes per phase. No overlapping edits.

## Verification

Tests first: identity, finite pending exposure, irregular elapsed time, strict boundary, gap/nonfinite/negative reset, no rearming inside continuous positives, no backfill, prefix/label independence, unchanged baselines. Study tests cover frozen dependencies, rank, retained event IDs, report reconciliation and no retrospective access. Then one selected audit, API integration, actual training reproduction and desktop/mobile raw-versus-confirmed checks. Passing code checks does not establish model quality.

## Sources

- [Prometheus alerting](https://prometheus.io/docs/prometheus/latest/configuration/alerting_rules/): pending-to-firing duration semantics, not a trained-model method.
- [scikit-learn thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html): scores and decision policy differ; separate selection data.
- [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html): disjoint calibration; monotonic mapping alone does not guarantee different alerts or valid final-refit thresholds.
- [Cobb et al.](https://proceedings.mlr.press/v151/cobb22a.html): false-detection calibration is sequential; no paper algorithm/guarantee claimed.
- [Sun et al.](https://proceedings.mlr.press/v235/sun24h.html): adaptive-threshold research alternative, not implemented here.

No upstream code copied. An MDPI persistence paper appeared in Exa, but its full page/PDF returned429/access failure; it is not verified implementation evidence.

## Observed result: retain V5

One frozen development scan completed. Waiting seconds -> detected/29, false starts, normal-alert seconds, median delay:

| Seconds | Events | False starts | Normal-alert seconds | Median delay |
|---:|---:|---:|---:|---:|
| 0 | 28 | 3 | 37 | 2 |
| 1 | 26 | 1 | 34 | 4 |
| 2 | 25 | 1 | 33 | 6 |
| 5 | 20 | 0 | 31 | 8.5 |
| 10 | 13 | 0 | 31 | 21 |

Every positive wait loses raw detected development event IDs, so only identity is feasible. All candidates preserve 9,179 normal seconds. Promote=false, status=retain_v5. No new raw-model refit, retrospective audit or activation occurred. Current V5 stays26/28 with56 false starts; this experiment has not fixed its alert rate. The source, API and UI extensions remain conditional on a future accepted version6; current version5 uses its original decisions.
