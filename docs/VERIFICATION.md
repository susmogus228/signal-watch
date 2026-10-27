# Research packaging and training pause

Date: 2026-10-27. The user paused further ML training and requested a GitHub-readable account of completed work. [README](../README.md), [results report](RESULTS.md), [V1-V13 history](RESEARCH_LOG.md) and [figure/data index](research/README.md) now separate active V5, rejected studies, partial gains and unresolved quality goals.

- Exporter verification: application Python runs scripts/build_research_summary.py --check without plotting or model dependencies. Ten saved JSON sources control fourteen result rows, three CSVs and three PNG/SVG pairs. Source/generator/output hashes match; the check requires the exact ten exported data/image files.
- A negative validation case omits one SVG from a copied manifest in work/. The check rejects that omission. The supplied manifest and model artifacts remain untouched.
- Independent Model QA reconciles 140 metric fields, all six frontier cutoffs and twelve gradual delays, three CSVs, ten source hashes and ten exported files. It reviews the three PNGs, historical availability/selection decisions and claim boundaries. Three low findings were fixed; follow-up verdict PASS. This is internal factual review, not scientific peer review or a new model-quality validation.
- All 172 pre-task protected files (all model/result artifacts plus the data manifest) retain their hashes. The 75 frozen V13 dependencies also match. No fitting, model inference, threshold sweep, activation, training API call, new application/global dependency or Git push/deploy occurs.
- Local marked-based Markdown previews are inspected with actual images at desktop 1280px and narrow 390/304px. No document overflow; images load and tables use bounded overflow. Main results and return-to-application links work. GitHub's live renderer, its Mermaid rendering, external clone setup, physical devices and field equipment are not tested.
- The new exporter and English documents receive content/link/format checks. Backend/API/frontend inference code remains outside this edit scope. The earlier 192-test result below belongs to V13 verification; we do not rerun or relabel it as a new documentation-task suite.

[Packaging evidence](evidence/research-packaging-check-2026-10-27.json) records current hashes/checks and separate review scope. R23's local documentation work is complete; R17/R20 quality remains unresolved and new research training is paused. This task prepares repository-native files without claiming a public upload.

# Historical ML verification: V13 Pressure scaling

Date: 2026-10-27. Pressure correction and the paired study are complete locally. Numerical verification passes; the joint model-quality screen fails. V5 remains active, R22's correction/comparison is complete and R17/R20 remain open.

- Full backend suite before the sole study: **192 passed, seven existing warnings, 54.63 seconds**. Sixteen new V13 tests cover the new transform, guards and study behavior. Frozen source remains unchanged afterward.
- One run: nine converged GMM and nine completed HGB fit calls, 202.825 seconds, three valid arms and 306 fixed points. None qualifies; no final fit/model, old-audit/teaser inference or activation follows.
- All 279 saved numerical audits pass. Maximum bank Pressure magnitude falls from about 1.64 million to 6.28; maximum fixed-parameter direct/expanded NLL difference is 3.64e-12 and minimum variance is 0.0010045758539011574. Fold 0 keeps its scale and matches V12 OOF exactly for all three arms. Physical regularization and initialization geometry also change; this does not isolate arithmetic as the cause of every alert change.
- At each arm's predeclared progressive-preserving comparison frontier, all retain both gradual intervals and 27/29 total hits. K1 stays at five false starts/37 normal-alert seconds; K2 worsens six to seventeen/37 to 75; K4 improves forty to twenty-one/176 to 111. K4 linear delay worsens three to forty-six seconds. Low-burden K1/K2 retain 26/29 and one start/28 seconds but miss that linear interval.
- Independent post-run Model QA reconstructs all 306 points and paired frontiers from saved predictions, then checks bank/scaler/numerical/provenance evidence without model fitting or new classifier inference. Detailed verdict and preserved hashes are recorded in the linked evidence.
- All 75 frozen dependencies, prior V12 scientific artifacts and six active V5 files remain unchanged. API/frontend source is unchanged; browser/build, external clean installation, publication, physical-device and field tests were not repeated. The real conditional final-model path remains unexercised because selection fails.

[V13 research and limits](ML_V13_PRESSURE_SCALING_RESEARCH.md), [verification evidence](evidence/v13-pressure-scaling-check.json), [current requirements](REQUIREMENTS.md).

# Historical related-work documentation verification

Date: 2026-10-27. Scope: the user requested online study of related approaches, comparison with our ML and inclusion in the report. [Comparison report](RELATED_WORK_COMPARISON.md) and [thirteen-source ledger](evidence/related-work-sources-2026-10-27.json) contain eight research papers, five technical sources, shared failure patterns, access limits, a report-ready paragraph and future priorities.

- Root inspected primary online sources through Agent Reach Exa and checked exact-version preprocessing guidance through Context7; installed scikit-learn remains 1.9.1. Abstract-only and selected-section reading are labelled. No external algorithm was benchmarked.
- Internal independent Model QA factual review: PASS for local evidence, source-scope/claim alignment and separation of proposals from completed work. The reviewer used the source ledger and supplied primary excerpts rather than repeating the full web retrieval. The review is not external scientific peer review.
- Documentation checks verify the source count/IDs, English content and local links. Frozen V12's 57 dependencies and the six active V5 files remain unchanged; earlier scientific results stay intact. [Current review evidence](evidence/related-work-check-2026-10-27.json).
- This task changes documentation only. It runs no model fit, classifier inference, new threshold scan, backend suite, frontend/browser test, deployment or publication. The earlier 176-test result below belongs to V12 source verification, not a new literature-review test run. R21 is complete; R17/R20 model quality remains open.

# Historical ML verification: V12 normal regimes

Date: 2026-10-27. The approved bounded experiment is implemented and complete locally. No candidate meets the joint quality screen; V5 remains active and R17/R20 remain open.

- Full backend suite: **176 passed, seven existing warnings, 69.30 seconds**, before the sole study run; all frozen source hashes remained unchanged afterward. The suite includes eleven V12 tests and seventeen conditional-diagnostic tests. Existing calibration/deprecation warnings remain; actual V12 mixture fits have no convergence warnings.
- One real-data study: nine converged GMM and nine completed HGB fit calls, 247.300 seconds, three valid arms and 306 predeclared points. Zero feasible points, no final model, no old-audit/teaser inference and no active replacement.
- Both progressive intervals retained with >=26 total hits: K1 gives 27/29, five false starts and 37 normal-alert seconds; K2 27/29, six and 37; K4 27/29, forty and 176. The low-burden K1/K2 points have 26/29, one start and 28 seconds but miss linear other-6.
- Independent QA reconstructs all 306 points and their recording/family/paired/progressive metrics, seven candidate/control probability banks, 41,472 repeated bank selections, 153 recording/class weight masses and all fitted scaler evidence. All appended scalers and 270 density summaries reproduce under the installed expanded Gaussian arithmetic. K1's OOF equals V10 B in all twelve records. These are correctness checks, not quality approval.
- Medium numerical-conditioning finding: Pressure IQR=0 in folds 1/2 and the chosen 1e-6 scale produce magnitudes up to 1,639,639. Two K4 fitted variances fall below the nominal added regularizer through cancellation. At fixed saved parameters, centered-residual density differs from the installed expanded formula by up to 0.045301 on held-out normals. This does not measure full EM error or explain all forty false starts. The fold-0 normal-coverage mismatch remains independently present. No corrected classifier or replacement result is claimed.
- All 57 V12 dependencies, prior frozen V10/V11 scientific evidence and six active V5 files match. API/frontend source is unchanged. Browser/build, public deployment, clean external installation, physical-device and field-equipment checks were not repeated. The real conditional final-model path remains unexercised because every study rejected selection.

[V12 research and numerical limits](ML_V12_NORMAL_REGIME_RESEARCH.md), [current evidence](evidence/v12-normal-regime-check.json), [requirements](REQUIREMENTS.md), [version history](RESEARCH_LOG.md). The broader learned-model quality goal is unresolved; these completed experiments do not make the active detector operationally reliable.

# Historical V10/V11 learned-model verification

Date: 2026-10-27. Both fixed studies completed and were rejected under their joint quality criteria. At this stage V12 had separate approval but no results; its later completed study appears above.

- V10: 15 actual HGB development fits, 204 fixed operating points, zero feasible candidates. The low-count point with 26/29 and one false start loses linear imbalance; retaining both progressive intervals at 27/29 gives five starts and 37 normal-alert seconds.
- V11: six development fits, 204 fixed points, zero feasible candidates. With both progressive intervals and at least 26 hits, lowest false counts are HGB31: 27/29, five starts, 38 seconds; ExtraTrees256: 27/29, eleven starts, 138 seconds. Pooled ranking/probability losses improve, but the alarm requirements do not.
- Independent Model QA reconstructed all 408 candidates, proper losses, recording/family/progressive slices, fit partitions/scalers/weights and V10 mining. No discrepancies; all 17 V10 and 29 V11 dependencies match. No final fit, old-audit/teaser model inference or activation follows either rejected run. Six active V5 files remain unchanged.
- Full backend suite at completion of V10/V11: **160 passed, seven existing warnings, 54.26 seconds**, before V12 source extensions. V12's conditional 198-feature dispatcher/identity changes then passed 17 targeted tests, including a selected point without a completed final model. The later 176-test suite above covers the completed V12 source work.
- API/frontend code is unchanged; no new browser, frontend-build, deployment, physical-device or field-equipment proof is claimed. R17/R20 model quality stays open. The conditional real-model diagnostic path remains unexercised because no qualifying final artifact exists.

[V10 research](ML_V10_TREND_MODEL_RESEARCH.md), [V11 research](ML_V11_TREND_BACKBONE_RESEARCH.md), [closed-study evidence](evidence/v10-v11-learned-model-check.json), [V12 protocol](ML_V12_NORMAL_REGIME_PLAN.md).

# V9 branch-policy and transfer verification

Date: 2026-10-27. The scoped diagnosis/change/transfer study is complete locally. R17 alert quality remains unresolved and V5 stays active.

- Full backend suite: **134 passed, 7 warnings, 97.36 seconds**. Existing Starlette/httpx and FrozenEstimator warnings remain; no global dependencies changed. New policy tests passed 7 cases; the strict teaser adapter/runner tests passed 11 cases after recorded red/green checks.
- Diagnosis reproduces all three original development false starts and all 29 event outcomes from hash-verified saved OOF predictions. All three original false episodes last one observation and start only in the innovation branch. Mode drift is not established.
- One fixed one-second residual-only policy: 27/29, one false start, 34 normal-alert seconds over 9,179 normal seconds. Lost event16 and largest common-event delay increase of 21 seconds are retained. The policy removes three original false starts but creates one confirmed start after an annotation boundary.
- One frozen separate-record comparison: exact 51,240-row to 6,405-observation pivot; 3/4 source interventions for V5 and V9, background starts 10/9 and duration 29/19 seconds over 5,678 assumed-background seconds. No fitting, tuning, old-28 inference or activation.
- Independent Model QA used its own scalar confirmation loop and rebuilt saved decisions, states, source pivot, masks, event outcomes, delays and exposure. No discrepancies; protocol/code/frozen-source review passed. Experimental correctness passes; overall model quality does not.
- Six active V5 files and the study dependencies remain unchanged. New code/docs remain English. UI/API implementation did not change, so this turn did not repeat browser/frontend builds or establish a new deployed state.

[Research and limitations](ML_V9_BRANCH_POLICY_RESEARCH.md), [current evidence](evidence/v9-branch-transfer-check.json), [version history](RESEARCH_LOG.md). Local one-shot scripts require the preserved ignored research inputs for reproduction; publication and a clean external machine were not tested.

# V8 component-study verification

Date: 2026-10-23. Active release remains V5. V8 returned no feasible new arm; it does not resolve R17's false-alert quality requirement.

- Full backend suite after the new research module: **116 passed, 7 warnings, 69.27 seconds**. Warnings cover the existing Starlette/httpx deprecation and the FrozenEstimator calibration-weight warning; the existing helper applies fit and calibration weights in their separate stages. No global packages changed.
- New module tests exercised fixed-bank pairing, finite eligibility, raw versus calibrated extraction, single-class proper losses, quantile deduplication, ranking/no-feasible behavior, recording-ID exclusion and causal prefix/gap/label invariance.
- One frozen run: 9 HGB/calibrator fits, 35 scan points, 5 fixed-cutoff diagnostics, 153.725 seconds. The saved V7 classifier OOF reproduces within absolute 1e-12 and zero relative tolerance. No old-28 inference, deployment fit or activation.
- All 24 frozen dependencies and six active V5 artifacts matched after the study. GET /api/health returned status ok/model_ready true; GET /api/model returned version 5.0.0.
- New research documents and local evidence links resolve; new owned code/docs contain no Cyrillic. The frontend/API source did not change, so this turn did not repeat the prior UI/build checks or claim new UI coverage.

[Research history](RESEARCH_LOG.md), [V8 results and limitations](ML_V8_ABLATION_RESEARCH.md). Independent final Model QA evidence accompanies the saved study check.

# V6/V7 false-alert experiment verification

Date: 2026-10-23. Active release remains V5. The implementation and experimental artifacts are verified; false-alert quality remains open.

- Full backend suite: 110 passed in 63.98 seconds. Warnings cover the existing Starlette/httpx deprecation, expected FrozenEstimator calibration-only weighting message and Windows pytest temporary-folder cleanup. No test failed; the global environment was unchanged.
- Final frontend typecheck, 4 logic tests, production build and 4 packaging tests passed. Vite retains its ECharts chunk-size advisory; no deployment occurred.
- V1/V4/V5 archive manifests: 11/18/26 files, all hashes match. Active V5 model, metadata, evaluation, selection, split and release manifest match their preserved originals.
- V6: one fixed five-policy OOF scan; positive waits lose previous hits. Promote=false; no new refit, retrospective audit or activation.
- V7: one real nested study, 147 fixed points and one preselected retrospective audit. Development: 27/29, 2 false starts, 10 normal-alert seconds. Retrospective: 23/28, 81 false starts, 386 seconds. Acceptance fails; V5 retained. No alternative audit threshold scan, candidate API retraining or activation.
- Model QA Specialist checked all 16 V7 dependencies, nested partitions/counts, selected OOF arrays/metrics/events, 147-point feasibility/rank, lost/gained IDs and retrospective group/recording sums. Verdict: PASS for experimental correctness and rejection; FAIL for activation. The reviewer did not fit candidates or perform new retrospective inference.

Live HTTP after restart passed /, health, model, evaluation, Swagger and OpenAPI. Active V5 100-row replay scores match the complete-record prefix; future labels and quality remain hidden. Inactive confirmation fields are empty. Budget2 reproduces the saved V5 development selection and leaves audit bytes unchanged. Browser Compare and Data & Model fit desktop1487px and mobile390px; error logs are empty. Updated captures appear in SCREENSHOTS.md.

The new code preserves V5 scoring. Conditional V6 confirmation visuals and V7 calibrated metadata/comparison have type/unit coverage; their inactive visual states and API training workflow were not exercised with a passing release. [Study outcome](ML_V7_RESEARCH.md), [current evidence](evidence/false-alert-studies-check.json), [requirements](REQUIREMENTS.md).

# Earlier V5 release verification (2026-10-23)

The selected complementary classifier/predictor detects 26/28 retrospective segments, meeting the requested minimum of 25. It produces 56 false starts over 18,208 normal seconds (11.072/h) and 272 normal-alert seconds (1.494%). The 2/h alert-rate limit fails and false starts increase by 30.2% versus V4. This is a local research workbench; the code checks below do not establish operational readiness.

- Backend: 75 tests passed, one Starlette/httpx deprecation warning. Coverage includes causal fusion, OOF reconstruction, onset suppression, original baselines, staged-source integrity, failed-training retention, branch prefix bounds, saved-budget hash/selection and HTTP evidence errors.
- After the cadence disclosure change, 27 targeted API/release/development tests passed, including both new cadence counterexamples. Scores are prefix-invariant given preloaded cadence; raw timestamp-stream initialization is not verified.
- Actual V5 API training job 496d0c0072b144cba147eb8de1b690eb completed in 250.915 seconds, reproducing all aggregate/family/per-record metrics and events. Model/report/file hashes match after reload. The original audited child stays preserved; serialization is semantically reproduced, not byte-identical.
- Frontend: typecheck, 4 logic tests, production build and 4 packaging tests passed. Main bundle 854.97 kB triggers the existing Vite size advisory.
- Live HTTP: model/report/file SHA agree; 100-row replay scores equal the corresponding complete-record prefix; future labels/quality remain hidden; warm-up is 2/30/30; event branch context is present; old-model replay returns 409; /docs, /openapi.json, / and /api/health return 200.
- Saved development budget exploration leaves frozen audit bytes unchanged. Budgets 0.25, 2 and 10 completed in 7.715, 0.336 and 0.090 seconds in the measured run. Budget 2 selects 28/29 with 3 false starts. These timings include local runtime variability and are not a benchmark guarantee.
- Browser after final build: desktop 1487 px and mobile 390 px have no document overflow; the preloaded-cadence note is visible before Start recorded replay. The selected valve interval shows 38-second delay and actual classifier branch context. Compare shows 26/28 and the failed false-alert limit; development budget 2 separately shows 28/29. Final browser error log is empty; current annotated captures appear in SCREENSHOTS.md.
- Independent Model QA reconciled grouped OOF arrays, all 307 candidate operating points, the declared rank, folds, source/dependency hashes and every retrospective aggregate/family/per-recording metric. The reviewer confirmed the recall success and false-alert failure without selecting another audit variant.

No Git push, public deployment, physical-equipment/phone test or new blind evaluation was performed. Source and runtime remain in D:/Projects/signal-watch. Original V1/V4 releases and the originally audited V5 model remain preserved.

# Historical V4 verification (2026-10-23)

The active study recognizes known abnormal states using weighted gradient boosting. The 25/28 target remains unmet: retrospective 22/28, 43 false starts / 18,208 normal seconds = 8.502/h, and 1.346% normal time in alert. Development selection is a separate 25/29 result. Software checks do not establish model usefulness.

- Backend: 57 tests passed, including weighted training/gap causality, label independence, original baselines, staged-training failure retention, publication provenance and zero-threshold display. One Starlette/httpx deprecation warning remains.
- Frontend: typecheck, 4 logic tests, production build and 4 packaging tests passed. Vite reports an 854 kB main bundle warning; no deployment or new dependency was added.
- Actual API training job f8515d5bdc454a5996048323c88efaac completed in about 125 seconds. It reproduced the selected point, aggregate/family metrics and each audit recording's metrics/events before replacement. Serialization hashes changed; this is semantic reproduction, not byte-identical joblib output.
- Live HTTP checks confirm model/report/file SHA agreement, causal 100-row prefix equality with full inference, hidden replay labels/metrics, method warm-up 2/30/30, unchanged audit bytes during development exploration, API docs 200 and old-model replay conflict 409.
- Independent Model QA reconciled all seven study/audit reports, groups, per-recording sums, stored OOF scores and provenance without another audit selection. No discrepancies were found. The reviewer confirmed both the target and retrospective alert-rate failure.
- Browser: Compare shows 22/28 with unmet-target verdict, separate development form shows 25/29 at budget 2, and observed sensor deviations carry the correct interpretation. Desktop 1487 px and mobile 390 px have no document overflow. Current screenshots appear in SCREENSHOTS.md.

No Git push, public deployment, physical-equipment test, physical-phone test or new independent blind test was performed. Original V1 preserved hashes still match. Source and runtime remain in D:/Projects/signal-watch; work/ holds reproduction scratch/evidence and runtime/ remains local.

## Historical V1 verification

The record below documents the original release and its superseded active-model values. Current results and sources are above and in MODEL_CARD.md / ML_EVALUATION.md.

# Local verification record

Date: 2026-10-23. Scope: the selected black/amber local SignalWatch application, real SKAB study, same-origin Python API and English documentation. No remote repository, profile or deployment changed.

## Automated checks

| Check | Observed result |
|---|---|
| `.venv/Scripts/python.exe -m pytest -q --basetemp=work/pytest-final` | 26 passed; one Starlette warning about the httpx TestClient integration |
| `.venv/Scripts/python.exe -m pip check` | No broken requirements |
| `npm run typecheck` in frontend | Passed after the final source edits |
| `npm test` in frontend | 4 passed: CSV headers, unknown versus zero values, readable API errors |
| `npm run build` in frontend | Production build passed; bundled ECharts triggers Vite's chunk-size advisory |
| `npm run test:sites` in frontend | 4 packaging/static-route tests passed; no Sites publication performed |
| English-source scan | No Cyrillic matches in owned Python/TypeScript/UI/docs/README files |

The backend tests cover strict actual CSV parsing, causality and real-artifact prefix checks, train/calibration/test boundaries, gaps, onset matching and physical exposure, label-free quality, local storage, replay bounds, input errors, frozen-report preservation, export and training failure/conflict handling.

Two new concurrency regressions first failed with HTTP 200 after deletion, then passed with HTTP 404: parsing and scoring cannot repopulate caches after an upload is deleted. Upload publication/deletion share a lifecycle lock. Replay creation rechecks the upload inside that lock; replay operations and export also maintain a consistent model/recording snapshot.

## Independent reviews

- Model QA Specialist independently rescored all 23 final recordings and prefixes at 0, 1, 30, 31, 100 and halfway. It found the original continuous-alarm matching error. Corrected onset_v2 reporting passed review with the model parameters, thresholds and split preserved.
- Code Reviewer reproduced the deletion/cache race and stale frontend reload ID. Both fixes passed follow-up review, including a delayed replay creation returning 404 after deletion with zero remaining sessions.
- UI Designer compared the approved concept and clear desktop/mobile captures. No P0/P1 findings. Root fixed the three P2 findings: filtered episode counts, an inline timestamp-gap legend, and 44-pixel mobile action targets.

The configured implementation-reviewer and design-reviewer agents could not run because their assigned models were unavailable on this account. The reviews above used the available Code Reviewer and UI Designer profiles. Failed agent launches do not count as reviews.

## Browser and live API checks

Chrome desktop 1487 x 1058, medium 1100 x 640, mobile 390 x 844 and narrow 304 x 844. These are viewport checks on Windows, not physical-phone or cross-browser certification.

| Flow | Observed evidence |
|---|---|
| Audit, filter and event selection | Missed-event selection updates the highlight; empty filters show an empty state; filtered count reads 1 of 2 |
| Sensor inspection | The sensor selector changes the actual chart; a top-residual sensor button opens its channel |
| Sequential replay | Start at 0, step to 1, play/pause to a larger prefix; labels hidden; warm-up visible |
| Persistence and model change | Saved cursor survives server restart; an older-model session returns 409; explicit restart resets it; reopening restores the new cursor |
| Frozen comparison | UI shows 5/28, 4/28 and 8/28 with normal-alert burden, false onsets and delays |
| Validation budget 2 | UI shows a separate validation_only result; final report hash unchanged by exploration |
| Report-to-event link | other-2-event-0 opens other-2, Annotated interval 1, detected, first alert 06:26 and 32-second delay |
| CSV import | In-app browser file picker rejects duplicate headers; accepts 150 real-source rows with no labels and explicit mapping |
| Unlabelled CSV | UI says labels/quality unavailable and marks episodes unlabelled; API metrics are null |
| Export | Chrome downloads JSON; downloaded source hash and model/view hash match the imported recording and artifact |
| Deletion | UI confirmation and Cancel checked. The local API deletes the disposable QA upload and related replay; subsequent reads return 404 and the catalog omits it. Final destructive UI confirmation was not clicked |
| Training | Train locally becomes busy, then completed; active artifact/report reload |
| API docs | Swagger UI renders the real routes; /docs and /openapi.json return HTTP 200 |
| Responsive layout | No page-level horizontal overflow on Replay/Compare/Data & Model at 390px; Replay also fits at 304px. Tables and episode queues scroll within their own regions |
| Keyboard | Tab reaches the visible skip link and navigation; amber 2px focus outline; Return opens Compare |
| Reduced motion | Emulated reduce preference matches and button transition duration is 0s; chart animation is disabled. Emulation restored |

The in-app browser's screenshot capture smeared DOM text, so final visual review used clear Chrome captures. Chrome's extension blocked file uploads without extra file-URL permission; the same upload test succeeded in the in-app browser without changing extension settings. Console inspection found unrelated MetaMask extension errors; no application runtime error appeared in the checked flows.

## Reproduction and current artifacts

Train locally reproduces the study; it does not optimize against the final test. The selected candidate, thresholds, aggregate final metrics, per-recording evidence and group slices match the pre-job result exactly. Model serialization bytes differ. The report also gained the source-generated burden/precision fields inside historical validation-selection trials. Model SHA, metadata timestamps and training duration update, so the report file is not byte-identical after retraining.

| Artifact | Current SHA-256 |
|---|---|
| model.joblib | 1c8554a484148129b281e7aa6657d078658d778b3cff6edbf3f4f16594a36cc2 |
| frozen_evaluation.json | 0d6715e66214ea04a3027123647eddd530b05ac9532629a50b4b5569d0f30791 |
| split_manifest.json | 870af2b6bb046fa6acba529d62163a5f44d893e09d6607f6eb1b6fb1ba77b941 |
| frozen_evaluation_legacy_overlap.json | 4f8ba3e8a9fc8031e9d9d78cede65b03e9730efa084f863b5febfbc8ee420699 |

Evidence JSON: [reproduction check](evidence/reproduction-check.json), [live workflows](evidence/live-workflows.json), [data/artifact/screenshot reconciliation](evidence/final-check.json). The last check verifies all 35 bundled source-file hashes, model/report identity, image format/dimensions and local Markdown links. Browser captures retain their original JPEG bytes and matching extensions. Runtime and scratch outputs are ignored by Git. Curated, described screenshots live in docs/screenshots/.

## Limits of completion

The agreed local application and study are implemented. The trained detector fails its operational burden check: 97.63% of observed normal time in alert, 5/28 event hits. No production usefulness, superiority, transfer, live equipment connection or new algorithm claim follows from this result. The original model/data cards retain the details.

No physical phone, screen-reader session, macOS/Linux install, clean external machine, public server or GitHub publication was tested in this scope. The README includes portable commands, but the recorded checks ran on this Windows environment. No new participant data was collected. Source license selection and public deployment remain outside this task.
