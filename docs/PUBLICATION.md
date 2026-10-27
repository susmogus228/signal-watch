# GitHub publication record

Published: **2026-10-27** at [susmogus228/signal-watch](https://github.com/susmogus228/signal-watch). The repository is public, its default branch is main, and GitHub Pages remains disabled. This publication includes the local workbench, public dataset, saved models, V1-V13 research evidence and README banner. Running the application requires the Python server described in the README.

## Verified source snapshot

Initial publication commit: [32fd73d7a226ce19bc1f63303ed0609af2d9da26](https://github.com/susmogus228/signal-watch/commit/32fd73d7a226ce19bc1f63303ed0609af2d9da26).

- The remote tree matches all 388 local source-snapshot blobs and their modes.
- An unauthenticated request receives HTTP 200 for the repository. Downloads of the pinned README, banner and Pressure trade-off PNG match their local SHA-256 values.
- A clean Git checkout preserves all 172 protected model/result/data files and all 75 frozen V13 dependencies. The saved-results exporter check passes there using the existing application Python environment.
- Git attributes preserve exact bytes across line-ending settings. This matters for the pinned dataset and research hashes.

The publication record pins the initial scientific snapshot. Later documentation-only commits record verification without changing models or measurements. Earlier packaging evidence describes the pre-banner, pre-publication phase; its historical document hashes are not a claim that today's README still has those bytes.

## Banner

[Original PNG](assets/signalwatch-banner.png), 2172 x 724 pixels, 3:1. SHA-256: 7a784c68611541f2d1d1a8ac51660333e612414a80fd953ea5b71dae69c54170.

The README banner loads on the actual public GitHub page at its original dimensions. It is a generated concept illustration, not measured telemetry. [Prompt and provenance](assets/BANNER.md) identify the built-in image_gen workflow. Existing portfolio banners remain unchanged.

## Checks performed for publication

| Check | Result |
|---|---|
| Frontend TypeScript check | PASS |
| Frontend logic tests | 4 passed |
| Production frontend build | PASS |
| Existing Sites packaging tests | 4 passed; no Sites deployment |
| Saved-results exporter in clean Git checkout | PASS, no fitting/inference |
| Protected file/dependency hashes | 172/172 and 75/75 match |
| Staged publication-content review | PASS, including banner delta |
| Public repository and pinned asset access | HTTP 200; downloaded hashes match |

Vite retains its bundle-size advisory: the main JavaScript chunk is 859.23 kB, 279.16 kB gzip. This build result does not establish performance optimisation or deployment readiness. The clean-checkout exercise is not an external clean-machine dependency installation or field-equipment test.

The independent publication review used a generic available agent after the named security profile failed to start. It checked staged paths, explicit secret patterns and dataset notices; it did not deserialize binary model files or perform a full AppSec/legal certification. Work directories, local uploads/sessions, environments, credentials and dependency caches are outside the Git snapshot. The 35 public SKAB CSVs retain their manifest hashes and upstream GPL-3.0 notices. No new project-source license was selected.

## Research boundary

New ML training and cutoff tuning remain paused. V5 remains active, V13 remains a rejected activation candidate, and R17/R20 model-quality goals stay open. Publication runs no new model fit, quality evaluation or training API call. The earlier 192-test backend result belongs to V13 source verification and was not repeated or relabelled here.

[Machine-readable publication evidence](evidence/github-publication-2026-10-27.json) records the source snapshot, public-access checks and scope. [RESULTS.md](RESULTS.md) remains the main account of measured performance and limitations.
