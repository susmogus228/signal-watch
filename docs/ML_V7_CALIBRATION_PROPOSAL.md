# Proposed V7: stable calibrated ML ensemble

Status: user approved the proposed ML architecture on 2026-10-23; one frozen development experiment and one selected retrospective audit completed; acceptance failed, V5 retained. Root: D:/Projects/signal-watch. Existing Python/scikit-learn/FastAPI and React/ECharts stack, pinned SKAB data and black/amber design stay. No paid service, external inference, new personal data, dependency installation, Git push or deployment.

## Problem and evidence

V5 has26/28 retrospective detections with56 false starts over18,208 normal seconds (11.072/h). Its reused grouped OOF gives28/29 and3 false starts. V6 elapsed confirmation removes false starts but loses short true development events; its frozen acceptance keeps V5 active. The underlying score needs better generalization, not just delayed output.

## Proposed model change

Keep the V5 feature definitions and the two model families. Replace the single final known-state HGB classifier with an ensemble of three record-group-disjoint classifier/calibrator pairs. Each pair fits the HGB/preprocessing on two thirds of development recordings and fits a sigmoid calibrator on its remaining third. Do not use class-balanced calibration to present artificial prevalence as a real probability. Average pair outputs. This is an architecture change: three fitted classifiers and calibrators replace one classifier. Keep the normal-only ExtraTrees predictor configuration.

Use nested grouped evaluation. Each of the three outer folds holds out complete recordings. Inside the remaining recordings, fit three classifier/calibrator pairs with disjoint fitting/calibration groups. No outer-held-out rows or labels enter preprocessing, HGB training or calibration. Reuse only the matching normal-predictor channel OOF from V5; its outer folds must match exactly. For deployment fit three final pairs on the twelve development recordings and refit the same normal predictor. Deploy those exact calibrated pairs, with no subsequent full-data classifier refit.

This targets variance and calibration transfer. It does not guarantee field-calibrated probabilities or lower false alarms. Feature selection, architecture expansion and automatic test-time learning are excluded.

## Bounded selection proposal

Freeze identity V5 control and ensemble fusion with maximum, top-two mean and all-channel mean residual aggregation. For each branch, use the fixed normal-OOF quantiles [.95,.975,.99,.995,.999,1] plus nextafter(max,+inf): at most49 threshold pairs per aggregation. No temporal wait in this study; isolate the learned-score change.

The user target25/28 implies a development floor of26/29 when applying the same minimum recall proportion. Explicitly allow losing some prior development events, while reporting every lost/gained ID; V6's stronger all28-retained constraint does not apply to this proposal. Require <=2 false starts/normal hour and <=5% normal time in alert. To qualify as a burden improvement, a new point must also have fewer false starts than V5's3 and no more than its37 normal-alert seconds. Among these qualifying points, rank by more event detections, fewer false starts, lower normal-alert time, lower measured delay and fixed candidate/threshold order. V5 is the fallback, not a competing point that overrides the explicitly permitted recall floor. No qualifying point means retain V5 and skip the retrospective audit. No fitting/selection on old28.

If development supports a winner, freeze it and conduct one retrospective regression audit. Activation still requires >=25/28, fewer than56 false starts, no greater normal-alert time and unchanged exposure. The <=2/h objective remains explicit. Failure retains V5 without testing other variants on old28. Reused development and retrospective records remain optimistic; they are not new blind evidence.

## Deliverables and verification

New versioned module/study/tests/artifacts and source/dependency manifests; actual nested-group training/calibration provenance; raw-versus-new event comparison with false burden and delay; staged API training reproduction before activation; accurate model card/API/UI and desktop/mobile checks. The existing API routes remain; model metadata identifies the ensemble and its calibration groups, without summing overlapping training examples.

Root owns integration and documentation. AI Engineer owns only the new scientific module/study/tests. Model QA Specialist independently reviews leakage, calibration boundaries, selection, counts and reproduction, read-only with an8-minute phase timebox. Codex memory preserves agreed requirements; the fresh Codebase Memory list has no SignalWatch graph, so source fallback applies.

## Primary references

- [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html): disjoint classifier/calibrator data; calibrated fold ensembles versus a final classifier refit.
- [scikit-learn threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html): separate model fitting and decision selection.
- [Cobb et al. sequential false detections](https://proceedings.mlr.press/v151/cobb22a.html): false-alarm operating points require their own evaluation.

No paper code or performance result is copied. This proposal promises an experiment and acceptance checks, not a guaranteed metric.
