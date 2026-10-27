# Design QA

Result: **passed for the checked local desktop/mobile scope**, 2026-10-23. Model utility is a separate failed result, explained in the interface and model card.

Reference: docs/design/approved-event-review.png, selected option 2. Actual: docs/screenshots/audit-desktop.jpg. Source viewport: 1487 x 1058; the final full-page capture includes content below the viewport.

The implementation retains the reference's black base, amber accent, compact top navigation, recording strip, left event queue, broad signal canvas, score graph, aligned timeline and paired evidence panels. Actual measurements, extra replay controls and explicit audit/provenance labels replace the illustrative concept contents. The UI makes no inferred mechanical diagnosis.

Independent UI review found no P0/P1 issues. Root fixed the three P2 findings and recaptured the affected views:

| Finding | Fix and check |
|---|---|
| Total episode count persisted during filtering | Counter shows 1 of 2 for one missed event; empty detected filter shows 0 of 2 |
| Red gap band lacked a nearby explanation | Signal header labels Red band: timestamp gap; full audit capture shows it beside the sensor selector |
| Mobile actions were 38/32px | Main actions and contributing-sensor buttons measure 44px at 390px and 304px |

Root checked 1487, 1100, 390 and 304-pixel layouts. DOM client/scroll widths match at 390 and 304px; queue/table scrolling stays local. Tab exposes the amber 2px focus outline and Return opens Compare. Reduced-motion emulation yields 0s transitions; chart animation stays disabled. Temporary viewport/media overrides were reset after testing.

The root also checked actual loading, empty filter, replay warm-up, unknown metrics without labels, duplicate CSV headers, changed-model restart and training completion. Browser interaction details and limitations: docs/VERIFICATION.md. Physical phone, screen reader and other operating systems remain untested. Defective in-app screenshots are excluded from docs/screenshots/; final review uses clear Chrome captures.
