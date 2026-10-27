# Research figures and inspectable data

Snapshot: 2026-10-27. These assets present saved experiments; the exporter performs no training, model inference or cutoff search. Start with [the results report](../RESULTS.md).

| Figure | Question | Source |
|---|---|---|
| [Audit PNG](figures/audit-tradeoff.png) / [SVG](figures/audit-tradeoff.svg) | Does higher event recall also control false starts? | V4/V5/V7 saved retrospective methods, identical 18,208 normal seconds |
| [V13 trade-off PNG](figures/pressure-tradeoff.png) / [SVG](figures/pressure-tradeoff.svg) | Are fewer false starts accompanied by gradual-event retention and acceptable delay? | V13 report's predeclared paired V12/V13 frontiers |
| [Pressure conditioning PNG](figures/pressure-conditioning.png) / [SVG](figures/pressure-conditioning.svg) | Did the training-bank scale defect disappear? | V13 K4 fit evidence, same bank/center and old/new scales |

The README embeds PNG for broad compatibility. SVG exports retain vector geometry for zooming. All figures use a static dark background, direct values and labelled units. Hatching/marker shapes supplement colour. Counts start at zero; the Pressure diagnostic explicitly uses a logarithmic axis. No animation or script is embedded.

## Data and provenance

- [chart-data.json](chart-data.json): full-precision plotted values, cohort definitions, source-file SHA-256 and JSON pointers. Earlier V1 uses different warm-up/exposure; it is not part of the common-exposure V4/V5/V7 chart.
- [Retrospective CSV](retrospective-results.csv): six scored versions, including observed normal exposure. A missing/unscored version is not exported as zero.
- [V12/V13 frontier CSV](v13-paired-frontiers.csv): own cutoffs, recall, false starts, normal-alert time and both gradual delays.
- [Pressure CSV](pressure-training-banks.csv): each fold's old/new scale and bank magnitudes. Old magnitude is derived as new magnitude times new_scale/old_scale, with unchanged bank and median.
- [manifest.json](manifest.json): exact source, exporter and output hashes; renderer version. Source paths are relative to the repository root.

## Check the supplied exports

```powershell
.\.venv\Scripts\python.exe scripts/build_research_summary.py --check
```

On macOS/Linux, substitute .venv/bin/python. This standard-library check needs no Matplotlib installation and does not load joblib models. It checks the existing image hashes rather than claiming to re-render images without a renderer.

## Regenerate figures

Use a separate authoring environment with **Matplotlib 3.11.2**. The application lock deliberately does not add plotting dependencies. For example, from the repository root:

```powershell
python -m venv work/figure-env
.\work\figure-env\Scripts\python.exe -m pip install matplotlib==3.11.2
.\work\figure-env\Scripts\python.exe scripts/build_research_summary.py
```

For macOS/Linux use work/figure-env/bin/python. The script reads ten saved JSON files and writes the chart JSON, three CSVs, three PNG/SVG pairs and a manifest under docs/research/. It uses the noninteractive Agg backend and the bundled DejaVu Sans font. Rendering does not call application training or inference.

The supplied images were rendered with an existing local authoring environment using Matplotlib 3.11.2 and Pillow 13.0.0. Different library/font/platform versions can change image bytes. Full-precision source values control the scientific claims; hashes identify this particular export. See [resource provenance](../RESOURCES.md) for documentation-version access limits.
