"""Export saved research results and static figures. No fitting or inference.

Run with --check using the application Python to verify source/output identity.
Rendering additionally needs Matplotlib 3.11.2 in an authoring environment;
the application dependency lock remains unchanged.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "research"
FIGURES = OUT / "figures"
FIELDS = (
    "total_events", "detected_events", "missed_events", "false_episodes",
    "observed_normal_seconds", "false_episodes_per_hour", "false_alert_seconds",
    "normal_alert_fraction", "median_delay_seconds", "mean_delay_seconds",
)
AUDITS = {
    "V1": "artifacts/v1/frozen_evaluation.json",
    "V2": "artifacts/v2/retrospective_audit.json",
    "V3": "artifacts/supervised/retrospective_audit.json",
    "V4": "artifacts/supervised-weighted/retrospective_audit.json",
    "V5": "artifacts/hybrid/retrospective_audit.json",
    "V7": "artifacts/calibrated/retrospective_audit.json",
}
SOURCE_HASHES: dict[str, str] = {}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: str) -> dict:
    source = ROOT / path
    SOURCE_HASHES[path] = digest(source)
    return json.loads(source.read_text(encoding="utf-8"))


def metrics(values: dict) -> dict:
    result = {k: values[k] for k in FIELDS}
    assert result["detected_events"] + result["missed_events"] == result["total_events"]
    assert abs(result["false_episodes_per_hour"] -
               result["false_episodes"] * 3600 / result["observed_normal_seconds"]) < 1e-10
    assert abs(result["normal_alert_fraction"] -
               result["false_alert_seconds"] / result["observed_normal_seconds"]) < 1e-10
    return result


def collect() -> dict:
    SOURCE_HASHES.clear()
    audits = [dict(version=v, source=p, json_pointer="/methods/learned",
                   **metrics(read(p)["methods"]["learned"])) for v, p in AUDITS.items()]
    development = []
    for v, p in [("V5", "artifacts/hybrid-development/report.json"),
                 ("V7", "artifacts/calibrated-development/report.json")]:
        development.append(dict(version=v, source=p, json_pointer="/metrics",
                                **metrics(read(p)["metrics"])))
    path = "artifacts/pressure-scaled-development/report.json"
    report = read(path)
    assert report["status"] == "no_feasible_candidate" and report["selected"] is None
    assert report["actual_top_level_fit_calls"] == 18
    assert not any(report[k] for k in ("activation", "old_test_scored", "teaser_scored"))
    pairwise = []
    for arm in ("K1", "K2", "K4"):
        for version in ("v12", "v13"):
            point = report["v12_same_k_comparisons"][arm][version]["progressive_recall_frontier"]
            events = {e["id"]: e for e in point["paired_outcomes"]}
            linear, exponential = (events[n] for n in ("other-6-event-0", "other-9-event-0"))
            assert linear["candidate_status"] == exponential["candidate_status"] == "detected"
            pairwise.append(dict(
                arm=arm, version=version.upper(), threshold=point["threshold"],
                **metrics(point["metrics"]),
                linear_delay_seconds=linear["candidate_delay_seconds"],
                exponential_delay_seconds=exponential["candidate_delay_seconds"],
                source=path,
                json_pointer=f"/v12_same_k_comparisons/{arm}/{version}/progressive_recall_frontier",
            ))
    fit_path = "artifacts/pressure-scaled-development/K4_fit_evidence.json"
    pressure = []
    for fold in read(fit_path)["folds"]:
        p = fold["pressure_scaling"]
        pressure.append(dict(
            fold=fold["fold"], old_scale=p["original_pressure_scale"],
            new_scale=p["new_pressure_scale"],
            old_bank_max_abs_z=p["bank_pressure_max_abs_z"] *
            p["new_pressure_scale"] / p["original_pressure_scale"],
            new_bank_max_abs_z=p["bank_pressure_max_abs_z"],
            guard_bound=p["guard_bound"], source=fit_path,
            json_pointer=f"/folds/{fold['fold']}/pressure_scaling",
        ))
    return dict(
        schema_version=1, evidence_as_of="2026-10-27", training_status="paused",
        active_version="V5", purpose="Presentation of saved experiments only",
        retrospective_audits=audits, development=development,
        v13_progressive_frontiers=pairwise, pressure_training_banks=pressure,
        pressure_old_max_derivation="new_bank_max_abs_z * new_scale / old_scale; same bank and center",
        interpretation=[
            "Previously inspected retrospective records; no new blind evaluation.",
            "V1 warm-up/eligible exposure differ from later versions; no common-exposure claim.",
            "V12/V13 frontier cutoffs differ; every displayed frontier retains both gradual intervals and 27/29 events.",
            "Pressure scaling also changes physical regularization and initialization geometry.",
        ],
        source_files_sha256=dict(SOURCE_HASHES),
    )


def csv_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def write_data(data: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "chart-data.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    for name, rows in [("retrospective-results", data["retrospective_audits"]),
                       ("v13-paired-frontiers", data["v13_progressive_frontiers"]),
                       ("pressure-training-banks", data["pressure_training_banks"])]:
        (OUT / f"{name}.csv").write_text(csv_text(rows), encoding="utf-8", newline="")


def render(data: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    FIGURES.mkdir(parents=True, exist_ok=True)
    bg, ink, muted, accent, before = "#0f0f0f", "#f4f2ec", "#b5b5b2", "#f3b24b", "#8d97a2"
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 13, "text.color": ink,
        "axes.facecolor": bg, "figure.facecolor": bg, "savefig.facecolor": bg,
        "axes.labelcolor": muted, "xtick.color": muted, "ytick.color": ink,
        "axes.edgecolor": "#575757", "svg.hashsalt": "signalwatch-research-snapshot",
    })

    def style(ax):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#303030", linewidth=.7)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0, pad=10)

    def save(fig, name, title, subtitle, note):
        fig.suptitle(title, x=.08, ha="left", fontsize=22, fontweight="bold", y=.98)
        fig.text(.08, .915, subtitle, fontsize=11.5, color=muted, va="top", linespacing=1.5)
        fig.text(.08, .022, note, fontsize=10.5, color=muted, va="bottom", linespacing=1.5)
        fig.tight_layout(rect=(.02, .11, .98, .85), h_pad=3)
        fig.savefig(FIGURES / f"{name}.png", dpi=220, metadata={"Software": "SignalWatch saved-report exporter"})
        fig.savefig(FIGURES / f"{name}.svg", metadata={"Date": None, "Creator": "SignalWatch saved-report exporter"})
        plt.close(fig)

    selected = [r for r in data["retrospective_audits"] if r["version"] in ("V4", "V5", "V7")]
    assert {r["observed_normal_seconds"] for r in selected} == {18208.0}
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.1))
    labels = ["V4  archived", "V5  active", "V7  rejected"]
    for ax, field, title, maximum, fmt in [
        (axes[0], "detected_events", "Detected annotation segments / 28", 31, lambda x: f"{int(x)}/28"),
        (axes[1], "false_episodes_per_hour", "False starts per observed normal hour", 19, lambda x: f"{x:.3f}/h"),
    ]:
        values = [r[field] for r in selected]
        ax.barh(labels, values, height=.52, color=[before, accent, before])
        ax.invert_yaxis(); ax.set_xlim(0, maximum); style(ax)
        ax.set_title(title, loc="left", fontsize=14, pad=13)
        for index, value in enumerate(values):
            ax.text(value + maximum*.018, index, fmt(value), va="center", fontsize=14, color=ink,
                    bbox=dict(facecolor=bg, edgecolor="none", pad=1))
    axes[0].set_xticks([0, 7, 14, 21, 28]); axes[0].axvline(25, color=ink, linestyle="--", linewidth=1)
    axes[1].axvline(2, color=ink, linestyle="--", linewidth=1)
    save(fig, "audit-tradeoff", "More detections, more false alerts",
         "Same retrospective corpus: 28 segments; 18,208 observed normal seconds.\nActive V5 meets the recall target and misses the false-alert-rate goal.",
         "Dashed goals: 25 detected segments; 2 false starts/hour.\nPreviously inspected data, not a blind test. Bars start at zero; lower false rate is better.")

    rows = {(r["arm"], r["version"]): r for r in data["v13_progressive_frontiers"]}
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.7))
    for ax, field, title, maximum in [
        (axes[0], "false_episodes", "False starts on labelled-normal observations", 46),
        (axes[1], "linear_delay_seconds", "Linear-defect detection delay (seconds)", 55),
    ]:
        for j, version in enumerate(("V12", "V13")):
            positions = [i + (j-.5)*.32 for i in range(3)]
            values = [rows[(arm, version)][field] for arm in ("K1", "K2", "K4")]
            ax.barh(positions, values, height=.27, label=version,
                    color=before if version == "V12" else accent,
                    hatch="///" if version == "V12" else None)
            for y, value in zip(positions, values):
                ax.text(value + .65, y, f"{value:g}", va="center", fontsize=13)
        ax.set_yticks(range(3), ["K1", "K2", "K4"]); ax.invert_yaxis()
        ax.set_xlim(0, maximum); ax.xaxis.set_major_locator(MaxNLocator(integer=True)); style(ax)
        ax.set_title(title, loc="left", fontsize=14, pad=15)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper right", bbox_to_anchor=(.98,.86),
               frameon=False, ncol=2, fontsize=11)
    save(fig, "pressure-tradeoff", "Pressure correction: a mixed result",
         "Every point retains 27/29 events and both gradual intervals.\nK4 false starts fall 40 to 21; its linear delay rises 3 to 46 seconds.",
         "Own cutoff per arm/version; same 9,179 normal seconds. K = mixture components.\nDescriptive development frontiers, not accepted models. V13 adds no waiting rule.")

    fig, ax = plt.subplots(figsize=(8.5, 6.1))
    for j, (version, key, color) in enumerate([
        ("V12", "old_bank_max_abs_z", before), ("V13", "new_bank_max_abs_z", accent)
    ]):
        values = [r[key] for r in data["pressure_training_banks"]]
        x = [i + (j-.5)*.18 for i in range(3)]
        ax.scatter(x, values, color=color, s=80, marker="s" if j == 0 else "o", label=version, zorder=3)
        for pos, value in zip(x, values):
            ax.annotate(f"{value:,.2f}", (pos,value), xytext=(0,12 if j==0 else -19),
                        textcoords="offset points", ha="center", fontsize=10.5, color=color)
    ax.set_yscale("log"); ax.set_ylim(1.5, 1e7); ax.set_xlim(-.45, 2.45)
    ax.set_xticks(range(3), ["Fold 0", "Fold 1", "Fold 2"])
    ax.set_ylabel("Maximum absolute normalized Pressure (log scale)", fontsize=12)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(axis="y", color="#303030", linewidth=.7)
    ax.axhline(128, color=ink, linestyle="--", linewidth=1, label="V13 bank guard = 128")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5)
    save(fig, "pressure-conditioning", "Numerical conditioning is corrected",
         "Maximum absolute Pressure z on each K4 normal-training bank.\nFold 0 keeps the same scale; folds 1/2 no longer expand values to millions.",
         "Log scale; the guard applies only to training banks. No clipping or held-out fitting.\nScaling also changes physical regularization and initialization; this is not alarm-quality proof.")


def check(data: dict) -> None:
    assert json.loads((OUT / "chart-data.json").read_text()) == data, "Chart data/source mismatch"
    for filename, rows in [("retrospective-results", data["retrospective_audits"]),
                           ("v13-paired-frontiers", data["v13_progressive_frontiers"]),
                           ("pressure-training-banks", data["pressure_training_banks"])]:
        with (OUT / f"{filename}.csv").open(encoding="utf-8", newline="") as stream:
            assert stream.read() == csv_text(rows)
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["generator_sha256"] == digest(Path(__file__)), "Generator changed since export"
    assert manifest["source_files_sha256"] == SOURCE_HASHES
    expected = {"docs/research/chart-data.json"}
    expected.update(f"docs/research/{name}.csv" for name in
                    ("retrospective-results", "v13-paired-frontiers", "pressure-training-banks"))
    expected.update(f"docs/research/figures/{name}.{extension}" for name in
                    ("audit-tradeoff", "pressure-tradeoff", "pressure-conditioning")
                    for extension in ("png", "svg"))
    assert set(manifest["outputs_sha256"]) == expected, "Missing or unexpected manifest output"
    for relative, expected in manifest["outputs_sha256"].items():
        assert digest(ROOT / relative) == expected, relative
    print("PASS: saved sources, chart JSON, three CSV exports and six figure files match.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify exports without importing plotting libraries")
    args = parser.parse_args()
    data = collect()
    if args.check:
        check(data)
        return
    write_data(data)
    render(data)
    import matplotlib
    files = [OUT / "chart-data.json", *OUT.glob("*.csv"), *FIGURES.glob("*.png"), *FIGURES.glob("*.svg")]
    manifest = dict(schema_version=1, evidence_as_of=data["evidence_as_of"],
                    generator="scripts/build_research_summary.py", generator_sha256=digest(Path(__file__)),
                    matplotlib_version=matplotlib.__version__, training_or_model_inference=False,
                    source_files_sha256=SOURCE_HASHES,
                    outputs_sha256={str(p.relative_to(ROOT)).replace("\\", "/"):digest(p) for p in sorted(files)})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(files)} evidence/figure files from {len(SOURCE_HASHES)} saved JSON sources.")


if __name__ == "__main__":
    main()
