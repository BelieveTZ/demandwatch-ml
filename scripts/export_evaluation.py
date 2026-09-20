"""Render measured results into the repository's evaluation record."""

import json
from pathlib import Path


def main():
    report = json.loads(Path("demandwatch/web/data/report.json").read_text())
    rows = [
        "# Evaluation record",
        "",
        f"Run `{report['run_id']}`. Recorded at {report['created_at']}. Python 3.12, CPU only.",
        "",
        "The checked-in [JSON](../demandwatch/web/data/report.json) and "
        "[per-hour CSV](../demandwatch/web/data/predictions.csv) are the measured evidence.",
        "",
        "## Source and eligibility",
        "",
        f"Parsed {report['dataset']['raw_rows']:,} source rows with "
        f"{report['dataset']['missing_hours']:,} absent hourly timestamps. "
        f"{report['dataset']['eligible_rows']:,} eligible rows remain after strictly historical "
        "feature requirements. The first week and incomplete histories are excluded.",
        "",
        f"CSV SHA-256: `{report['dataset']['csv_sha256']}`.",
        "",
        "## Frozen protocol",
        "",
        "| Block | Rows | Start | End |",
        "| --- | ---: | --- | --- |",
    ]
    for name, block in report["splits"].items():
        rows.append(f"| {name} | {block['n']:,} | {block['start']} | {block['end']} |")
    rows += [
        "",
        "Four candidates (one Ridge, three HGB configurations) are ranked by validation "
        "MAE. The winner is refit on train + validation, frozen, then calibrated. "
        "The final test does not select parameters or interval width.",
        "",
        "| Candidate | Validation MAE | Configuration |",
        "| --- | ---: | --- |",
    ]
    for candidate in report["validation"]:
        rows.append(
            f"| {candidate['model']} | {candidate['mae']:.3f} "
            f"| `{json.dumps(candidate['parameters'], sort_keys=True)}` |"
        )
    rows += [
        "",
        "## Final test",
        "",
        "| Model | MAE | RMSE | Hours |",
        "| --- | ---: | ---: | ---: |",
    ]
    for result in report["metrics"]:
        rows.append(
            f"| {result['model']} | {result['mae']:.3f} | {result['rmse']:.3f} | {result['n']:,} |"
        )
    chosen = next(row for row in report["metrics"] if row["model"] == report["selected_model"])
    baseline = next(row for row in report["metrics"] if row["model"] == "Seasonal 168h")
    improvement = (1 - chosen["mae"] / baseline["mae"]) * 100
    uncertainty = report["intervals"]
    rows += [
        "",
        f"Selected: **{report['selected_model']}**. MAE reduction versus Seasonal 168h: "
        f"**{improvement:.2f}%** on identical eligible rows.",
        "",
        f"90% target interval: empirical test coverage **{uncertainty['coverage']:.2%}**, "
        f"mean total width **{uncertainty['mean_width']:.2f}** rides. "
        f"Residual radius **{uncertainty['radius']:.2f}**, fitted using "
        f"{uncertainty['calibration_n']:,} calibration hours. The nonnegative floor means "
        "mean total width can be less than twice the radius.",
        "",
        "## Error slices",
        "",
        "| Slice | Hours | MAE | RMSE | Coverage |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for item in report["slices"]:
        rows.append(
            f"| {item['slice']} | {item['n']} | {item['mae']:.2f} "
            f"| {item['rmse']:.2f} | {item['coverage']:.2%} |"
        )
    rows += [
        "",
        "Peak-hour coverage falls below the nominal target even though aggregate coverage "
        "is higher. A single absolute-error radius overcovers quiet hours and undercovers "
        "busy ones. This run retains that failure rather than retuning on the test period.",
        "",
        "## Timing and monitoring",
        "",
    ]
    benchmark = report["benchmark"]
    rows += [
        f"Warm single-row prediction: p50 **{benchmark['p50_ms']:.3f} ms**, "
        f"p95 **{benchmark['p95_ms']:.3f} ms**, n={benchmark['n']}. "
        "One CPU thread; includes DataFrame feature selection and estimator prediction; "
        "excludes network, HTTP parsing, SQLite and cold start.",
        "",
        f"Platform: {benchmark['platform']}; processor: {benchmark['processor']}.",
        "",
        "PSI uses calibration-period fixed quantile bins, explicit out-of-support bins and "
        "0.5-count smoothing. 0.2 is a review heuristic. Calendar/time trend shifts are expected. "
        "The synthetic monitor test adds 0.25 to prior temperature (clipped to [0,1]) and "
        "multiplies demand lags/means by 1.6. It is not an accuracy experiment.",
        "",
        "## Limits",
        "",
    ]
    rows.extend(f"- {note}" for note in report["caveats"])
    rows += [
        "",
        "Validation permutation importance uses the train-only winner; correlated "
        "lag features make this approximate model reliance, not causal explanation. "
        "No city-transfer, probabilistic calibration under drift, or multi-day forecast claim.",
        "",
    ]
    Path("docs/EVALUATION.md").write_text("\n".join(rows), encoding="utf-8")


if __name__ == "__main__":
    main()
