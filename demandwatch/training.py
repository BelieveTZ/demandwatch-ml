"""Chronological model selection, frozen calibration, and final evaluation."""

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from demandwatch.data import FEATURES, load_source, make_features, split_blocks
from demandwatch.monitoring import drift_report, make_reference, regression_metrics
from demandwatch.uncertainty import calibrate_radius, interval

SEED = 42
CANDIDATES = [
    {"max_leaf_nodes": 15, "learning_rate": 0.08, "max_iter": 200, "l2_regularization": 1.0},
    {"max_leaf_nodes": 31, "learning_rate": 0.08, "max_iter": 200, "l2_regularization": 1.0},
    {"max_leaf_nodes": 31, "learning_rate": 0.05, "max_iter": 350, "l2_regularization": 5.0},
]


def ridge_model():
    categorical = ["hour", "weekday", "month", "workingday", "holiday", "weather_previous"]
    numeric = [column for column in FEATURES if column not in categorical]
    transform = ColumnTransformer(
        [
            ("calendar", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    return make_pipeline(transform, Ridge(alpha=10))


def boosted_model(parameters: dict):
    return HistGradientBoostingRegressor(
        **parameters, loss="poisson", early_stopping=False, random_state=SEED
    )


def predict_nonnegative(model, frame):
    return np.maximum(0, model.predict(frame[FEATURES]))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def train(source_path: Path, output: Path, report_path: Path) -> dict:
    with threadpool_limits(limits=1):
        return _train(source_path, output, report_path)


def _train(source_path: Path, output: Path, report_path: Path) -> dict:
    started = time.perf_counter()
    source, metadata = load_source(source_path)
    frame = make_features(source)
    blocks = split_blocks(frame)
    fit, validation, calibration, test = (blocks[key] for key in blocks)
    validation_scores, fitted_candidates = [], []
    candidates = [("Ridge", ridge_model(), {"alpha": 10})] + [
        ("Histogram gradient boosting", boosted_model(parameters), parameters)
        for parameters in CANDIDATES
    ]
    for name, model, parameters in candidates:
        model.fit(fit[FEATURES], fit.target)
        score = regression_metrics(validation.target, predict_nonnegative(model, validation))
        validation_scores.append({"model": name, "parameters": parameters, **score})
        fitted_candidates.append(model)
    selected_index = int(np.argmin([score["mae"] for score in validation_scores]))
    selected_name, _, selected_parameters = candidates[selected_index]
    # No final-test feedback changes selection. Refit BEFORE residual calibration.
    combined = pd.concat([fit, validation])
    ridge = ridge_model().fit(combined[FEATURES], combined.target)
    best_hgb_index = min(range(1, len(candidates)), key=lambda i: validation_scores[i]["mae"])
    hgb = boosted_model(candidates[best_hgb_index][2]).fit(combined[FEATURES], combined.target)
    selected = ridge if selected_index == 0 else hgb
    radius = calibrate_radius(calibration.target, predict_nonnegative(selected, calibration))
    predictions = predict_nonnegative(selected, test)
    lower, upper = interval(predictions, radius)
    covered = (test.target.to_numpy() >= lower) & (test.target.to_numpy() <= upper)
    metrics = [
        {"model": "Seasonal 24h", **regression_metrics(test.target, test.lag_24h)},
        {"model": "Seasonal 168h", **regression_metrics(test.target, test.lag_168h)},
        {"model": "Ridge", **regression_metrics(test.target, predict_nonnegative(ridge, test))},
        {
            "model": "Histogram gradient boosting",
            **regression_metrics(test.target, predict_nonnegative(hgb, test)),
        },
    ]
    slices = []
    slice_masks = {
        "Working day": test.workingday == 1,
        "Non-working day": test.workingday == 0,
        "Morning peak (07–09)": test.hour.between(7, 9),
        "Evening peak (16–19)": test.hour.between(16, 19),
        "Overnight (00–05)": test.hour.between(0, 5),
        "November": test.month == 11,
        "December": test.month == 12,
    }
    for name, mask in slice_masks.items():
        slices.append(
            {
                "slice": name,
                **regression_metrics(test.target[mask], predictions[mask]),
                "coverage": float(covered[mask].mean()),
            }
        )
    # Explanation uses the original train-only winner on validation, not final-test tuning.
    importance = permutation_importance(
        fitted_candidates[selected_index],
        validation[FEATURES],
        validation.target,
        scoring="neg_mean_absolute_error",
        n_repeats=3,
        random_state=SEED,
        n_jobs=1,
    )
    reference = make_reference(calibration[FEATURES])
    stressed = test[FEATURES].copy()
    stressed["temp_previous"] = (stressed.temp_previous + 0.25).clip(0, 1)
    for column in ["lag_1h", "lag_24h", "lag_168h", "mean_24h", "mean_168h"]:
        stressed[column] *= 1.6
    latencies = []
    predict_nonnegative(selected, test.iloc[:1])
    for i in range(100):
        row = test.iloc[i : i + 1]
        before = time.perf_counter()
        predict_nonnegative(selected, row)
        latencies.append((time.perf_counter() - before) * 1000)
    configuration = {
        "features": FEATURES,
        "selected_model": selected_name,
        "parameters": selected_parameters,
        "data_sha256": metadata["csv_sha256"],
        "sklearn_version": sklearn.__version__,
        "split_protocol": "calendar-v1",
        "feature_protocol": "strict-past-v1",
        "seed": SEED,
        "pipeline_sha256": hashlib.sha256(
            b"".join(
                (Path(__file__).parent / name).read_text(encoding="utf-8").encode("utf-8")
                for name in ["data.py", "training.py", "uncertainty.py", "monitoring.py"]
            )
        ).hexdigest(),
    }
    run_id = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:12]
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "model.joblib"
    joblib.dump(
        {
            "model": selected,
            "features": FEATURES,
            "radius": radius,
            "run_id": run_id,
            "reference": reference,
            "sklearn_version": sklearn.__version__,
        },
        artifact_path,
    )
    artifact_hash = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    report = {
        "run_id": run_id,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset": {
            **metadata,
            "eligible_rows": len(frame),
            "excluded_observed_rows": len(source) - len(frame),
        },
        "splits": {
            key: {"n": len(value), "start": str(value.index.min()), "end": str(value.index.max())}
            for key, value in blocks.items()
        },
        "selected_model": selected_name,
        "configuration": configuration,
        "validation": validation_scores,
        "metrics": metrics,
        "intervals": {
            "target": 0.9,
            "coverage": float(covered.mean()),
            "mean_width": float((upper - lower).mean()),
            "radius": radius,
            "n": len(test),
            "calibration_n": len(calibration),
        },
        "importance": sorted(
            [
                {"feature": name, "mae_increase": float(value), "std": float(std)}
                for name, value, std in zip(
                    FEATURES, importance.importances_mean, importance.importances_std, strict=True
                )
            ],
            key=lambda item: item["mae_increase"],
            reverse=True,
        ),
        "slices": slices,
        "series": [
            {
                "timestamp": str(timestamp),
                "actual": float(row.target),
                "prediction": float(predictions[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
                "seasonal": float(row.lag_168h),
            }
            for i, (timestamp, row) in enumerate(test.iterrows())
        ],
        "examples": [
            {
                "timestamp": str(test.index[i]),
                "actual": float(test.target.iloc[i]),
                "prediction": float(predictions[i]),
                "lower": float(lower[i]),
                "upper": float(upper[i]),
                "features": {name: float(test.iloc[i][name]) for name in FEATURES},
            }
            for i in np.linspace(0, len(test) - 1, 12, dtype=int)
        ],
        "monitoring": {
            "reference_window": "calibration",
            "reference_n": len(calibration),
            "observed": drift_report(reference, test[FEATURES]),
            "shifted": drift_report(reference, stressed),
        },
        "benchmark": {
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "n": 100,
            "scope": "warm single-row model prediction, one CPU thread; excludes HTTP",
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "training_seconds": time.perf_counter() - started,
        "artifact_sha256": artifact_hash,
        "caveats": [
            "Rolling one-hour-ahead backtest, assuming previous-hour counts arrive immediately.",
            "Previous-hour weather only; no observed target-hour weather or target components.",
            "Temporal dependence: 90% is a calibration target, not a coverage guarantee.",
            "Missing source hours and incomplete lag histories are excluded, not zero-filled.",
            "Two historical years in one city; no current-system or cross-city validation.",
            "PSI thresholds are heuristics. Seasonal calendar drift can be expected.",
            "Synthetic stress shifts exercise monitors; they are not live production incidents.",
            "Latency is local, warm, single-threaded model time, not an API service-level claim.",
        ],
    }
    write_json(
        output / "manifest.json",
        {
            **configuration,
            "run_id": run_id,
            "artifact_sha256": artifact_hash,
            "artifact_file": "model.joblib",
            "radius": radius,
        },
    )
    write_json(output / "report.json", report)
    write_json(report_path, report)
    test_results = pd.DataFrame(
        {
            "actual": test.target,
            "prediction": predictions,
            "lower": lower,
            "upper": upper,
            "seasonal_24h": test.lag_24h,
            "seasonal_168h": test.lag_168h,
        }
    )
    test_results.to_csv(report_path.parent / "predictions.csv", index_label="timestamp")
    test_results.to_csv(output / "predictions.csv", index_label="timestamp")
    return report
