"""Fixed-reference drift diagnostics and delayed-label performance."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error


def regression_metrics(truth, prediction) -> dict:
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    if not len(truth):
        raise ValueError("Cannot evaluate an empty batch")
    return {
        "n": len(truth),
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(root_mean_squared_error(truth, prediction)),
    }


def make_reference(frame: pd.DataFrame) -> dict:
    result = {}
    for column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        margin = max(float(np.ptp(values)) * 1e-6, 1e-6)
        # Separate unseen extremes, including changes to a constant reference.
        edges = np.unique(
            np.r_[-np.inf, values.min() - margin, quantiles[1:-1], values.max() + margin, np.inf]
        )
        counts, _ = np.histogram(values, bins=edges)
        proportions = (counts + 0.5) / (len(values) + 0.5 * len(counts))
        result[column] = {
            "edges": [None if not np.isfinite(x) else float(x) for x in edges],
            "proportions": proportions.tolist(),
            "n": len(values),
        }
    return result


def drift_report(reference: dict, current: pd.DataFrame, threshold: float = 0.2) -> dict:
    if current.empty:
        return {"n": 0, "status": "insufficient_data", "features": []}
    result = []
    for column, definition in reference.items():
        values = current[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Drift input must be finite")
        edges = np.asarray([-np.inf, *definition["edges"][1:-1], np.inf])
        counts, _ = np.histogram(values, bins=edges)
        observed = (counts + 0.5) / (len(values) + 0.5 * len(counts))
        expected = np.asarray(definition["proportions"])
        psi = float(np.sum((observed - expected) * np.log(observed / expected)))
        result.append({"feature": column, "psi": psi, "alert": psi >= threshold})
    result.sort(key=lambda item: item["psi"], reverse=True)
    return {
        "n": len(current),
        "threshold": threshold,
        "status": "insufficient_data"
        if len(current) < 100
        else ("review" if any(item["alert"] for item in result) else "stable"),
        "features": result,
        "note": "PSI is a heuristic distribution check, not a statistical test or proof of error.",
    }
