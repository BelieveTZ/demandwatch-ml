"""Residual calibration; temporal coverage is empirical, not guaranteed."""

import math

import numpy as np


def calibrate_radius(y_true, predictions, alpha: float = 0.1) -> float:
    truth, prediction = np.asarray(y_true), np.asarray(predictions)
    if truth.ndim != 1 or truth.shape != prediction.shape or not len(truth):
        raise ValueError("Calibration arrays must be nonempty, one-dimensional and aligned")
    if not 0 < alpha < 1 or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("Calibration requires finite data and alpha in (0,1)")
    rank = math.ceil((len(truth) + 1) * (1 - alpha))
    if rank > len(truth):
        raise ValueError("Insufficient calibration observations for a finite interval")
    residuals = np.abs(truth - prediction)
    return float(np.partition(residuals, rank - 1)[rank - 1])


def interval(predictions, radius: float):
    if radius < 0 or not np.isfinite(radius):
        raise ValueError("Radius must be finite and nonnegative")
    prediction = np.maximum(0, np.asarray(predictions))
    return np.maximum(0, prediction - radius), prediction + radius
