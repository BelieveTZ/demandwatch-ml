import numpy as np
import pandas as pd
import pytest

from demandwatch.monitoring import drift_report, make_reference
from demandwatch.uncertainty import calibrate_radius, interval


def test_finite_sample_order_statistic():
    assert calibrate_radius(np.arange(1, 11), np.zeros(10), alpha=0.2) == 9
    with pytest.raises(ValueError, match="Insufficient"):
        calibrate_radius([1, 2], [0, 0], alpha=0.1)


@pytest.mark.parametrize(
    "truth,prediction,alpha",
    [
        ([], [], 0.1),
        ([1], [1, 2], 0.1),
        ([np.nan], [0], 0.1),
        ([1], [0], 0),
    ],
)
def test_invalid_calibration_rejected(truth, prediction, alpha):
    with pytest.raises(ValueError):
        calibrate_radius(truth, prediction, alpha)


def test_nonnegative_interval():
    low, high = interval([5, 100], 20)
    np.testing.assert_array_equal(low, [0, 80])
    np.testing.assert_array_equal(high, [25, 120])


def test_drift_identical_and_strong_shift():
    frame = pd.DataFrame({"feature": np.linspace(0, 1, 1000)})
    reference = make_reference(frame)
    stable = drift_report(reference, frame)
    shifted = drift_report(reference, frame + 10)
    assert stable["status"] == "stable"
    assert stable["features"][0]["psi"] == pytest.approx(0)
    assert shifted["status"] == "review"
    assert shifted["features"][0]["psi"] > 0.2


def test_small_and_empty_samples_not_claimed_stable():
    frame = pd.DataFrame({"feature": np.arange(100)})
    reference = make_reference(frame)
    assert drift_report(reference, frame.iloc[:5])["status"] == "insufficient_data"
    assert drift_report(reference, frame.iloc[:0])["status"] == "insufficient_data"


def test_constant_reference_is_finite():
    frame = pd.DataFrame({"feature": [1] * 100})
    report = drift_report(make_reference(frame), frame)
    assert np.isfinite(report["features"][0]["psi"])
    assert drift_report(make_reference(frame), frame + 1)["status"] == "review"
