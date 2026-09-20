from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from demandwatch.api import load_artifact
from demandwatch.data import FEATURES, load_source, make_features, split_blocks
from demandwatch.training import train

SOURCE = Path("data/raw/hour.csv")


@pytest.mark.skipif(not SOURCE.exists(), reason="Run download to enable real-data integration")
def test_training_artifact_roundtrip_and_test_metrics(tmp_path):
    report = train(SOURCE, tmp_path / "model", tmp_path / "published" / "report.json")
    artifact = load_artifact(tmp_path / "model")
    source, _ = load_source(SOURCE)
    test = split_blocks(make_features(source))["test"]
    prediction = np.maximum(0, artifact["model"].predict(test[FEATURES]))
    np.testing.assert_allclose(prediction, [row["prediction"] for row in report["series"]])
    measured = float(np.abs(test.target - prediction).mean())
    chosen = next(row for row in report["metrics"] if row["model"] == report["selected_model"])
    assert measured == pytest.approx(chosen["mae"])
    assert report["dataset"]["raw_rows"] == 17379
    assert chosen["n"] == report["splits"]["test"]["n"]
    assert report["splits"]["validation"]["end"] < report["splits"]["calibration"]["start"]

    changed = pd.read_csv(SOURCE)
    changed.loc[changed.dteday >= "2012-11-01", "cnt"] *= 3
    changed_source = tmp_path / "changed.csv"
    changed.to_csv(changed_source, index=False)
    changed_report = train(changed_source, tmp_path / "changed", tmp_path / "changed-report.json")
    changed_artifact = load_artifact(tmp_path / "changed")
    assert changed_report["validation"] == report["validation"]
    assert changed_report["intervals"]["radius"] == report["intervals"]["radius"]
    np.testing.assert_allclose(changed_artifact["model"].predict(test[FEATURES]), prediction)
