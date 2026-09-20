import hashlib
import json

import joblib
import pandas as pd
import pytest
import sklearn
from fastapi.testclient import TestClient
from sklearn.dummy import DummyRegressor

from demandwatch.api import create_app, load_artifact
from demandwatch.data import FEATURES
from demandwatch.monitoring import make_reference


@pytest.fixture
def example():
    return {
        "hour": 8,
        "weekday": 1,
        "month": 11,
        "workingday": 1,
        "holiday": 0,
        "trend_days": 675,
        "lag_1h": 50,
        "lag_24h": 100,
        "lag_168h": 120,
        "mean_24h": 90,
        "mean_168h": 80,
        "temp_previous": 0.5,
        "humidity_previous": 0.6,
        "wind_previous": 0.2,
        "weather_previous": 1,
    }


@pytest.fixture
def artifact_dir(tmp_path, example):
    frame = pd.DataFrame([example], columns=FEATURES)
    model = DummyRegressor(strategy="constant", constant=100).fit(frame, [100])
    artifact = {
        "model": model,
        "features": FEATURES,
        "radius": 20,
        "run_id": "test-v1",
        "reference": make_reference(frame),
    }
    path = tmp_path / "model.joblib"
    joblib.dump(artifact, path)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "sklearn_version": sklearn.__version__,
                "run_id": "test-v1",
            }
        )
    )
    return tmp_path


def test_prediction_feedback_and_monitoring(artifact_dir, example):
    client = TestClient(create_app(artifact_dir))
    assert client.get("/api/health").json()["ready"]
    prediction = client.post("/api/predict", json={"features": example})
    assert prediction.status_code == 200
    result = prediction.json()
    assert result["prediction"] == 100
    assert result["lower"] == 80
    assert result["upper"] == 120
    assert client.get("/api/monitoring").json()["performance"] is None
    label = {"prediction_id": result["prediction_id"], "actual": 110}
    assert client.post("/api/feedback", json=label).status_code == 200
    assert client.post("/api/feedback", json=label).status_code == 200
    assert client.post("/api/feedback", json={**label, "actual": 999}).status_code == 409
    monitored = client.get("/api/monitoring").json()
    assert monitored["labeled_n"] == 1
    assert monitored["performance"]["mae"] == 10
    assert monitored["performance"]["coverage"] == 1
    assert monitored["drift"]["status"] == "insufficient_data"
    assert (
        client.post("/api/feedback", json={**label, "prediction_id": "absent"}).status_code == 404
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hour", 24),
        ("hour", 1.2),
        ("lag_1h", -1),
        ("temp_previous", 2),
        ("casual", 20),
        ("humidity_previous", "NaN"),
        ("month", 0),
    ],
)
def test_bad_features_rejected(artifact_dir, example, field, value):
    client = TestClient(create_app(artifact_dir))
    example[field] = value
    assert client.post("/api/predict", json={"features": example}).status_code == 422


def test_missing_feature_rejected(artifact_dir, example):
    del example["lag_24h"]
    client = TestClient(create_app(artifact_dir))
    assert client.post("/api/predict", json={"features": example}).status_code == 422


def test_saved_mode_does_not_fake_predictions(tmp_path, example):
    client = TestClient(create_app(tmp_path))
    assert client.get("/api/health").json()["ready"] is False
    assert client.post("/api/predict", json={"features": example}).status_code == 503


def test_corrupt_artifact_fails_closed(artifact_dir):
    with (artifact_dir / "model.joblib").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(artifact_dir)


def test_report_and_model_stay_on_same_snapshot(artifact_dir):
    path = artifact_dir / "report.json"
    path.write_text(json.dumps({"run_id": "test-v1"}))
    client = TestClient(create_app(artifact_dir))
    path.write_text(json.dumps({"run_id": "test-v2"}))
    assert client.get("/data/report.json").json()["run_id"] == "test-v1"
    with pytest.raises(ValueError, match="versions differ"):
        create_app(artifact_dir)
