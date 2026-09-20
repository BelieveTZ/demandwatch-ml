"""Loopback-only dashboard, validated inference, and delayed-label monitoring."""

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Annotated
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field
from threadpoolctl import threadpool_limits

from demandwatch.data import FEATURES
from demandwatch.monitoring import drift_report, regression_metrics
from demandwatch.store import PredictionStore
from demandwatch.uncertainty import interval

Count = Annotated[float, Field(ge=0, le=10000, allow_inf_nan=False)]
Normalized = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class FeatureInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    month: int = Field(ge=1, le=12)
    workingday: int = Field(ge=0, le=1)
    holiday: int = Field(ge=0, le=1)
    trend_days: float = Field(ge=0, le=36525)
    lag_1h: Count
    lag_24h: Count
    lag_168h: Count
    mean_24h: Count
    mean_168h: Count
    temp_previous: Normalized
    humidity_previous: Normalized
    wind_previous: Normalized
    weather_previous: int = Field(ge=1, le=4)


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: FeatureInput


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str = Field(min_length=1, max_length=64)
    actual: Count


def load_artifact(directory: Path):
    path = directory / "model.joblib"
    if not path.exists():
        return None
    manifest = json.loads((directory / "manifest.json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["artifact_sha256"]:
        raise ValueError("Model checksum mismatch; retrain from trusted source")
    if manifest["sklearn_version"] != sklearn.__version__:
        raise ValueError("Model/library version mismatch; install lockfile or retrain")
    # Only load artifacts produced locally. Pickle checksums are not signatures.
    artifact = joblib.load(path)
    if artifact["features"] != FEATURES or artifact["run_id"] != manifest["run_id"]:
        raise ValueError("Model schema/version does not match manifest")
    return artifact


def create_app(artifact_directory: Path = Path("artifacts/current")) -> FastAPI:
    app = FastAPI(title="DemandWatch ML", version="0.1.0")
    artifact = load_artifact(artifact_directory)
    store = PredictionStore(artifact_directory / "predictions.sqlite") if artifact else None
    inference_lock = threading.Lock()
    web = Path(__file__).parent / "web"
    evidence_directory = artifact_directory if artifact else web / "data"
    report_path = evidence_directory / "report.json"
    saved_report = json.loads(report_path.read_text()) if report_path.exists() else None
    if artifact and saved_report and saved_report["run_id"] != artifact["run_id"]:
        raise ValueError("Model and report versions differ; finish training before serving")
    csv_path = evidence_directory / "predictions.csv"
    saved_csv = csv_path.read_bytes() if csv_path.exists() else None

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "ready": artifact is not None,
            "mode": "local_inference" if artifact else "saved_replay",
            "model_version": artifact["run_id"] if artifact else None,
        }

    @app.post("/api/predict")
    def predict(request: PredictionRequest):
        if artifact is None:
            raise HTTPException(503, "No local model. Run: python -m demandwatch.cli train")
        features = request.features.model_dump()
        frame = pd.DataFrame([features], columns=FEATURES)
        started = time.perf_counter()
        with inference_lock, threadpool_limits(limits=1):
            value = float(max(0, artifact["model"].predict(frame)[0]))
        lower, upper = interval([value], artifact["radius"])
        result = {
            "prediction_id": str(uuid4()),
            "prediction": value,
            "lower": float(lower[0]),
            "upper": float(upper[0]),
            "model_version": artifact["run_id"],
            "latency_ms": (time.perf_counter() - started) * 1000,
            "interval_target": 0.9,
            "note": "Historical one-hour model; empirical interval, no coverage guarantee.",
        }
        store.add(result["prediction_id"], features, result)
        return result

    @app.post("/api/feedback")
    def feedback(request: FeedbackRequest):
        if store is None:
            raise HTTPException(503, "Local model unavailable")
        try:
            store.label(request.prediction_id, request.actual)
        except KeyError as error:
            raise HTTPException(404, str(error)) from error
        except ValueError as error:
            raise HTTPException(409, str(error)) from error
        return {"status": "recorded", "prediction_id": request.prediction_id}

    @app.get("/api/monitoring")
    def monitoring():
        if artifact is None:
            raise HTTPException(503, "Local model unavailable")
        rows = store.recent(artifact["run_id"])
        current = pd.DataFrame([row["features"] for row in rows], columns=FEATURES)
        labeled = [row for row in rows if row["actual"] is not None]
        performance = None
        if labeled:
            truth = np.array([row["actual"] for row in labeled])
            values = np.array([row["prediction"] for row in labeled])
            performance = {
                **regression_metrics(truth, values),
                "coverage": float(
                    np.mean([row["lower"] <= row["actual"] <= row["upper"] for row in labeled])
                ),
            }
        return {
            "model_version": artifact["run_id"],
            "window": "latest 1000 predictions",
            "drift": drift_report(artifact["reference"], current),
            "performance": performance,
            "labeled_n": len(labeled),
            "note": "Locally submitted traffic/labels, possibly hypothetical; not production.",
        }

    @app.get("/data/report.json", include_in_schema=False)
    def report():
        if saved_report is None:
            raise HTTPException(404, "Saved report missing. Train the model first.")
        return JSONResponse(saved_report)

    @app.get("/data/predictions.csv", include_in_schema=False)
    def predictions_csv():
        if saved_csv is None:
            raise HTTPException(404, "Prediction export missing. Train the model first.")
        return Response(saved_csv, media_type="text/csv")

    app.mount("/", StaticFiles(directory=web, html=True), name="dashboard")
    return app
