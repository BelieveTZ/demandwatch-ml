"""Validated source ingestion and strictly historical feature construction."""

import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SOURCE_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
EXPECTED_CSV_SHA256 = "e03de4ee4ef4dc376ac6e04bf829673c6269e8eba5c60fa121640fa2f829504f"
FEATURES = [
    "hour",
    "weekday",
    "month",
    "workingday",
    "holiday",
    "trend_days",
    "lag_1h",
    "lag_24h",
    "lag_168h",
    "mean_24h",
    "mean_168h",
    "temp_previous",
    "humidity_previous",
    "wind_previous",
    "weather_previous",
]


def download(destination: Path) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "DemandWatch/0.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = response.read(10_000_001)
    if len(archive) > 10_000_000:
        raise ValueError("Unexpected archive size")
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        content = bundle.read("hour.csv")
    if hashlib.sha256(content).hexdigest() != EXPECTED_CSV_SHA256:
        raise ValueError("UCI source checksum changed; review provenance before training")
    (destination / "hour.csv").write_bytes(content)
    metadata = {
        "source": SOURCE_URL,
        "citation": "Fanaee-T, H. (2013). Bike Sharing. UCI. DOI:10.24432/C5W894",
        "license": "CC BY 4.0",
        "archive_sha256": hashlib.sha256(archive).hexdigest(),
        "csv_sha256": hashlib.sha256(content).hexdigest(),
    }
    (destination / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def load_source(path: Path) -> tuple[pd.DataFrame, dict]:
    source = pd.read_csv(path)
    required = {
        "dteday",
        "hr",
        "cnt",
        "workingday",
        "holiday",
        "temp",
        "hum",
        "windspeed",
        "weathersit",
    }
    if not required.issubset(source.columns):
        raise ValueError(f"Missing source columns: {sorted(required - set(source.columns))}")
    if source[list(required)].isna().any().any():
        raise ValueError("Source has missing required values")
    if not source.hr.between(0, 23).all() or not (source.hr % 1 == 0).all():
        raise ValueError("Hour must be an integer in 0..23")
    if not source.cnt.ge(0).all() or not np.isfinite(source.cnt).all():
        raise ValueError("Counts must be finite and nonnegative")
    for column in ["temp", "hum", "windspeed"]:
        if not source[column].between(0, 1).all():
            raise ValueError(f"{column} must use UCI's normalized 0..1 scale")
    for column in ["holiday", "workingday"]:
        if not source[column].isin([0, 1]).all():
            raise ValueError(f"Invalid {column}")
    if not source.weathersit.isin([1, 2, 3, 4]).all():
        raise ValueError("Invalid weather category")
    timestamps = pd.to_datetime(source.dteday, format="%Y-%m-%d") + pd.to_timedelta(
        source.hr, unit="h"
    )
    if timestamps.duplicated().any():
        raise ValueError("Duplicate hourly timestamps")
    source.index = pd.DatetimeIndex(timestamps, name="timestamp")
    source = source.sort_index()
    complete = pd.date_range(source.index.min(), source.index.max(), freq="h")
    metadata = {
        "raw_rows": len(source),
        "missing_hours": len(complete) - len(source),
        "start": str(source.index.min()),
        "end": str(source.index.max()),
        "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return source, metadata


def make_features(source: pd.DataFrame) -> pd.DataFrame:
    # Reindex first: row shifts on the original data are not hourly shifts.
    grid = source.reindex(pd.date_range(source.index.min(), source.index.max(), freq="h"))
    frame = pd.DataFrame(index=grid.index)
    frame.index.name = "timestamp"
    frame["hour"] = grid.index.hour
    frame["weekday"] = grid.index.dayofweek
    frame["month"] = grid.index.month
    frame["workingday"] = grid.workingday
    frame["holiday"] = grid.holiday
    frame["trend_days"] = (grid.index - pd.Timestamp("2011-01-01")).total_seconds() / 86400
    for lag in [1, 24, 168]:
        frame[f"lag_{lag}h"] = grid.cnt.shift(lag)
    past = grid.cnt.shift(1)
    frame["mean_24h"] = past.rolling(24, min_periods=12).mean()
    frame["mean_168h"] = past.rolling(168, min_periods=84).mean()
    for name, column in [
        ("temp", "temp"),
        ("humidity", "hum"),
        ("wind", "windspeed"),
        ("weather", "weathersit"),
    ]:
        frame[f"{name}_previous"] = grid[column].shift(1)
    frame["target"] = grid.cnt
    return frame.dropna(subset=[*FEATURES, "target"])


def split_blocks(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    blocks = {
        "train": frame.loc[frame.index < "2012-07-01"],
        "validation": frame.loc[(frame.index >= "2012-07-01") & (frame.index < "2012-09-01")],
        "calibration": frame.loc[(frame.index >= "2012-09-01") & (frame.index < "2012-11-01")],
        "test": frame.loc[frame.index >= "2012-11-01"],
    }
    if any(len(block) < 100 for block in blocks.values()):
        raise ValueError("Each fixed chronological block needs at least 100 eligible hours")
    return blocks
