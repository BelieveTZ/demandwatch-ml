"""Small local inference ledger with explicit, immutable feedback."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


class PredictionStore:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, created_at TEXT NOT NULL, model_version TEXT NOT NULL,
                    features TEXT NOT NULL, prediction REAL NOT NULL,
                    lower REAL NOT NULL, upper REAL NOT NULL, actual REAL
                )
            """)

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def add(self, identifier: str, features: dict, result: dict):
        with self.connect() as connection:
            connection.execute(
                "INSERT INTO predictions VALUES (?, ?, ?, ?, ?, ?, ?, NULL)",
                (
                    identifier,
                    datetime.now(UTC).isoformat(),
                    result["model_version"],
                    json.dumps(features),
                    result["prediction"],
                    result["lower"],
                    result["upper"],
                ),
            )

    def label(self, identifier: str, actual: float):
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT actual FROM predictions WHERE id = ?", (identifier,)
            ).fetchone()
            if row is None:
                raise KeyError("Unknown prediction ID")
            if row["actual"] is not None and row["actual"] != actual:
                raise ValueError("Prediction already has a different ground-truth label")
            connection.execute(
                "UPDATE predictions SET actual = ? WHERE id = ?", (actual, identifier)
            )

    def recent(self, model_version: str, limit: int = 1000) -> list[dict]:
        with self.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM predictions WHERE model_version = ? ORDER BY rowid DESC LIMIT ?",
                (model_version, limit),
            ).fetchall()
        return [{**dict(row), "features": json.loads(row["features"])} for row in rows]
