"""Prediction ledger, matched actuals, drift statistics and performance windows."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from energy.model import metrics


class Ledger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id TEXT PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    forecast_at TEXT NOT NULL, version TEXT NOT NULL, features TEXT NOT NULL,
                    prediction REAL NOT NULL, lower REAL NOT NULL, upper REAL NOT NULL,
                    actual REAL, actual_received_at TEXT
                );
                CREATE INDEX IF NOT EXISTS version_time ON predictions(version, created_at);
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY, created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    payload TEXT NOT NULL
                );
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def log(self, timestamps, version, features, predictions):
        ids = [str(uuid4()) for _ in timestamps]
        with self.connect() as db:
            db.executemany(
                "INSERT INTO predictions (id,forecast_at,version,features,prediction,lower,upper) "
                "VALUES (?,?,?,?,?,?,?)",
                [
                    (
                        id_,
                        str(ts),
                        version,
                        json.dumps(feature),
                        float(pred.prediction),
                        float(pred.lower),
                        float(pred.upper),
                    )
                    for id_, ts, feature, pred in zip(
                        ids,
                        timestamps,
                        features.to_dict("records"),
                        predictions.itertuples(),
                        strict=True,
                    )
                ],
            )
        return ids

    def actual(self, prediction_id, value):
        with self.connect() as db:
            row = db.execute(
                "SELECT actual FROM predictions WHERE id=?", (prediction_id,)
            ).fetchone()
            if row is None:
                raise KeyError(prediction_id)
            if row["actual"] is not None and row["actual"] != value:
                raise ValueError("Actual already recorded with a different value")
            db.execute(
                "UPDATE predictions SET actual=?, actual_received_at=CURRENT_TIMESTAMP WHERE id=?",
                (value, prediction_id),
            )

    def recent(self, version, limit):
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM predictions WHERE version=? ORDER BY rowid DESC LIMIT ?",
                (version, limit),
            ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def decision(self, payload):
        with self.connect() as db:
            db.execute("INSERT INTO decisions(payload) VALUES (?)", (json.dumps(payload),))

    def evaluation_watermark(self):
        """Never reuse a promotion holdout, including one that rejected a candidate."""
        with self.connect() as db:
            decisions = db.execute("SELECT payload FROM decisions").fetchall()
        ends = [json.loads(row["payload"]).get("validation_end") for row in decisions]
        return max((pd.Timestamp(end) for end in ends if end), default=None)


def psi(reference, current, bins=10):
    reference, current = np.asarray(reference, dtype=float), np.asarray(current, dtype=float)
    # A constant reference needs explicit narrow boundaries to detect a new level.
    quantiles = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(quantiles) == 1:
        delta = max(abs(quantiles[0]) * 1e-6, 1e-6)
        edges = np.array([-np.inf, quantiles[0] - delta, quantiles[0] + delta, np.inf])
    else:
        edges = np.r_[-np.inf, quantiles, np.inf]
    expected = np.histogram(reference, edges)[0].astype(float) + 0.5
    observed = np.histogram(current, edges)[0].astype(float) + 0.5
    expected /= expected.sum()
    observed /= observed.sum()
    return float(np.sum((observed - expected) * np.log(observed / expected)))


def drift_report(reference, current, settings):
    if len(current) < settings.min_monitor_samples:
        return {
            "status": "insufficient_data",
            "drifted": False,
            "samples": len(current),
            "features": {},
        }
    features = {}
    alpha = settings.ks_alpha / len(reference.columns)  # Bonferroni across input features.
    for column in reference:
        ref, cur = reference[column].to_numpy(), current[column].to_numpy()
        statistic, pvalue = ks_2samp(ref, cur)
        score = psi(ref, cur)
        features[column] = {
            "psi": score,
            "ks_statistic": float(statistic),
            "ks_pvalue": float(pvalue),
            "drifted": bool(score > settings.psi_threshold and pvalue < alpha),
        }
    return {
        "status": "ok",
        "drifted": any(f["drifted"] for f in features.values()),
        "samples": len(current),
        "ks_adjusted_alpha": alpha,
        "features": features,
    }


def snapshot(ledger, bundle, version, settings):
    rows = ledger.recent(version, settings.monitor_window)
    current = pd.DataFrame([json.loads(row["features"]) for row in rows])
    drift = drift_report(bundle.reference_, current, settings)
    matched = [row for row in rows if row["actual"] is not None]
    performance = {"status": "insufficient_data", "samples": len(matched), "degraded": False}
    if len(matched) >= settings.min_monitor_samples:
        score = metrics([r["actual"] for r in matched], [r["prediction"] for r in matched])
        performance.update(
            status="ok",
            **score,
            reference_mae=bundle.validation_mae_,
            degraded=score["mae"] > bundle.validation_mae_ * settings.mae_degradation_ratio,
        )
    return {"model_version": version, "drift": drift, "performance": performance}
