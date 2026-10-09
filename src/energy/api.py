"""FastAPI factory: pinned registry version, durable predictions, bounded monitoring."""

import logging
import os
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, Response
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from energy.config import load_settings, runtime_dir
from energy.features import online_features
from energy.insights import summarize
from energy.monitoring import Ledger, snapshot
from energy.schemas import ActualRequest, BatchRequest, ForecastRequest, ForecastResponse
from energy.training import load_champion

logger = logging.getLogger(__name__)


def create_app(settings=None, model_loader=load_champion, ledger=None):
    settings = settings or load_settings(os.getenv("ENERGY_CONFIG", "config/default.yaml"))
    ledger = ledger or Ledger(runtime_dir() / "predictions.sqlite")
    prom = CollectorRegistry()
    predictions_total = Counter(
        "energy_predictions_total", "Persisted forecasts", ["version"], registry=prom
    )
    errors = Counter(
        "energy_prediction_errors_total", "Failed inference/storage calls", registry=prom
    )
    latency = Histogram("energy_prediction_seconds", "Inference plus storage time", registry=prom)
    drift_gauge = Gauge(
        "energy_input_drift", "Input drift: -1 unknown, 0 absent, 1 detected", registry=prom
    )
    mae_gauge = Gauge(
        "energy_recent_mae_kw", "Recent MAE; -1 if insufficient actuals", registry=prom
    )
    samples = Gauge("energy_actual_samples", "Actuals in the current version window", registry=prom)
    ready = Gauge("energy_model_ready", "Registry model loaded successfully", registry=prom)

    @asynccontextmanager
    async def lifespan(app):
        try:
            app.state.bundle, app.state.version = model_loader(settings)
            ready.set(1)
        except Exception:
            logger.exception("Could not load registry champion")
            app.state.bundle, app.state.version = None, None
            ready.set(0)
        yield

    app = FastAPI(title="Energy Demand Forecasting Service", version="0.1.0", lifespan=lifespan)

    def require_model():
        if app.state.bundle is None:
            raise HTTPException(503, "Model unavailable. Train a champion and restart the API.")
        return app.state.bundle, app.state.version

    def monitor():
        bundle, version = require_model()
        result = snapshot(ledger, bundle, version, settings)
        drift_gauge.set(
            int(result["drift"]["drifted"]) if result["drift"]["status"] == "ok" else -1
        )
        mae_gauge.set(result["performance"].get("mae", -1))
        samples.set(result["performance"]["samples"])
        return result

    def predict_many(items):
        bundle, version = require_model()
        started = time.perf_counter()
        try:
            features = pd.concat(
                [
                    online_features(
                        item.forecast_at, item.temperature, int(item.holiday), item.history
                    )
                    for item in items
                ],
                ignore_index=True,
            )
            predictions = bundle.predict(features)
            ids = ledger.log(
                [i.forecast_at.isoformat() for i in items], version, features, predictions
            )
            predictions_total.labels(version=version).inc(len(items))
            return [
                ForecastResponse(
                    prediction_id=pid,
                    forecast_at=item.forecast_at,
                    model_version=version,
                    prediction_kw=row.prediction,
                    lower_kw=row.lower,
                    upper_kw=row.upper,
                    nominal_coverage=bundle.upper_quantile - bundle.lower_quantile,
                )
                for pid, item, row in zip(ids, items, predictions.itertuples(), strict=True)
            ]
        except Exception as exc:
            errors.inc()
            logger.exception("Prediction failed")
            raise HTTPException(503, "Prediction could not be computed and persisted") from exc
        finally:
            latency.observe(time.perf_counter() - started)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        logger.error("Unhandled service error", exc_info=exc)
        return JSONResponse(status_code=500, content={"detail": "Internal service error"})

    @app.get("/health")
    def health():
        _, version = require_model()
        with ledger.connect() as db:
            db.execute("SELECT 1 FROM predictions LIMIT 1")
        return {"status": "ok", "model_version": version}

    @app.get("/model-version")
    def model_version():
        bundle, version = require_model()
        return {
            "name": settings.model_name,
            "version": version,
            "train_end": bundle.train_end_,
            "refresh_policy": "pinned until process restart",
        }

    @app.post("/predict", response_model=ForecastResponse)
    def predict(item: ForecastRequest):
        return predict_many([item])[0]

    @app.post("/predict/batch", response_model=list[ForecastResponse])
    def batch(payload: BatchRequest):
        return predict_many(payload.items)

    @app.post("/actuals")
    def actual(payload: ActualRequest):
        try:
            ledger.actual(payload.prediction_id, payload.actual_kw)
        except KeyError as exc:
            raise HTTPException(404, "Unknown prediction ID") from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        return {"status": "recorded"}

    @app.get("/monitoring")
    def monitoring():
        return monitor()

    @app.get("/metrics")
    def prometheus():
        if app.state.bundle is not None:
            monitor()
        return Response(
            generate_latest(prom), media_type="text/plain; version=0.0.4; charset=utf-8"
        )

    @app.post("/insights")
    def insights(item: ForecastRequest):
        forecast = predict_many([item])[0]
        return {"forecast": forecast, **summarize(forecast, monitor())}

    return app
