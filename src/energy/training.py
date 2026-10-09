"""Chronological training, measured baselines, and MLflow lifecycle operations."""

import hashlib
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from mlflow.models import infer_signature

from energy.config import tracking_uri
from energy.data import clean
from energy.features import supervised
from energy.model import ForecastBundle, evaluate, metrics


def registry():
    uri = tracking_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


def load_champion(settings):
    client = registry()
    version = client.get_model_version_by_alias(settings.model_name, "champion")
    # Resolve alias once; a concurrent promotion cannot change the version being loaded.
    model = mlflow.sklearn.load_model(f"models:/{settings.model_name}/{version.version}")
    return model, str(version.version)


def champion_exists(settings):
    try:
        registry().get_model_version_by_alias(settings.model_name, "champion")
        return True
    except MlflowException as exc:
        if exc.error_code == "RESOURCE_DOES_NOT_EXIST":
            return False
        raise


def prepare(frame, settings):
    cleaned, report = clean(frame)
    X, y, timestamps = supervised(cleaned)
    split = int(len(X) * (1 - settings.validation_fraction))
    if split < 168 or len(X) - split < 48:
        raise ValueError("Insufficient complete rows for chronological train/validation split")
    return X, y, timestamps, split, report


def fit_candidate(frame, settings):
    X, y, timestamps, split, quality = prepare(frame, settings)
    bundle = ForecastBundle(
        **{
            key: getattr(settings, key)
            for key in (
                "n_estimators",
                "max_depth",
                "learning_rate",
                "seed",
                "lower_quantile",
                "upper_quantile",
            )
        }
    )
    bundle.fit(X.iloc[:split], y.iloc[:split])
    bundle.train_end_ = timestamps.iloc[split - 1].isoformat()
    bundle.evaluation_end_ = timestamps.iloc[-1].isoformat()
    scores = {
        "gradient_boosting": evaluate(bundle, X.iloc[split:], y.iloc[split:]),
        "naive": metrics(y.iloc[split:], X.iloc[split:].lag_1),
        "seasonal_naive": metrics(y.iloc[split:], X.iloc[split:].lag_168),
    }
    bundle.validation_mae_ = scores["gradient_boosting"]["mae"]
    report = {
        "synthetic": True,
        "forecast_horizon_hours": 1,
        "train_rows": split,
        "validation_rows": len(X) - split,
        "dropped_rows": len(frame) - len(X),
        "train_start": timestamps.iloc[0].isoformat(),
        "train_end": bundle.train_end_,
        "validation_start": timestamps.iloc[split].isoformat(),
        "validation_end": bundle.evaluation_end_,
        "data_sha256": hashlib.sha256(frame.to_csv(index=False).encode()).hexdigest(),
        "metrics": scores,
        "data_quality": quality,
        "feature_importance": dict(
            zip(X.columns, bundle.point_.feature_importances_.tolist(), strict=True)
        ),
        "nominal_interval_coverage": settings.upper_quantile - settings.lower_quantile,
    }
    return bundle, report, X.iloc[split:], y.iloc[split:]


def register(bundle, report, settings, run_name):
    client = registry()
    mlflow.set_experiment(settings.experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_params(settings.model_dump())
        mlflow.set_tags(
            {
                "data_kind": "synthetic",
                "data_sha256": report["data_sha256"],
                "train_end": report["train_end"],
            }
        )
        for name, values in report["metrics"].items():
            mlflow.log_metrics({f"{name}_{k}": v for k, v in values.items() if v is not None})
        mlflow.log_dict(report, "evaluation.json")
        example = bundle.reference_.head(2)
        mlflow.sklearn.log_model(
            bundle,
            artifact_path="model",
            signature=infer_signature(example, bundle.predict(example)),
            input_example=example,
            pip_requirements=["scikit-learn==1.8.0", "pandas==2.2.3", "numpy==2.3.5"],
        )
        try:
            client.get_registered_model(settings.model_name)
        except MlflowException as exc:
            if exc.error_code != "RESOURCE_DOES_NOT_EXIST":
                raise
            client.create_registered_model(settings.model_name)
        version = client.create_model_version(
            settings.model_name, f"runs:/{run.info.run_id}/model", run_id=run.info.run_id
        )
        client.set_model_version_tag(
            settings.model_name, version.version, "evaluation_end", report["validation_end"]
        )
        return str(version.version), run.info.run_id


def train(frame, settings, output: Path):
    if champion_exists(settings):
        raise ValueError("Champion already exists; use retrain so promotion is evaluated")
    bundle, report, _, _ = fit_candidate(frame, settings)
    version, run_id = register(bundle, report, settings, "initial-training")
    registry().set_registered_model_alias(settings.model_name, "champion", version)
    report.update(version=version, run_id=run_id)
    output.mkdir(parents=True, exist_ok=True)
    (output / "training_results.json").write_text(json.dumps(report, indent=2))
    return report
