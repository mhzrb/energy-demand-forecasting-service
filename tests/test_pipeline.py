import numpy as np
import pandas as pd
import pytest

from energy.data import generate
from energy.model import metrics
from energy.training import load_champion, prepare, registry, train


def test_real_mlflow_registry_roundtrip_and_time_split(trained):
    settings, root, report = trained
    model, version = load_champion(settings)
    assert version == report["version"]
    X, y, timestamps, split, _ = prepare(generate(settings.hours, faults=False), settings)
    assert timestamps.iloc[:split].max() < timestamps.iloc[split:].min()
    assert pd.Timestamp(model.train_end_) < pd.Timestamp(report["validation_start"])
    pred = model.predict(X.iloc[split:])
    assert np.isfinite(pred.to_numpy()).all()
    assert (pred.lower <= pred.prediction).all() and (pred.prediction <= pred.upper).all()
    assert set(report["metrics"]) == {"gradient_boosting", "naive", "seasonal_naive"}
    run = registry().get_run(report["run_id"])
    assert (
        run.data.metrics["gradient_boosting_mae"] == report["metrics"]["gradient_boosting"]["mae"]
    )
    assert (root / "output/training_results.json").exists()


def test_initial_training_cannot_overwrite_champion(trained):
    settings, root, _ = trained
    with pytest.raises(ValueError, match="Champion already exists"):
        train(generate(720), settings, root)


def test_metric_zero_handling():
    result = metrics([0, 10], [2, 12])
    assert result["mae"] == 2 and result["rmse"] == 2
    assert result["mape_pct"] == 20 and result["mape_excluded_zeros"] == 1
    assert metrics([0], [1])["mape_pct"] is None
