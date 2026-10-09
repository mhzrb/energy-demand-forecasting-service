import numpy as np
import pandas as pd
import pytest

from energy.config import Settings
from energy.monitoring import Ledger, drift_report, psi, snapshot
from energy.training import load_champion


def test_drift_stable_shifted_and_constant():
    settings = Settings()
    rng = np.random.default_rng(4)
    ref = pd.DataFrame({"temperature": rng.normal(10, 2, 1000)})
    assert not drift_report(ref, ref.copy(), settings)["drifted"]
    assert drift_report(ref, ref + 15, settings)["drifted"]
    assert psi(np.ones(500), np.ones(500) * 2) > 0.2
    assert drift_report(ref, ref.head(5), settings)["status"] == "insufficient_data"


def test_ledger_actuals_version_isolation_and_performance(trained, tmp_path):
    settings, _, _ = trained
    model, version = load_champion(settings)
    ledger = Ledger(tmp_path / "ledger.sqlite")
    X = model.reference_.tail(60)
    predictions = model.predict(X)
    ids = ledger.log(["2026-01-01T00:00:00Z"] * len(X), version, X, predictions)
    assert (
        snapshot(ledger, model, version, settings)["performance"]["status"] == "insufficient_data"
    )
    for pid, predicted in zip(ids, predictions.prediction, strict=True):
        ledger.actual(pid, float(predicted) + 150)
    performance = snapshot(ledger, model, version, settings)["performance"]
    assert performance["mae"] == pytest.approx(150) and performance["degraded"]
    assert ledger.recent("other-version", 100) == []
    ledger.actual(ids[0], float(predictions.prediction.iloc[0]) + 150)
    with pytest.raises(ValueError):
        ledger.actual(ids[0], 0)
    with pytest.raises(KeyError):
        ledger.actual("missing", 0)
