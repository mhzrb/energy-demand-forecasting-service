import pytest
from fastapi.testclient import TestClient

from energy.api import create_app
from energy.demo import request_example
from energy.monitoring import Ledger


@pytest.fixture
def client(trained, tmp_path):
    settings, _, _ = trained
    ledger = Ledger(tmp_path / "api.sqlite")
    with TestClient(create_app(settings=settings, ledger=ledger)) as client:
        yield client, settings, ledger


def test_api_single_batch_actual_monitor_metrics_insights(client):
    api, settings, ledger = client
    payload = request_example(settings)
    assert api.get("/health").status_code == 200
    assert api.get("/model-version").json()["version"]
    response = api.post("/predict", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["lower_kw"] <= result["prediction_kw"] <= result["upper_kw"]
    assert api.post("/predict/batch", json={"items": [payload, payload]}).status_code == 200
    actual = {"prediction_id": result["prediction_id"], "actual_kw": 500}
    assert api.post("/actuals", json=actual).status_code == 200
    assert api.post("/actuals", json=actual).status_code == 200
    assert api.post("/actuals", json={**actual, "actual_kw": 501}).status_code == 409
    assert api.post("/actuals", json={**actual, "prediction_id": "none"}).status_code == 404
    assert api.get("/monitoring").json()["performance"]["samples"] == 1
    assert 'energy_predictions_total{version="1"} 3.0' in api.get("/metrics").text
    insight = api.post("/insights", json=payload).json()
    assert insight["provider"] == "template" and "synthetic" in insight["summary"]
    assert len(ledger.recent(result["model_version"], 100)) == 4


@pytest.mark.parametrize(
    "change",
    [
        {"history": [1.0]},
        {"temperature": 100},
        {"forecast_at": "2025-01-01T00:00:00"},
        {"forecast_at": "2025-01-01T00:01:00Z"},
        {"history_end": "2025-01-01T00:00:00Z"},
        {"extra": 5},
        {"history": [-1.0] * 168},
    ],
)
def test_request_contract_rejects_bad_values(client, change):
    api, settings, _ = client
    response = api.post("/predict", json={**request_example(settings), **change})
    assert response.status_code == 422


def test_batch_limits(client):
    api, settings, _ = client
    assert api.post("/predict/batch", json={"items": []}).status_code == 422
    assert (
        api.post("/predict/batch", json={"items": [request_example(settings)] * 129}).status_code
        == 422
    )


def test_missing_model_is_explicit_503(tmp_path):
    def fail(_):
        raise FileNotFoundError("No model")

    with TestClient(create_app(model_loader=fail, ledger=Ledger(tmp_path / "empty.sqlite"))) as api:
        assert api.get("/health").status_code == 503
        assert "energy_model_ready 0.0" in api.get("/metrics").text


def test_storage_failure_does_not_return_unlogged_prediction(client, monkeypatch):
    api, settings, ledger = client

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(ledger, "log", fail)
    assert api.post("/predict", json=request_example(settings)).status_code == 503


def test_llm_failure_uses_template(client, monkeypatch):
    api, settings, _ = client
    monkeypatch.setenv("INSIGHT_PROVIDER", "openai")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    response = api.post("/insights", json=request_example(settings)).json()
    assert response["provider"] == "template" and response["fallback_reason"] == "missing_llm_key"
