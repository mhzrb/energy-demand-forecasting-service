import json

import pandas as pd
import pytest

from energy import retraining
from energy.data import generate
from energy.monitoring import Ledger
from energy.retraining import promotion_gate, retrain, should_retrain
from energy.training import load_champion, registry


def test_trigger_logic_and_gate():
    monitor = {
        "drift": {"status": "insufficient_data", "drifted": False},
        "performance": {"status": "ok", "degraded": False},
    }
    assert not should_retrain(monitor)
    monitor["performance"]["degraded"] = True
    assert should_retrain(monitor) == ["degraded_mae"]
    monitor["drift"] = {"status": "ok", "drifted": True}
    assert should_retrain(monitor) == ["input_drift", "degraded_mae"]
    assert promotion_gate(20, 15, 0.02)
    assert not promotion_gate(20, 20, 0.02)
    assert not promotion_gate(20, 19.9, 0.02)
    assert not promotion_gate(0, 0, 0.02)


def test_skipped_retraining_is_audited(trained, tmp_path):
    settings, _, _ = trained
    ledger = Ledger(tmp_path / "skip.sqlite")
    result = retrain(generate(720), settings, ledger, tmp_path)
    assert result["status"] == "skipped"
    with ledger.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] == 1
    assert json.loads((tmp_path / "retraining_decision.json").read_text())["promoted"] is False


@pytest.mark.parametrize("accept", [False, True])
def test_actual_registration_and_promotion_branches(trained, tmp_path, monkeypatch, accept):
    settings, _, _ = trained
    original, version = load_champion(settings)
    ledger = Ledger(tmp_path / "retrain.sqlite")
    # Isolate gate control flow; the demo separately measures an unmocked promotion.
    monkeypatch.setattr(
        retraining,
        "snapshot",
        lambda *args: {
            "drift": {"status": "ok", "drifted": True},
            "performance": {"status": "insufficient_data", "degraded": False},
        },
    )
    monkeypatch.setattr(retraining, "promotion_gate", lambda *args: accept)
    start = pd.Timestamp(original.evaluation_end_) + pd.Timedelta(hours=1)
    frame = generate(960, seed=99, start=start, faults=False, drift=True)
    try:
        result = retrain(frame, settings, ledger, tmp_path)
        assert result["promoted"] is accept
        _, now = load_champion(settings)
        assert now == (result["candidate_version"] if accept else version)
        assert pd.Timestamp(result["validation_start"]) > pd.Timestamp(original.evaluation_end_)
        assert "champion_gate" in result and "candidate_gate" in result
    finally:
        registry().set_registered_model_alias(settings.model_name, "champion", version)


def test_rejects_reusing_champion_evaluation_data(trained, tmp_path, monkeypatch):
    settings, _, _ = trained
    monkeypatch.setattr(
        retraining,
        "snapshot",
        lambda *args: {
            "drift": {"status": "ok", "drifted": True},
            "performance": {"status": "insufficient_data", "degraded": False},
        },
    )
    with pytest.raises(ValueError, match="prior evaluation"):
        retrain(generate(960), settings, Ledger(tmp_path / "x.sqlite"), tmp_path)
