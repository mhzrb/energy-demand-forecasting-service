"""Reproducible drift, promotion, and untouched future-evaluation demonstration."""

import json
from pathlib import Path

import pandas as pd

from energy.data import clean, generate
from energy.features import supervised
from energy.model import evaluate
from energy.monitoring import Ledger, drift_report, snapshot
from energy.retraining import retrain
from energy.training import load_champion


def run_demo(settings, output: Path, ledger: Ledger):
    original, original_version = load_champion(settings)
    start = pd.Timestamp(original.evaluation_end_) + pd.Timedelta(hours=1)
    # 70 days for adaptation+gate, followed by 14 days untouched by the promotion decision.
    frame = generate(24 * 84, settings.seed + 1, start, faults=False, drift=True)
    adaptation = frame.iloc[: 24 * 70].copy()
    X, y, timestamps = supervised(adaptation)
    trigger_X, trigger_y, trigger_ts = X.iloc[:336], y.iloc[:336], timestamps.iloc[:336]
    ids = ledger.log(
        trigger_ts.astype(str).tolist(), original_version, trigger_X, original.predict(trigger_X)
    )
    for pid, value in zip(ids, trigger_y, strict=True):
        ledger.actual(pid, float(value))
    before = snapshot(ledger, original, original_version, settings)
    decision = retrain(adaptation, settings, ledger, output)
    promoted, promoted_version = load_champion(settings)
    full_X, full_y, full_ts = supervised(frame)
    test_mask = full_ts >= frame.timestamp.iloc[24 * 70]
    test_X, test_y = full_X.loc[test_mask], full_y.loc[test_mask]
    after = drift_report(promoted.reference_, test_X, settings)
    result = {
        "synthetic": True,
        "scenario": "temperature +11 C; meter gain 1.22 and offset +65 kW",
        "trigger_window_start": trigger_ts.iloc[0].isoformat(),
        "trigger_window_end": trigger_ts.iloc[-1].isoformat(),
        "before_monitor": before,
        "decision": decision,
        "after_input_drift": after,
        "future_test_start": full_ts.loc[test_mask].iloc[0].isoformat(),
        "future_test_rows": len(test_y),
        "original_version": original_version,
        "serving_version_after": promoted_version,
        "future_test_original": evaluate(original, test_X, test_y),
        "future_test_serving_after": evaluate(promoted, test_X, test_y),
        "note": (
            "Future test never used for training or promotion. Drift can remain after retraining."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "drift_demo.json").write_text(json.dumps(result, indent=2))
    # Small plot with the actual predictions and intervals from this run.
    forecast = promoted.predict(test_X)
    examples = pd.DataFrame(
        {
            "timestamp": full_ts.loc[test_mask],
            "actual_kw": test_y,
            "original_kw": original.predict(test_X).prediction,
            "serving_kw": forecast.prediction,
            "lower_kw": forecast.lower,
            "upper_kw": forecast.upper,
        }
    )
    examples.to_csv(output / "demo_forecasts.csv", index=False)
    return result


def request_example(settings):
    frame, _ = clean(generate(192, settings.seed, faults=False))
    return {
        "forecast_at": frame.timestamp.iloc[168].isoformat(),
        "history_end": frame.timestamp.iloc[167].isoformat(),
        "temperature": float(frame.temperature.iloc[168]),
        "holiday": bool(frame.holiday.iloc[168]),
        "history": frame.demand.iloc[:168].tolist(),
    }
