import numpy as np
import pandas as pd
import pytest

from energy.data import DataQualityError, clean, generate, validate
from energy.features import feature_frame, online_features, supervised


def test_generator_reproducible_and_faults_reported():
    a, b = generate(960), generate(960)
    pd.testing.assert_frame_equal(a, b)
    assert validate(a)["errors"]
    fixed, report = clean(a)
    assert report["warnings"] and not report["errors"]
    assert fixed.demand.isna().any()  # Labels were not filled.


@pytest.mark.parametrize("fault", ["schema", "cadence", "duplicate", "range", "holiday", "type"])
def test_validation_rejects_invalid_data(fault):
    frame = generate(720, faults=False)
    if fault == "schema":
        frame = frame.drop(columns="demand")
    elif fault == "cadence":
        frame = frame.drop(index=200)
    elif fault == "duplicate":
        frame.loc[100, "timestamp"] = frame.timestamp.iloc[99]
    elif fault == "range":
        frame.loc[:100, "temperature"] = np.inf
    elif fault == "holiday":
        frame.loc[10, "holiday"] = 2
    else:
        frame["demand"] = frame.demand.astype(object)
        frame.loc[10, "demand"] = "broken"
    with pytest.raises(DataQualityError):
        clean(frame)


def test_no_future_target_leakage_even_through_imputation():
    frame = generate(720, faults=False)
    frame.loc[249:250, "demand"] = np.nan
    clean_a, _ = clean(frame)
    a = feature_frame(clean_a)
    changed = frame.copy()
    changed.loc[251:, "demand"] += 500
    clean_b, _ = clean(changed)
    b = feature_frame(clean_b)
    pd.testing.assert_frame_equal(a.loc[:251], b.loc[:251])
    assert a.loc[252, "lag_1"] != b.loc[252, "lag_1"]
    assert a.loc[251, "lag_1"] == frame.loc[248, "demand"]


def test_temperature_fill_is_bounded_and_never_backfills():
    frame = generate(720, faults=False)
    frame.loc[:1, "temperature"] = np.nan
    frame.loc[200:204, "temperature"] = np.nan
    cleaned, _ = clean(frame)
    assert cleaned.temperature.iloc[:2].isna().all()
    assert cleaned.temperature.iloc[200:203].notna().all()
    assert cleaned.temperature.iloc[203:205].isna().all()


def test_feature_values_and_online_offline_parity():
    frame = generate(720, faults=False)
    offline = feature_frame(frame)
    t = 300
    online = online_features(
        frame.timestamp.iloc[t],
        frame.temperature.iloc[t],
        frame.holiday.iloc[t],
        frame.demand.iloc[t - 168 : t].tolist(),
    )
    pd.testing.assert_frame_equal(offline.iloc[[t]].reset_index(drop=True), online)
    assert offline.lag_24.iloc[t] == frame.demand.iloc[t - 24]
    assert offline.lag_168.iloc[t] == frame.demand.iloc[t - 168]
    assert offline.rolling_mean_24.iloc[t] == pytest.approx(frame.demand.iloc[t - 24 : t].mean())
    X, y, _ = supervised(frame)
    assert X.index.min() == 168 and len(X) == len(y) == 552


def test_duplicate_columns_have_clear_error():
    frame = generate(720, faults=False)
    duplicate = pd.concat([frame, frame[["demand"]]], axis=1)
    assert validate(duplicate)["errors"]
