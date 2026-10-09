"""Causal, one-hour-ahead features shared by offline and online code."""

import numpy as np
import pandas as pd

FEATURES = ["temperature", "holiday", "hour_sin", "hour_cos", "weekday", "year_sin",
            "year_cos", "lag_1", "lag_24", "lag_168", "rolling_mean_24", "rolling_std_24"]


def feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ts = pd.to_datetime(frame.timestamp, utc=True)
    # Filling is causal and bounded. Targets in frame.demand are never filled.
    past = frame.demand.ffill(limit=3).shift(1)
    result = pd.DataFrame(index=frame.index)
    result["temperature"] = frame.temperature
    result["holiday"] = frame.holiday
    result["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    result["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
    result["weekday"] = ts.dt.dayofweek
    result["year_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
    result["year_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)
    result["lag_1"] = past
    result["lag_24"] = past.shift(23)
    result["lag_168"] = past.shift(167)
    result["rolling_mean_24"] = past.rolling(24, min_periods=24).mean()
    result["rolling_std_24"] = past.rolling(24, min_periods=24).std(ddof=0)
    return result[FEATURES]


def supervised(frame: pd.DataFrame):
    features = feature_frame(frame)
    keep = features.notna().all(axis=1) & frame.demand.notna()
    return features.loc[keep], frame.loc[keep, "demand"], frame.loc[keep, "timestamp"]


def online_features(timestamp, temperature, holiday, history) -> pd.DataFrame:
    """History is exactly 168 consecutive hourly actuals, oldest first, ending at t-1."""
    ts = pd.Timestamp(timestamp)
    frame = pd.DataFrame({
        "timestamp": pd.date_range(end=ts, periods=169, freq="h"),
        "temperature": [temperature] * 169, "holiday": [holiday] * 169,
        "demand": [*history, np.nan],
    })
    return feature_frame(frame).tail(1).reset_index(drop=True)
