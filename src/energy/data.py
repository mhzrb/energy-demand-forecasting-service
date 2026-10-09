"""Seeded hourly synthetic data and explicit quality reports. Units: kW, degrees C."""

import numpy as np
import pandas as pd

COLUMNS = {"timestamp", "temperature", "holiday", "demand"}
RANGES = {"temperature": (-50, 60), "demand": (0, 10000), "holiday": (0, 1)}


class DataQualityError(ValueError):
    def __init__(self, report: dict):
        self.report = report
        super().__init__(f"Data validation failed: {report['errors']}")


def generate(hours=9600, seed=42, start="2024-01-01T00:00:00Z", faults=True, drift=False):
    rng = np.random.default_rng(seed)
    ts = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    day = ts.dayofyear.to_numpy()
    hour = ts.hour.to_numpy()
    temperature = (
        11 + 11 * np.sin(2 * np.pi * (day - 100) / 365.25)
        + 3 * np.sin(2 * np.pi * (hour - 8) / 24) + rng.normal(0, 1.4, hours)
    )
    if drift:
        temperature += 11  # Heat-wave stress test, not a climate projection.
    holiday = np.array([(t.month, t.day) in {(1, 1), (5, 1), (12, 25), (12, 26)} for t in ts])
    demand = (
        410 + 75 * np.sin(2 * np.pi * (hour - 7) / 24)
        + 45 * (ts.dayofweek.to_numpy() < 5) - 60 * holiday
        + 9 * np.maximum(16 - temperature, 0) + 14 * np.maximum(temperature - 22, 0)
        + 20 * np.cos(2 * np.pi * day / 365.25) + rng.normal(0, 12, hours)
    )
    if drift:
        demand = 1.22 * demand + 65  # Demand-meter gain/offset change.
    frame = pd.DataFrame({"timestamp": ts, "temperature": temperature,
                          "holiday": holiday.astype(int), "demand": demand})
    if faults:
        for column in ("temperature", "demand"):
            missing = rng.choice(hours, max(1, hours // 200), replace=False)
            broken = rng.choice(hours, max(1, hours // 500), replace=False)
            frame.loc[missing, column] = np.nan
            frame.loc[broken, column] = 99999.0
    return frame


def validate(frame: pd.DataFrame, allow_sensor_faults=False) -> dict:
    errors, warnings = [], []
    report = {"rows": len(frame), "errors": errors, "warnings": warnings, "columns": {}}
    if set(frame.columns) != COLUMNS:
        errors.append(f"Expected columns {sorted(COLUMNS)}; got {sorted(frame.columns)}")
        return report
    if len(frame) < 169:
        errors.append("At least 169 hourly rows are required")
    ts = pd.to_datetime(frame.timestamp, errors="coerce", utc=True)
    if ts.isna().any() or ts.duplicated().any() or not ts.is_monotonic_increasing:
        errors.append("Timestamps must be valid, unique, and increasing")
    if not ts.diff().iloc[1:].eq(pd.Timedelta(hours=1)).all():
        errors.append("Timestamps must have exactly hourly cadence; reindex upstream")
    for column, (low, high) in RANGES.items():
        values = pd.to_numeric(frame[column], errors="coerce")
        invalid_type = frame[column].notna() & values.isna()
        invalid_range = values.notna() & (~np.isfinite(values) | ~values.between(low, high))
        if column == "holiday":
            invalid_range |= values.notna() & ~values.isin([0, 1])
        bad = values.isna() | invalid_range
        report["columns"][column] = {
            "missing": int(values.isna().sum()), "out_of_range": int(invalid_range.sum()),
            "bad_row_indices": np.flatnonzero(bad).tolist()[:20],
        }
        if invalid_type.any():
            errors.append(f"{column}: nonnumeric values")
        if bad.any():
            message = f"{column}: {bad.sum()} missing/invalid observations ({bad.mean():.2%})"
            if allow_sensor_faults and column != "holiday" and bad.mean() <= 0.05:
                warnings.append(message)
            else:
                errors.append(message)
    return report


def clean(frame: pd.DataFrame):
    """Quarantine invalid sensors; forward-fill temperature only, never the training label."""
    report = validate(frame, allow_sensor_faults=True)
    if report["errors"]:
        raise DataQualityError(report)
    out = frame.copy()
    out["timestamp"] = pd.to_datetime(out.timestamp, utc=True)
    for column in ("temperature", "demand"):
        values = pd.to_numeric(out[column], errors="coerce")
        low, high = RANGES[column]
        out[column] = values.where(values.between(low, high))
    out["temperature"] = out.temperature.ffill(limit=3)
    return out, report
