"""Validated configuration and local runtime locations."""

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed: int = 42
    hours: int = Field(default=9600, ge=720, le=50000)
    start: str = "2024-01-01T00:00:00Z"
    validation_fraction: float = Field(default=0.2, gt=0.05, lt=0.5)
    n_estimators: int = Field(default=80, ge=5, le=300)
    max_depth: int = Field(default=3, ge=1, le=6)
    learning_rate: float = Field(default=0.07, gt=0, le=1)
    lower_quantile: float = Field(default=0.1, gt=0, lt=0.5)
    upper_quantile: float = Field(default=0.9, gt=0.5, lt=1)
    model_name: str = "energy-demand"
    experiment: str = "energy-demand"
    min_monitor_samples: int = Field(default=48, ge=10)
    monitor_window: int = Field(default=336, ge=10)
    psi_threshold: float = Field(default=0.2, gt=0)
    ks_alpha: float = Field(default=0.01, gt=0, lt=1)
    mae_degradation_ratio: float = Field(default=1.25, gt=1)
    min_relative_improvement: float = Field(default=0.02, ge=0, lt=1)

    @model_validator(mode="after")
    def check_window(self):
        if self.monitor_window < self.min_monitor_samples:
            raise ValueError("monitor_window must cover min_monitor_samples")
        return self


def load_settings(path: str | Path = "config/default.yaml") -> Settings:
    with Path(path).open() as stream:
        return Settings(**yaml.safe_load(stream))


def runtime_dir() -> Path:
    path = Path(os.getenv("ENERGY_RUNTIME", "runtime")).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def tracking_uri() -> str:
    return os.getenv("MLFLOW_TRACKING_URI", (runtime_dir() / "mlruns").as_uri())
