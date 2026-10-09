"""Public one-hour forecast contract."""

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Demand = Annotated[float, Field(ge=0, le=10000, allow_inf_nan=False)]


class ForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forecast_at: datetime
    history_end: datetime
    temperature: float = Field(ge=-50, le=60, allow_inf_nan=False)
    holiday: bool
    history: list[Demand] = Field(min_length=168, max_length=168)

    @field_validator("forecast_at", "history_end")
    @classmethod
    def utc_hour(cls, value):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Timestamp must include a timezone")
        value = value.astimezone(UTC)
        if value.minute or value.second or value.microsecond:
            raise ValueError("Timestamp must be exactly on a UTC hour")
        return value

    @model_validator(mode="after")
    def history_alignment(self):
        if (self.forecast_at - self.history_end).total_seconds() != 3600:
            raise ValueError("history_end must be exactly one hour before forecast_at")
        return self


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ForecastRequest] = Field(min_length=1, max_length=128)


class ForecastResponse(BaseModel):
    prediction_id: str
    forecast_at: datetime
    model_version: str
    prediction_kw: float
    lower_kw: float
    upper_kw: float
    nominal_coverage: float


class ActualRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prediction_id: str = Field(min_length=1, max_length=64)
    actual_kw: Demand
