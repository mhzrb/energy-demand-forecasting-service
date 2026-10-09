"""Point and quantile estimators serialized together as one registry artifact."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ForecastBundle(BaseEstimator):
    def __init__(
        self,
        n_estimators=80,
        max_depth=3,
        learning_rate=0.07,
        seed=42,
        lower_quantile=0.1,
        upper_quantile=0.9,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.seed = seed
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y):
        params = dict(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            random_state=self.seed,
            min_samples_leaf=12,
        )
        self.point_ = GradientBoostingRegressor(loss="squared_error", **params).fit(X, y)
        self.lower_ = GradientBoostingRegressor(
            loss="quantile", alpha=self.lower_quantile, **params
        ).fit(X, y)
        self.upper_ = GradientBoostingRegressor(
            loss="quantile", alpha=self.upper_quantile, **params
        ).fit(X, y)
        self.reference_ = X.tail(2000).copy().reset_index(drop=True)
        self.feature_names_in_ = np.array(X.columns)
        return self

    def predict(self, X):
        if list(X.columns) != list(self.feature_names_in_):
            raise ValueError("Feature names/order differ from the training contract")
        point = np.maximum(0, self.point_.predict(X))
        low = np.maximum(0, self.lower_.predict(X))
        high = np.maximum(0, self.upper_.predict(X))
        return pd.DataFrame(
            {
                "prediction": point,
                "lower": np.minimum.reduce([low, high, point]),
                "upper": np.maximum.reduce([low, high, point]),
            },
            index=X.index,
        )


def metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    nonzero = np.abs(actual) > 1e-8
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mape_pct": (
            float(np.mean(np.abs((actual[nonzero] - predicted[nonzero]) / actual[nonzero])) * 100)
            if nonzero.any()
            else None
        ),
        "mape_excluded_zeros": int((~nonzero).sum()),
    }


def evaluate(bundle, X, y):
    predictions = bundle.predict(X)
    result = metrics(y, predictions.prediction)
    result["interval_coverage"] = float(
        ((y >= predictions.lower) & (y <= predictions.upper)).mean()
    )
    result["mean_interval_width"] = float((predictions.upper - predictions.lower).mean())
    return result
