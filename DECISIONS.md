# Design decisions

- **One-hour forecast horizon:** preceding 168 hourly actuals are available at prediction time. Chronological evaluation is walk-forward with observed actuals, not recursive multi-step forecasting.
- **UTC:** reject non-hour timestamps in serving; avoid implicit daylight-saving shifts. Production ingestion must normalize local meter timestamps explicitly.
- **Causal quality handling:** quarantine out-of-range demand, never impute labels. Bounded forward-fill (three hours) of historical demand/temperature uses past values only. Drop rows with unresolved inputs. Reject more than 5% invalid sensor values and any invalid calendar values.
- **Lightweight stack:** CPU gradient boosting, file-based MLflow tracking/registry, SQLite prediction ledger, no external datasets or paid dependencies.
- **Quantiles:** 10th/90th percentile models produce a nominal 80% interval. Sort endpoints and include the point estimate to resolve crossing. Report actual coverage; this is not a calibrated guarantee.
- **Weather assumption:** synthetic temperature is known at forecast time. Real deployment must use vintage-correct weather forecasts, not realized future weather.
