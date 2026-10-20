# Design decisions

- **One-hour forecast horizon:** preceding 168 hourly actuals are available at prediction time. Chronological evaluation is walk-forward with observed actuals, not recursive multi-step forecasting.
- **UTC:** reject non-hour timestamps in serving; avoid implicit daylight-saving shifts. Production ingestion must normalize local meter timestamps explicitly.
- **Causal quality handling:** quarantine out-of-range demand, never impute labels. Bounded forward-fill (three hours) of historical demand/temperature uses past values only. Drop rows with unresolved inputs. Reject more than 5% invalid sensor values and any invalid calendar values.
- **Lightweight stack:** CPU gradient boosting, file-based MLflow tracking/registry, SQLite prediction ledger, no external datasets or paid dependencies.
- **Quantiles:** 10th/90th percentile models produce a nominal 80% interval. Sort endpoints and include the point estimate to resolve crossing. Report actual coverage; this is not a calibrated guarantee.
- **Weather assumption:** synthetic temperature is known at forecast time. Real deployment must use vintage-correct weather forecasts, not realized future weather.
- **Registry:** MLflow 2.22 skinny SDK keeps local installs small and supports file-store model versions and aliases. The bundle contains the point estimator, both quantile estimators, reference features and evaluation metadata. Only load trusted artifacts.
- **Promotion:** more than 2% lower MAE on an identical, later chronological holdout; initial training refuses to replace an existing champion. Linux/macOS process lock serializes retraining. API pins a resolved version until restart, preventing mixed-version requests.
- **Monitoring:** bounded last-336-prediction window per model version, minimum 48 observations; drift requires both PSI > 0.2 and Bonferroni-corrected KS p < 0.01 / feature count. Input drift is not proof of forecast degradation. Actuals are immutable except for idempotent duplicate submissions.
- **Demo:** 70 drifted days for adaptation and promotion, then 14 untouched future days for before/after assessment. No claim that synthetic performance transfers to real meters.
- **LLM:** opt-in only; deterministic fallback; send aggregate summary only. Output is marked unverified generated text. No remote LLM calls are required by any default command.
- **Holdout reuse:** the decision ledger supplies an evaluation watermark even for rejected candidates. Future retraining must use a new time window. Monitoring observations must precede the promotion holdout, preventing trigger selection from looking at that holdout.
- **Reproducibility:** locked Python 3.12 dependencies and a fresh isolated repeat compare exact data hashes and metrics. Runtime artifacts are ignored by Git; measured small JSON/CSV/log/PNG evidence is committed.
- **Deployment boundary:** Docker could not be used locally. The image build and Compose config validation are confirmed by GitHub Actions; running the full stack and rendering dashboards remain NOT VERIFIED. Defaults bind host ports to loopback and use no remote LLM.
