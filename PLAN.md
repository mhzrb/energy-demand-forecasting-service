# Plan

1. Establish a lightweight Python package, seeded hourly data, strict validation, causal features, and tests.
2. Train gradient boosting point/quantile models with chronological evaluation; track runs and version/promote models with local MLflow.
3. Serve validated single/batch forecasts and stakeholder summaries; persist predictions/actuals and expose drift, performance, and Prometheus metrics.
4. Implement guarded retraining and a reproducible heat-wave/sensor-shift demo with saved evidence.
5. Add Docker/Compose, Grafana/Prometheus provisioning, CI, and comprehensive tests; exercise a live API.
6. Record measured results and verification limits, finish phase commits, and audit authorship. No remotes or pushes.

Resource budget: small synthetic data only, CPU training, no heavyweight model downloads. Explicitly label unexecuted integrations NOT VERIFIED.

Status: all six phases completed; execution evidence and explicit verification limits are in docs/VERIFICATION.md.
