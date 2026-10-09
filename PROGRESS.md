# Progress

## Done
- Plan and phase commits, using the confirmed local author identity only.
- Seeded synthetic raw data, schema/range/cadence/missing-value reports, causal shared features.
- Point/quantile gradient boosting, chronological evaluation and both naive baselines.
- Real MLflow file-store runs, artifacts, registered versions and champion alias.
- FastAPI single/batch inference, validation, readiness, version, actuals and deterministic insights.
- Durable SQLite forecast ledger, version-isolated PSI/KS and performance monitoring, Prometheus endpoint.
- Serialized retraining, later holdout promotion/rejection, fresh-window guards and decision audit trail.
- Actual heat-wave/meter-shift demo with saved before/after measurements and untouched future test.
- 36 passing tests; ruff lint and format clean; exact isolated reproducibility rerun passed.
- Live Uvicorn process called over HTTP: 9 checks passed, process then stopped cleanly.
- Docker/Compose, provisioned Grafana and Prometheus, optional LLM adapter, CI configuration.
- README, design decisions, measured evidence, figure, verification report and resume bullets.
- Final local author/committer/trailer/remote audit.

## Next
- No remaining local implementation or verification blockers.
- Optional: user publishes to GitHub, runs the prepared CI and container stack, or adapts to real sensors.

## Known issues / limits
- Docker, Compose runtime, Grafana, MLflow UI and hosted CI are NOT VERIFIED.
- Remote LLM integration, clean-OS install, real data and load behavior are NOT VERIFIED.
- Drift remains detectable after retraining; future interval coverage is 73.81%, below nominal 80%.
- Dependency deprecation warnings remain; their full text is retained in the test log.
- Local single-series/single-process architecture; API restart needed after promotion.
