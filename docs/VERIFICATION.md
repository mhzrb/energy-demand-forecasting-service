# Verification report

Executed in the provided current folder on Linux with **Python 3.12.14**. No Git remote was configured and no code was pushed. The virtual environment reused preinstalled scientific packages to avoid large downloads; application and development dependencies were installed and aligned to `requirements.lock`. An installation in a completely clean OS environment is NOT VERIFIED here.

## Commands actually run

Commands below omit the `.venv/bin/` prefix for readability. Complete final-gate argument lists, exit codes and measured elapsed times are in [verification/report.json](verification/report.json); captured output is in [verification/](verification/).

| Command | Observed result |
|---|---|
| `pip install -e '.[dev]'`, then `pip install -r requirements.lock` | Succeeded; all locked dependency versions satisfied |
| `energy generate --output runtime/raw.csv` | Generated 9,600 seeded hourly observations |
| `energy validate runtime/raw.csv` | Passed structural/quality tolerance gate with explicit sensor warnings; [data_validation.json](data_validation.json) |
| `energy train --output docs` | Fitted point + quantile models; tracked a run; registered and promoted version 1; [training_results.json](training_results.json) |
| `energy demo --output docs` | Both drift and degraded-MAE triggers; trained/registered version 2; promoted on later holdout; evaluated on 336 untouched future hours |
| `ruff check .` | Clean; exit 0 |
| `ruff format --check .` | All 22 Python files formatted; exit 0 |
| `pytest` | **36 passed**, 16 dependency warnings; exit 0 |
| `python scripts/smoke_api.py` | Started Uvicorn and made **9 real HTTP checks** over loopback; all passed; [api_smoke.json](api_smoke.json) |
| `python scripts/plot_demo.py` | Created the saved figure from the actual demo forecast CSV; visually inspected |
| `python scripts/verify.py` | Repeated lint, format, tests, isolated training/demo and live HTTP smoke; every step passed |
| YAML parsing of Compose, CI and provisioning; JSON parsing of dashboard | Syntax parsed successfully; this does not validate container execution |

The isolated repeat matched the original **input SHA-256, all training/baseline metrics, both future-test metric dictionaries and promotion outcome exactly**. It used a temporary independent tracking store which was removed after comparison; the original `runtime/mlruns` and `runtime/predictions.sqlite` remain available locally. The repeat took about 7.4 seconds for training and 4.7 seconds for the demo in this execution environment; these are not benchmarks for the user's physical laptop.

The final HTTP smoke exercised `/health`, `/model-version`, `/predict`, `/predict/batch`, `/actuals`, `/monitoring`, `/insights`, `/metrics`, and a 422 invalid-temperature request. The service process was terminated cleanly afterwards. Its registry model and SQLite ledger persist; start Uvicorn with the README command to serve it again.

An initial HTTP smoke attempt failed because inherited proxy variables required an unavailable SOCKS transport. The loopback client now uses `trust_env=False`; the actual API smoke was rerun successfully. No external network call is required for this check.

## Measured outcomes

- Initial validation: model MAE **15.2723 kW**, naive **21.3487 kW**, seasonal-naive **22.7851 kW**. Nominal 80% interval coverage: **81.1633%**.
- Promotion holdout: incumbent MAE **57.6959 kW**, challenger **16.2966 kW**. Version 2 promoted.
- Untouched shifted future test: incumbent MAE **68.8392 kW**, serving-after MAE **22.2684 kW**; a **67.65%** reduction in this particular synthetic scenario.
- Future interval coverage: **28.87% → 73.81%**, still below nominal 80%. Input drift remained flagged after promotion.

No result here is a claim about actual meters, consultancy customers, carbon reductions, or a production deployment. Feature importance is saved with the training report; it is a model diagnostic, not causal attribution.

## NOT VERIFIED

- Docker executable/build, Compose runtime, Grafana rendering, Prometheus scraping and MLflow web UI. Docker is not installed. Configuration files exist; future CI is configured to build the image.
- GitHub Actions execution. Only local commits exist; no remote configured or push performed.
- Any real OpenAI-compatible LLM call. Deterministic output and missing-key fallback were tested; live integration was not called.
- Python 3.11, native Windows, a fresh isolated OS install, load limits, authentication/TLS, distributed or crash-safe cross-store operation, real sensor generalization, interval calibration.
