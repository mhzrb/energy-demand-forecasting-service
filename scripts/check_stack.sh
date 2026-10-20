#!/usr/bin/env bash
# End-to-end check of the running Docker Compose stack: API -> Prometheus -> Grafana.
# Start the stack first (see README), then run:
#   GRAFANA_ADMIN_PASSWORD=<your password> bash scripts/check_stack.sh
set -u
: "${GRAFANA_ADMIN_PASSWORD:?Set GRAFANA_ADMIN_PASSWORD to the Grafana admin password}"

failures=0
request_file="$(mktemp)"
trap 'rm -f "$request_file"' EXIT

# wait_for "description" timeout_seconds command [args...]  (retries every 3 s)
wait_for() {
  local what=$1 limit=$2
  shift 2
  local deadline=$((SECONDS + limit))
  until "$@" >/dev/null 2>&1; do
    if ((SECONDS >= deadline)); then
      echo "FAIL: ${what} (no success within ${limit}s)"
      failures=$((failures + 1))
      return 1
    fi
    sleep 3
  done
  echo "ok:   ${what}"
}

# Each check succeeds only if the HTTP request works AND the body has the expected text.
api_healthy()      { [[ $(curl -fs localhost:8000/health) == *'"status":"ok"'* ]]; }
mlflow_healthy()   { curl -fs localhost:5000/health; }
prom_healthy()     { curl -fs localhost:9090/-/healthy; }
grafana_healthy()  { curl -fs localhost:3000/api/health; }
prom_target_up()   { [[ $(curl -fs localhost:9090/api/v1/targets) == *'"health":"up"'* ]]; }
prom_has_metric()  { [[ $(curl -fs "localhost:9090/api/v1/query?query=energy_predictions_total") == *'"result":[{'* ]]; }
dashboard_listed() { [[ $(curl -fs -u "admin:${GRAFANA_ADMIN_PASSWORD}" "localhost:3000/api/search?query=Energy") == *'Energy Demand Service'* ]]; }
grafana_to_prom()  { [[ $(curl -fs -u "admin:${GRAFANA_ADMIN_PASSWORD}" "localhost:3000/api/datasources/proxy/uid/prometheus/api/v1/query?query=energy_predictions_total") == *'"result":[{'* ]]; }
send_prediction()  { curl -fs localhost:8000/predict -H 'Content-Type: application/json' --data @"$request_file"; }

wait_for "API /health reports ok" 120 api_healthy
wait_for "MLflow UI responds" 120 mlflow_healthy
wait_for "Prometheus is healthy" 120 prom_healthy
wait_for "Grafana is healthy" 120 grafana_healthy

# Create one forecast so that the metrics have data to scrape.
docker compose exec -T api energy example >"$request_file"
wait_for "API serves a forecast (/predict)" 30 send_prediction

wait_for "Prometheus scrapes the API (target is up)" 90 prom_target_up
wait_for "Prometheus holds the API's forecast counter" 90 prom_has_metric
wait_for "Grafana lists the provisioned dashboard" 60 dashboard_listed
wait_for "Grafana can query Prometheus through its data source" 60 grafana_to_prom

if ((failures > 0)); then
  echo "${failures} check(s) failed."
  exit 1
fi
echo "All stack checks passed."
