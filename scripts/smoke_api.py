"""Start a real loopback API process, exercise HTTP routes, and save responses."""

import argparse
import json
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx

from energy.config import load_settings
from energy.demo import request_example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("docs/api_smoke.json"))
    args = parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.with_suffix(".log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "energy.api:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            stdout=log,
            stderr=log,
        )
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=10, trust_env=False
            ) as client:
                for _ in range(100):
                    if process.poll() is not None:
                        raise RuntimeError("API exited before startup; inspect smoke log")
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    time.sleep(0.1)
                else:
                    raise RuntimeError("API did not become ready within ten seconds")
                results = {"transport": "real HTTP over loopback", "checks": []}

                def call(method, path, expected=200, payload=None):
                    response = client.request(method, path, json=payload)
                    assert response.status_code == expected, (
                        path,
                        response.status_code,
                        response.text,
                    )
                    body = response.text if path == "/metrics" else response.json()
                    results["checks"].append(
                        {
                            "method": method,
                            "path": path,
                            "status": response.status_code,
                            "response": body,
                        }
                    )
                    return body

                payload = request_example(load_settings())
                call("GET", "/health")
                call("GET", "/model-version")
                pred = call("POST", "/predict", payload=payload)
                assert pred["lower_kw"] <= pred["prediction_kw"] <= pred["upper_kw"]
                call("POST", "/predict/batch", payload={"items": [payload, payload]})
                call(
                    "POST",
                    "/actuals",
                    payload={"prediction_id": pred["prediction_id"], "actual_kw": 500},
                )
                call("GET", "/monitoring")
                call("POST", "/insights", payload=payload)
                metric_text = call("GET", "/metrics")
                assert "energy_predictions_total" in metric_text
                call("POST", "/predict", 422, {**payload, "temperature": 100})
                results["passed"] = True
                args.output.write_text(json.dumps(results, indent=2))
                print(f"{len(results['checks'])} real HTTP checks passed; evidence: {args.output}")
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


if __name__ == "__main__":
    main()
