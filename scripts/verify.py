"""Run the local verification gate and repeat the synthetic experiment in isolation."""

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def main():
    evidence = Path("docs/verification")
    evidence.mkdir(parents=True, exist_ok=True)
    report = {"python": platform.python_version(), "platform": platform.system(), "commands": []}

    def run(name, args, env=None):
        started = time.perf_counter()
        result = subprocess.run(args, capture_output=True, text=True, env=env, check=False)
        (evidence / f"{name}.log").write_text(result.stdout + result.stderr)
        report["commands"].append(
            {
                "command": args,
                "exit_code": result.returncode,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }
        )
        if result.returncode:
            raise RuntimeError(f"{name} failed; inspect {evidence / f'{name}.log'}")
        print(f"PASS {name}", flush=True)

    run("ruff", [sys.executable, "-m", "ruff", "check", "."])
    run("format", [sys.executable, "-m", "ruff", "format", "--check", "."])
    run("pytest", [sys.executable, "-m", "pytest"])
    with tempfile.TemporaryDirectory(prefix="energy-verification-") as work:
        env = os.environ.copy()
        env.pop("MLFLOW_TRACKING_URI", None)
        env["ENERGY_RUNTIME"] = str(Path(work) / "runtime")
        output = Path(work) / "evidence"
        run(
            "repeat_train",
            [sys.executable, "-m", "energy.cli", "train", "--output", str(output)],
            env,
        )
        run(
            "repeat_demo",
            [sys.executable, "-m", "energy.cli", "demo", "--output", str(output)],
            env,
        )
        expected = json.loads(Path("docs/training_results.json").read_text())
        repeated = json.loads((output / "training_results.json").read_text())
        assert expected["data_sha256"] == repeated["data_sha256"]
        assert expected["metrics"] == repeated["metrics"], "Training metrics changed"
        expected = json.loads(Path("docs/drift_demo.json").read_text())
        repeated = json.loads((output / "drift_demo.json").read_text())
        for key in ("future_test_original", "future_test_serving_after"):
            assert expected[key] == repeated[key], f"Demo metric changed: {key}"
        assert expected["decision"]["promoted"] == repeated["decision"]["promoted"]
        report["reproducibility"] = (
            "Exact training metrics, input hash and future-test metrics matched"
        )
    run("http_smoke", [sys.executable, "scripts/smoke_api.py"])
    report["docker"] = (
        "NOT VERIFIED: executable absent"
        if not shutil.which("docker")
        else ("NOT VERIFIED: this script does not build or launch containers")
    )
    report["remote_llm"] = "NOT VERIFIED: no live provider calls"
    report["github_actions"] = "NOT VERIFIED: no repository push or hosted run"
    report["passed"] = True
    (evidence / "report.json").write_text(json.dumps(report, indent=2))
    print("Verification complete", flush=True)


if __name__ == "__main__":
    main()
