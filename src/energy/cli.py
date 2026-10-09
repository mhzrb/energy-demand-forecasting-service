"""Small, explicit command-line pipeline entry points."""

import argparse
import json
from pathlib import Path

import pandas as pd

from energy.config import load_settings, runtime_dir
from energy.data import generate, validate
from energy.demo import request_example, run_demo
from energy.monitoring import Ledger
from energy.retraining import retrain
from energy.training import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("train", "demo", "retrain"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--output", type=Path, default=Path("docs"))
        if name in {"train", "retrain"}:
            cmd.add_argument("--data", type=Path, required=name == "retrain")
    gen = sub.add_parser("generate")
    gen.add_argument("--output", type=Path, default=Path("runtime/raw.csv"))
    check = sub.add_parser("validate")
    check.add_argument("data", type=Path)
    sub.add_parser("example")
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.command == "generate":
        frame = generate(settings.hours, settings.seed, settings.start)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False)
        print(args.output)
    elif args.command == "validate":
        report = validate(pd.read_csv(args.data), allow_sensor_faults=True)
        print(json.dumps(report, indent=2))
        if report["errors"]:
            raise SystemExit(1)
    elif args.command == "train":
        frame = (
            pd.read_csv(args.data)
            if args.data
            else generate(settings.hours, settings.seed, settings.start)
        )
        result = train(frame, settings, args.output)
        print(json.dumps(result["metrics"], indent=2))
    elif args.command == "example":
        print(json.dumps(request_example(settings), indent=2))
    else:
        ledger = Ledger(runtime_dir() / "predictions.sqlite")
        if args.command == "demo":
            result = run_demo(settings, args.output, ledger)
            print(
                json.dumps(
                    {
                        k: result[k]
                        for k in (
                            "original_version",
                            "serving_version_after",
                            "future_test_original",
                            "future_test_serving_after",
                        )
                    },
                    indent=2,
                )
            )
        else:
            result = retrain(pd.read_csv(args.data), settings, ledger, args.output)
            print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
