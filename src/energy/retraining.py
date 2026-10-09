"""Single-writer retraining with a chronological champion/challenger promotion gate."""

import fcntl
import json
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from energy.config import runtime_dir
from energy.model import evaluate
from energy.monitoring import snapshot
from energy.training import fit_candidate, load_champion, register, registry


@contextmanager
def retrain_lock():
    # Linux/macOS advisory process lock; released automatically even on process failure.
    with (runtime_dir() / "retrain.lock").open("w") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another retraining job is running") from exc
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def should_retrain(monitor):
    reasons = []
    if monitor["drift"]["status"] == "ok" and monitor["drift"]["drifted"]:
        reasons.append("input_drift")
    if monitor["performance"]["status"] == "ok" and monitor["performance"]["degraded"]:
        reasons.append("degraded_mae")
    return reasons


def promotion_gate(champion_mae, candidate_mae, minimum_improvement):
    return bool(candidate_mae < champion_mae * (1 - minimum_improvement))


def retrain(frame, settings, ledger, output: Path):
    with retrain_lock():
        champion, version = load_champion(settings)
        monitor = snapshot(ledger, champion, version, settings)
        reasons = should_retrain(monitor)
        decision = {
            "champion_before": version,
            "reasons": reasons,
            "monitor": monitor,
            "promoted": False,
            "status": "skipped",
        }
        if reasons:
            # Do not recycle data previously used for training or promotion evaluation.
            if pd.to_datetime(frame.timestamp, utc=True).min() <= pd.Timestamp(
                champion.evaluation_end_
            ):
                raise ValueError(
                    "Retraining data must start after champion's prior evaluation window"
                )
            challenger, report, gate_X, gate_y = fit_candidate(frame, settings)
            current_score = evaluate(champion, gate_X, gate_y)
            candidate_score = report["metrics"]["gradient_boosting"]
            candidate_version, run_id = register(challenger, report, settings, "retraining")
            promoted = promotion_gate(
                current_score["mae"], candidate_score["mae"], settings.min_relative_improvement
            )
            if promoted:
                registry().set_registered_model_alias(
                    settings.model_name, "champion", candidate_version
                )
            decision.update(
                status="promoted" if promoted else "rejected",
                promoted=promoted,
                candidate_version=candidate_version,
                run_id=run_id,
                champion_gate=current_score,
                candidate_gate=candidate_score,
                min_relative_improvement=settings.min_relative_improvement,
                validation_start=report["validation_start"],
                validation_end=report["validation_end"],
                data_sha256=report["data_sha256"],
            )
        ledger.decision(decision)
        output.mkdir(parents=True, exist_ok=True)
        (output / "retraining_decision.json").write_text(json.dumps(decision, indent=2))
        return decision
