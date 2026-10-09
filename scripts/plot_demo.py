"""Optional static figure: pip install -e '.[plots]', then run this script."""

from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

frame = pd.read_csv("docs/demo_forecasts.csv", parse_dates=["timestamp"]).head(168)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, (ax, residuals) = plt.subplots(
    2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]}
)
fig.suptitle("Synthetic energy demand · unseen week after the promotion gate", x=0.07, ha="left")
ax.fill_between(
    frame.timestamp,
    frame.lower_kw,
    frame.upper_kw,
    color="#16a085",
    alpha=0.18,
    label="Serving model: nominal 80% interval",
)
ax.plot(frame.timestamp, frame.actual_kw, color="#172b4d", lw=1.5, label="Actual synthetic demand")
ax.plot(
    frame.timestamp, frame.original_kw, color="#cc704b", lw=1, alpha=0.8, label="Original model"
)
ax.plot(frame.timestamp, frame.serving_kw, color="#138a72", lw=1.2, label="Promoted model")
ax.set_ylabel("Demand (kW)")
ax.legend(loc="upper left", fontsize=8, ncol=2)
residuals.plot(frame.timestamp, frame.original_kw - frame.actual_kw, color="#cc704b", lw=1)
residuals.plot(frame.timestamp, frame.serving_kw - frame.actual_kw, color="#138a72", lw=1)
residuals.axhline(0, color="#172b4d", lw=0.8)
residuals.set_ylabel("Error (kW)")
residuals.set_xlabel("UTC date · one-hour-ahead forecasts with observed historical demand")
for axis in (ax, residuals):
    axis.grid(alpha=0.15)
fig.autofmt_xdate(rotation=0, ha="center")
fig.tight_layout(rect=[0, 0, 1, 0.95])
Path("docs").mkdir(exist_ok=True)
fig.savefig("docs/drift_demo.png", dpi=150)
plt.close(fig)
