"""Plot external-monitor audit selectivity from frozen raw results."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/e6_external_monitors"
OUT = ROOT / "results/figures/fig_external_monitor_selectivity"


def run() -> None:
    events = pd.read_csv(RAW / "matched_security.csv")
    frame = events[(events.attack == "A4_task_conditioned") &
                   (events.budget == events.budget.max())].copy()
    fig, axis = plt.subplots(figsize=(3.45, 2.35))
    for monitor, group in frame.groupby("monitor"):
        group = group.sort_values("gamma").reset_index(drop=True)
        group["audit_fraction"] = (group.index + 1) / len(group)
        found = (group.joint_adverse_no_alarm_rate > 0).cumsum()
        total = max(int((group.joint_adverse_no_alarm_rate > 0).sum()), 1)
        axis.plot(group.audit_fraction, found / total, linewidth=1.8,
                  label=monitor.upper())
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", label="random order")
    axis.set(xlabel="fraction audited, smallest margin first",
             ylabel="fraction of failures found", xlim=(0, 1), ylim=(0, 1.02))
    axis.grid(alpha=.2); axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    run()
