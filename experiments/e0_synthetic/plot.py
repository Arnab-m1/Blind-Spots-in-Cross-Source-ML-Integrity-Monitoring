"""Plot the exact blind example and raw-versus-monitor phase diagram."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results" / "raw" / "e0_synthetic"
FIGURES = ROOT / "results" / "figures"


def run() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((RAW / "metadata.json").read_text(encoding="utf-8"))
    with (RAW / "phase_diagram.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8))

    grid = np.linspace(-1.2, 1.2, 100)
    for level in (-1.0, -0.5, 0.0, 0.5, 1.0):
        axes[0].plot(grid, grid - level, color="0.72", linewidth=1)
    axes[0].arrow(
        0,
        0,
        0.75,
        0.75,
        width=0.015,
        head_width=0.10,
        color="#c23b22",
        length_includes_head=True,
    )
    axes[0].set(
        xlim=(-1.2, 1.2),
        ylim=(-1.2, 1.2),
        xlabel=r"source $x_1$",
        ylabel=r"source $x_2$",
        title="(a) Exact harmful blind direction",
        aspect="equal",
    )
    axes[0].text(0.13, 0.82, r"$b(t)=(t,t)$", color="#c23b22")
    axes[0].text(-1.12, -0.98, r"gray: $x_1-x_2$ level sets", fontsize=8)

    colors = {0.20: "#0072B2", 0.25: "#009E73", 0.30: "#D55E00", 0.40: "#CC79A7", 0.55: "#333333"}
    for alpha in metadata["alphas"]:
        selected = [
            row
            for row in rows
            if float(row["alpha"]) == alpha
            and row["observer"] == "quadratic_monitor"
        ]
        horizons = np.asarray([int(row["horizon"]) for row in selected])
        theory = np.asarray([float(row["theoretical_error_sum"]) for row in selected])
        empirical = np.asarray([float(row["empirical_error_sum"]) for row in selected])
        axes[1].plot(
            horizons,
            theory,
            color=colors[alpha],
            linewidth=1.8,
            label=rf"$\alpha={alpha:.2f}$",
        )
        axes[1].scatter(horizons, empirical, color=colors[alpha], s=15)
    axes[1].set_xscale("log", base=2)
    axes[1].set(
        xlabel="observations T",
        ylabel=r"optimal error sum $\alpha_T+\beta_T$",
        title=r"(b) Quadratic-monitor boundary $T^{-1/4}$",
        ylim=(-0.03, 1.03),
    )
    axes[1].legend(frameon=False, fontsize=8, ncol=2)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"fig_synthetic_phase_diagram.{suffix}", dpi=220)
    plt.close(fig)
    print(f"wrote synthetic figure to {FIGURES}")


if __name__ == "__main__":
    run()
