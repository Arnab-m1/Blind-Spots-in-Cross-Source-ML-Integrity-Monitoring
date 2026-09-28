"""Compare task-aware source hardening with generic baselines on ICRISAT."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.certificates import task_visibility_certificate


SYSTEM = ROOT / "results" / "raw" / "e1_icrisat_linear" / "linear_system.npz"
RAW = ROOT / "results" / "raw" / "e5_hardening"
FIGURES = ROOT / "results" / "figures"
FEATURES = ["rainfall", "temperature", "irrigation", "fertilizer", "acreage"]
SEED = 42


def supports():
    for size in range(1, len(FEATURES) + 1):
        yield from itertools.combinations(range(len(FEATURES)), size)


def augmented_restricted_monitor(
    monitor: np.ndarray, support: tuple[int, ...], protected: set[int]
) -> np.ndarray:
    restricted = monitor[:, support]
    rows = [restricted]
    for source in sorted(protected):
        if source in support:
            row = np.zeros(len(support))
            row[support.index(source)] = 1.0
            rows.append(row[None, :])
    return np.vstack(rows)


def evaluate(
    monitor: np.ndarray,
    gradient: np.ndarray,
    protected: set[int],
) -> dict[str, float]:
    gammas: list[float] = []
    harmful: list[float] = []
    for support in supports():
        jacobian = augmented_restricted_monitor(monitor, support, protected)
        certificate = task_visibility_certificate(jacobian, gradient[list(support)])
        gammas.append(certificate.gamma)
        harmful.append(certificate.harmful_nullspace_score)
    positive = [value for value in gammas if value > 1e-12]
    return {
        "worst_gamma": float(min(gammas)),
        "median_gamma": float(np.median(gammas)),
        "minimum_positive_gamma": float(min(positive)) if positive else 0.0,
        "blind_supports": float(np.count_nonzero(np.asarray(gammas) <= 1e-12)),
        "worst_harmful_projection": float(max(harmful)),
        "adaptive_unit_budget_harm": float(max(harmful)),
    }


def greedy_task_aware(monitor: np.ndarray, gradient: np.ndarray, budget: int) -> set[int]:
    selected: set[int] = set()
    for _ in range(budget):
        choices = []
        for source in set(range(len(FEATURES))) - selected:
            candidate = selected | {source}
            metrics = evaluate(monitor, gradient, candidate)
            # Lexicographic objective avoids an all-zero worst margin plateau.
            objective = (
                -metrics["blind_supports"],
                metrics["minimum_positive_gamma"],
                metrics["median_gamma"],
            )
            choices.append((objective, source))
        selected.add(max(choices)[1])
    return selected


def generic_conditioning(monitor: np.ndarray, budget: int) -> set[int]:
    selected: set[int] = set()
    for _ in range(budget):
        scores = []
        for source in set(range(len(FEATURES))) - selected:
            candidate = selected | {source}
            augmented = np.vstack([monitor, np.eye(len(FEATURES))[sorted(candidate)]])
            scores.append((float(np.linalg.svd(augmented, compute_uv=False)[-1]), source))
        selected.add(max(scores)[1])
    return selected


def run() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    system = np.load(SYSTEM)
    monitor = system["monitor"]
    gradient = system["task_gradient"]
    rng = np.random.default_rng(SEED)
    random_order = rng.permutation(len(FEATURES)).tolist()
    gradient_order = np.argsort(-np.abs(gradient)).tolist()

    rows: list[dict[str, object]] = []
    for budget in range(len(FEATURES) + 1):
        methods = {
            "D0_none": set(),
            "D1_random": set(random_order[:budget]),
            "D2_conditioning": generic_conditioning(monitor, budget),
            "D3_task_gradient": set(gradient_order[:budget]),
            "D4_task_aware": greedy_task_aware(monitor, gradient, budget),
        }
        for method, selected in methods.items():
            metrics = evaluate(monitor, gradient, selected)
            rows.append(
                {
                    "budget": budget,
                    "method": method,
                    "protected_sources": "+".join(FEATURES[index] for index in sorted(selected)),
                    **metrics,
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(RAW / "hardening_results.csv", index=False)
    (RAW / "metadata.json").write_text(
        json.dumps({"seed": SEED, "features": FEATURES}, indent=2) + "\n"
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    styles = {
        "D1_random": ("#999999", "o"),
        "D2_conditioning": ("#0072B2", "s"),
        "D3_task_gradient": ("#E69F00", "^"),
        "D4_task_aware": ("#009E73", "D"),
    }
    for method, (color, marker) in styles.items():
        group = table[table["method"] == method]
        axes[0].plot(group["budget"], group["blind_supports"], color=color, marker=marker, label=method)
        axes[1].plot(group["budget"], group["worst_harmful_projection"], color=color, marker=marker, label=method)
    axes[0].set(xlabel="protected-source budget", ylabel="blind supports", title="(a) Blind-support elimination")
    axes[1].set(xlabel="protected-source budget", ylabel="worst harmful projection", title="(b) Residual attack headroom")
    for axis in axes:
        axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_icrisat_hardening.pdf")
    fig.savefig(FIGURES / "fig_icrisat_hardening.png", dpi=220)
    plt.close(fig)
    print(table[table["method"] == "D4_task_aware"].to_string(index=False))


if __name__ == "__main__":
    run()
