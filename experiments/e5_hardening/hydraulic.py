"""Hydraulic direct-sensor-anchor hardening and adaptive re-evaluation."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import SENSOR_SPECS, TARGET_NAMES, summary_parameter_jacobian

MODEL = ROOT / "results" / "models" / "e3_hydraulic" / "seed_17.joblib"
DATA = ROOT / "data" / "hydraulic" / "processed"
RAW = ROOT / "results" / "raw" / "e5_hardening"
FIGURES = ROOT / "results" / "figures"
SEED = 17


def supports():
    for size in (1, 2):
        yield from itertools.combinations(range(len(SENSOR_SPECS)), size)


def columns(support):
    return np.concatenate([np.arange(index * 2, (index + 1) * 2) for index in support])


def evaluate(monitor, gradients, protected):
    gammas, harmful = [], []
    for support in supports():
        cols = columns(support)
        rows = [monitor[:, cols]]
        for source in protected.intersection(support):
            local = support.index(source)
            anchor = np.zeros((2, len(cols)))
            anchor[:, local * 2:(local + 1) * 2] = np.eye(2)
            rows.append(anchor)
        jacobian = np.vstack(rows)
        for gradient in gradients:
            certificate = task_visibility_certificate(jacobian, gradient[cols])
            gammas.append(certificate.gamma)
            harmful.append(certificate.harmful_nullspace_score)
    positive = np.asarray(gammas)[np.asarray(gammas) > 1e-12]
    return {
        "blind_task_supports": int(np.sum(np.asarray(gammas) <= 1e-12)),
        "minimum_positive_gamma": float(np.min(positive)) if len(positive) else 0.0,
        "median_gamma": float(np.median(gammas)),
        "adaptive_unit_budget_harm": float(np.max(harmful)),
    }


def greedy(monitor, gradients, budget):
    selected = set()
    for _ in range(budget):
        choices = []
        for source in set(range(len(SENSOR_SPECS))) - selected:
            metrics = evaluate(monitor, gradients, selected | {source})
            objective = (-metrics["blind_task_supports"], metrics["minimum_positive_gamma"], metrics["median_gamma"])
            choices.append((objective, source))
        selected.add(max(choices)[1])
    return selected


def run():
    fit = joblib.load(MODEL)
    data = np.load(DATA / "cycle_features.npz")
    splits = np.load(DATA / "grouped_splits.npz")
    train = splits["seed_17_train"]
    features = data["features"]
    sensor_means = features[train].reshape(-1, len(SENSOR_SPECS), 5)[..., 0]
    bias_scales = np.maximum(np.std(sensor_means, axis=0), 1e-8)
    physical_j = summary_parameter_jacobian(
        np.mean(features[train], axis=0, keepdims=True), bias_scales
    )[0]
    feature_j = physical_j / fit["scaler"].scale_[:, None]
    monitor = fit["monitor"] @ feature_j
    gradients = [fit["linear"].coef_[index] @ feature_j for index in range(4)]
    rng = np.random.default_rng(SEED)
    random_order = rng.permutation(len(SENSOR_SPECS)).tolist()
    task_order = np.argsort(-np.max(np.abs(np.asarray(gradients)).reshape(4, 17, 2), axis=(0, 2))).tolist()
    generic_order = np.argsort(-np.linalg.norm(monitor.reshape(8, 17, 2), axis=(0, 2))).tolist()
    rows = []
    for budget in range(5):
        methods = {
            "D0_none": set(),
            "D1_random": set(random_order[:budget]),
            "D2_conditioning": set(generic_order[:budget]),
            "D3_task_gradient": set(task_order[:budget]),
            "D4_task_aware": greedy(monitor, gradients, budget),
        }
        for method, protected in methods.items():
            rows.append({
                "budget": budget,
                "method": method,
                "protected_sources": "+".join(SENSOR_SPECS[index].name for index in sorted(protected)),
                **evaluate(monitor, gradients, protected),
            })
    table = pd.DataFrame(rows)
    table.to_csv(RAW / "hydraulic_hardening_results.csv", index=False)
    (RAW / "hydraulic_metadata.json").write_text(json.dumps({
        "seed": SEED,
        "candidate_semantics": "A protected source adds two direct integrity-anchor rows for its gain and scaled-bias controls.",
        "evaluated_supports": "all source supports of size one and two across four tasks",
    }, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for method in ("D1_random", "D2_conditioning", "D3_task_gradient", "D4_task_aware"):
        group = table[table.method == method]
        axes[0].plot(group.budget, group.blind_task_supports, marker="o", label=method)
        axes[1].plot(group.budget, group.adaptive_unit_budget_harm, marker="o", label=method)
    axes[0].set(xlabel="protected-source budget", ylabel="blind task/support pairs", title="(a) Hydraulic blind pairs")
    axes[1].set(xlabel="protected-source budget", ylabel="adaptive attack headroom", title="(b) Hydraulic residual harm")
    for axis in axes: axis.grid(alpha=0.2)
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_hydraulic_hardening.pdf")
    fig.savefig(FIGURES / "fig_hydraulic_hardening.png", dpi=220)
    plt.close(fig)
    print(table[table.method == "D4_task_aware"].to_string(index=False))


if __name__ == "__main__":
    run()
