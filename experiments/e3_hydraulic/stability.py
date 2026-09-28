"""Five-split stability and task-conditioning controls for Hydraulic results."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import (
    SENSOR_SPECS,
    TARGET_NAMES,
    summary_parameter_jacobian,
    transform_summary_features,
)

from run import SEEDS

DATA = ROOT / "data" / "hydraulic" / "processed"
MODELS = ROOT / "results" / "models" / "e3_hydraulic"
RAW = ROOT / "results" / "raw" / "e3_hydraulic"
BUDGETS = (0.1, 0.25, 0.5, 0.75, 1.0)


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def columns(support: tuple[int, ...]) -> np.ndarray:
    return np.concatenate([np.arange(2 * sensor, 2 * sensor + 2) for sensor in support])


def transform(features: np.ndarray, theta: np.ndarray, bias_scales: np.ndarray) -> np.ndarray:
    return transform_summary_features(features, theta[0::2], theta[1::2] * bias_scales)


def run() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA / "cycle_features.npz")
    splits = np.load(DATA / "grouped_splits.npz")
    features = data["features"]
    rows: list[dict[str, object]] = []
    comparator_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        fit = joblib.load(MODELS / f"seed_{seed}.joblib")
        train = splits[f"seed_{seed}_train"]
        validation = splits[f"seed_{seed}_validation"]
        test = splits[f"seed_{seed}_test"]
        z_validation = fit["scaler"].transform(features[validation])
        z_test = fit["scaler"].transform(features[test])
        monitor = fit["pca"].components_[-8:]
        threshold = float(np.quantile(np.sum((z_validation @ monitor.T) ** 2, axis=1), 0.95))
        sensor_means = features[train].reshape(-1, len(SENSOR_SPECS), 5)[..., 0]
        bias_scales = np.maximum(np.std(sensor_means, axis=0), 1e-8)
        physical_jacobian = summary_parameter_jacobian(
            np.mean(features[train], axis=0, keepdims=True), bias_scales
        )[0]
        parameter_jacobian = physical_jacobian / fit["scaler"].scale_[:, None]
        monitor_parameters = monitor @ parameter_jacobian
        gradients = fit["linear"].coef_ @ parameter_jacobian
        clean_predictions = fit["linear"].predict(z_test)

        attack_supports = [
            support
            for size in (1, 2)
            for support in itertools.combinations(range(len(SENSOR_SPECS)), size)
        ]
        rng = np.random.default_rng(17)
        audit_supports = [
            support
            for size in (1, 2, 3)
            for support in itertools.combinations(range(len(SENSOR_SPECS)), size)
        ]
        size_four = list(itertools.combinations(range(len(SENSOR_SPECS)), 4))
        audit_supports.extend(
            size_four[index]
            for index in rng.choice(len(size_four), 500, replace=False)
        )
        blind = 0
        total = 0
        for support in audit_supports:
            cols = columns(support)
            restricted = monitor_parameters[:, cols]
            for task_index in range(4):
                certificate = task_visibility_certificate(
                    restricted, gradients[task_index, cols]
                )
                total += 1
                blind += int(certificate.is_blind)

        for support in attack_supports:
            cols = columns(support)
            restricted = monitor_parameters[:, cols]
            singular_values = np.linalg.svd(restricted, compute_uv=False)
            sigma_min = float(singular_values[-1]) if singular_values.size else 0.0
            _, _, vh = np.linalg.svd(restricted, full_matrices=True)
            generic = unit(vh[-1])
            for task_index, task_name in enumerate(TARGET_NAMES[:4]):
                gradient = gradients[task_index, cols]
                certificate = task_visibility_certificate(restricted, gradient)
                conditioned = unit(minimum_evidence_attack(restricted, gradient, 0.25))
                if seed == SEEDS[0]:
                    comparator_rows.append({
                        "seed": seed,
                        "support": "+".join(SENSOR_SPECS[index].name for index in support),
                        "support_size": len(support),
                        "task": task_name,
                        "rank": certificate.rank,
                        "sigma_min": sigma_min,
                        "h": certificate.harmful_nullspace_score,
                        "gamma": certificate.gamma,
                    })
                for attack_name, direction in (
                    ("A2_generic_stealth", generic),
                    ("A4_task_conditioned", conditioned),
                ):
                    for budget in BUDGETS:
                        local = budget * direction
                        theta = np.zeros(2 * len(SENSOR_SPECS))
                        theta[cols] = local
                        attacked_z = fit["scaler"].transform(
                            transform(features[test], theta, bias_scales)
                        )
                        attacked_predictions = fit["linear"].predict(attacked_z)[:, task_index]
                        scores = np.sum((attacked_z @ monitor.T) ** 2, axis=1)
                        rows.append({
                            "seed": seed,
                            "support": "+".join(SENSOR_SPECS[index].name for index in support),
                            "task": task_name,
                            "attack": attack_name,
                            "budget": budget,
                            "gamma": certificate.gamma,
                            "realized_task_shift": float(np.mean(attacked_predictions - clean_predictions[:, task_index])),
                            "detection_rate": float(np.mean(scores > threshold)),
                        })
        for row in rows:
            if row["seed"] == seed:
                row["blind_fraction_seed"] = blind / total

    attacks = pd.DataFrame(rows)
    attacks.to_csv(RAW / "seed_stability_attacks.csv", index=False)
    comparators = pd.DataFrame(comparator_rows)
    comparators.to_csv(RAW / "task_metric_comparison.csv", index=False)
    positive = comparators[comparators.gamma > 0]
    ratios = positive.groupby("support").gamma.agg(lambda values: values.max() / values.min())
    example_support = str(ratios.idxmax())
    example_attacks = attacks[
        (attacks.seed == SEEDS[0])
        & (attacks.support == example_support)
        & (attacks.attack == "A4_task_conditioned")
    ].groupby("task").agg(
        median_abs_task_shift=("realized_task_shift", lambda values: float(np.median(np.abs(values)))),
        median_detection=("detection_rate", "median"),
    ).reset_index()
    example = comparators[comparators.support == example_support].merge(example_attacks, on="task")
    example.to_csv(RAW / "task_metric_example.csv", index=False)
    per_seed = attacks.groupby(["seed", "attack"]).agg(
        median_abs_task_shift=("realized_task_shift", lambda values: float(np.median(np.abs(values)))),
        median_detection=("detection_rate", "median"),
        blind_fraction=("blind_fraction_seed", "first"),
    ).reset_index()
    per_seed.to_csv(RAW / "seed_stability_summary.csv", index=False)
    aggregate = per_seed.groupby("attack").agg(
        harm_median=("median_abs_task_shift", "median"),
        harm_min=("median_abs_task_shift", "min"),
        harm_max=("median_abs_task_shift", "max"),
        detection_median=("median_detection", "median"),
        detection_min=("median_detection", "min"),
        detection_max=("median_detection", "max"),
        blind_fraction_median=("blind_fraction", "median"),
        blind_fraction_min=("blind_fraction", "min"),
        blind_fraction_max=("blind_fraction", "max"),
    ).reset_index()
    aggregate.to_csv(RAW / "seed_stability_aggregate.csv", index=False)
    metadata = {
        "seeds": list(SEEDS),
        "attack_supports_per_seed": 153,
        "audit_supports_per_seed": 1333,
        "tasks": 4,
        "budgets": list(BUDGETS),
        "comparison_note": "rank and sigma_min are task-independent within each support; gamma and h are task-conditioned",
        "task_metric_example_support": example_support,
    }
    (RAW / "seed_stability_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(per_seed.to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    run()
