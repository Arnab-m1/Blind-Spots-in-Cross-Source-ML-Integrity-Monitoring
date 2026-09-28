"""Hydraulic attacks, support audit, and monitor-compression ablation."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from satml_monitoring.attacks import harmful_tangent_attack, minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import (
    SENSOR_SPECS,
    TARGET_NAMES,
    summary_parameter_jacobian,
    transform_summary_features,
)

from run import SEVERITY_SIGNS, TARGET_VALUES, nearest_class

DATA = ROOT / "data" / "hydraulic" / "processed"
MODEL = ROOT / "results" / "models" / "e3_hydraulic" / "seed_17.joblib"
RAW = ROOT / "results" / "raw" / "e3_hydraulic"
FIGURES = ROOT / "results" / "figures"
SEED = 17
BUDGETS = (0.1, 0.25, 0.5, 0.75, 1.0)
DIMENSIONS = (1, 2, 4, 8, 85)


def unit(vector):
    norm = float(np.linalg.norm(vector))
    return np.asarray(vector, dtype=float) / max(norm, 1e-12)


def support_family(support):
    names = [SENSOR_SPECS[index].name for index in support]
    prefixes = {name[:2] for name in names}
    if prefixes <= {"PS"}:
        return "pressure_only"
    if prefixes <= {"FS"}:
        return "flow_only"
    if prefixes <= {"TS"}:
        return "temperature_only"
    return f"heterogeneous_{len(support)}"


def oracle_direction(monitor, gradient, monitor_budget=0.05):
    result = minimize(
        lambda value: -float(gradient @ value),
        np.zeros(gradient.size),
        method="SLSQP",
        constraints=[
            {"type": "ineq", "fun": lambda value: 1.0 - np.linalg.norm(value)},
            {
                "type": "ineq",
                "fun": lambda value: monitor_budget - np.linalg.norm(monitor @ value),
            },
        ],
        options={"maxiter": 200, "ftol": 1e-10},
    )
    if not result.success:
        return None
    return unit(result.x)


def columns(support):
    return np.concatenate([np.arange(i * 2, (i + 1) * 2) for i in support])


def attacked_features(features, theta, bias_scales):
    gain = theta[0::2]
    bias = theta[1::2] * bias_scales
    return transform_summary_features(features, gain, bias)


def run() -> None:
    started = time.perf_counter()
    bundle = np.load(DATA / "cycle_features.npz")
    split_bundle = np.load(DATA / "grouped_splits.npz")
    fit = joblib.load(MODEL)
    features = bundle["features"]
    train_idx = split_bundle["seed_17_train"]
    val_idx = split_bundle["seed_17_validation"]
    test_idx = split_bundle["seed_17_test"]
    z_train = fit["scaler"].transform(features[train_idx])
    z_val = fit["scaler"].transform(features[val_idx])
    z_test = fit["scaler"].transform(features[test_idx])
    severity_targets = bundle["targets"][:, :4].astype(float) * SEVERITY_SIGNS
    clean_prediction = fit["linear"].predict(z_test)
    sensor_means = features[train_idx].reshape(-1, len(SENSOR_SPECS), 5)[..., 0]
    bias_scales = np.maximum(np.std(sensor_means, axis=0), 1e-8)
    feature_jacobian = summary_parameter_jacobian(
        np.mean(features[train_idx], axis=0, keepdims=True), bias_scales
    )[0]
    standardized_jacobian = feature_jacobian / fit["scaler"].scale_[:, None]
    full_components = fit["pca"].components_
    rng = np.random.default_rng(SEED)
    support_pool = []
    for size in (1, 2, 3):
        support_pool.extend(itertools.combinations(range(len(SENSOR_SPECS)), size))
    size_four = list(itertools.combinations(range(len(SENSOR_SPECS)), 4))
    support_pool.extend(size_four[i] for i in rng.choice(len(size_four), 500, replace=False))

    compression_rows = []
    attack_rows = []
    oracle_failures = 0
    for dimension in DIMENSIONS:
        monitor = full_components[-dimension:]
        val_score = np.sum((z_val @ monitor.T) ** 2, axis=1)
        threshold = float(np.quantile(val_score, 0.95))
        clean_score = np.sum((z_test @ monitor.T) ** 2, axis=1)
        monitor_parameters = monitor @ standardized_jacobian
        corruption_alarms = []
        for _ in range(100):
            sensor = int(rng.integers(0, len(SENSOR_SPECS)))
            theta = np.zeros(2 * len(SENSOR_SPECS))
            theta[2 * sensor : 2 * sensor + 2] = 0.5 * unit(rng.normal(size=2))
            corrupted = fit["scaler"].transform(
                attacked_features(features[test_idx], theta, bias_scales)
            )
            corruption_alarms.append(np.mean(np.sum((corrupted @ monitor.T) ** 2, axis=1) > threshold))
        gammas = []
        blind = 0
        # All task/support combinations; the linear model makes these state-independent.
        for support in support_pool:
            cols = columns(support)
            restricted = monitor_parameters[:, cols]
            for task_index, task_name in enumerate(TARGET_NAMES[:4]):
                full_gradient = fit["linear"].coef_[task_index] @ standardized_jacobian
                gradient = full_gradient[cols]
                cert = task_visibility_certificate(restricted, gradient)
                gammas.append(cert.gamma)
                blind += int(cert.is_blind)
                if dimension == 8 and len(support) <= 2:
                    _, _, vh = np.linalg.svd(restricted, full_matrices=True)
                    directions = {
                        "A0_random": unit(rng.normal(size=len(cols))),
                        "A1_task_gradient": unit(gradient),
                        "A2_generic_stealth": unit(vh[-1]),
                    }
                    conditioned = minimum_evidence_attack(
                        restricted, gradient, 0.25
                    )
                    directions["A4_task_conditioned"] = unit(conditioned)
                    directions["A6_constrained_oracle"] = oracle_direction(
                        restricted, gradient
                    )
                    if directions["A6_constrained_oracle"] is None:
                        oracle_failures += 1
                    if cert.is_blind:
                        tangent = unit(harmful_tangent_attack(restricted, gradient, 1.0))
                        directions["A3_harmful_tangent"] = tangent
                        directions["A5_linear_continuation"] = tangent

                    for attack_name, direction in directions.items():
                        if direction is None:
                            continue
                        for budget in BUDGETS:
                            shift = budget * direction
                            full_shift = np.zeros(2 * len(SENSOR_SPECS)); full_shift[cols] = shift
                            attacked_physical = attacked_features(
                                features[test_idx], full_shift, bias_scales
                            )
                            attacked_z = fit["scaler"].transform(attacked_physical)
                            attacked_score = np.sum((attacked_z @ monitor.T) ** 2, axis=1)
                            alarm = attacked_score > threshold
                            attacked_prediction = fit["linear"].predict(attacked_z)[:, task_index]
                            classes_physical = TARGET_VALUES[task_index] * SEVERITY_SIGNS[task_index]
                            classes = (classes_physical - fit["target_scaler"].mean_[task_index]) / fit["target_scaler"].scale_[task_index]
                            clean_class = nearest_class(clean_prediction[:, task_index], classes)
                            attacked_class = nearest_class(attacked_prediction, classes)
                            true_class = fit["target_scaler"].transform(severity_targets[test_idx])[:, task_index]
                            harmful = np.abs(attacked_prediction - true_class) > np.abs(
                                clean_prediction[:, task_index] - true_class
                            )
                            attack_rows.append({
                                "support": "+".join(SENSOR_SPECS[i].name for i in support),
                                "support_family": support_family(support),
                                "support_size": len(support), "task": task_name,
                                "attack": attack_name,
                                "gamma": cert.gamma, "is_blind": cert.is_blind,
                                "budget": budget, "perturbation_norm": float(np.linalg.norm(shift)),
                                "task_shift": float(gradient @ shift),
                                "monitor_shift": float(np.linalg.norm(restricted @ shift)),
                                "realized_task_shift": float(np.mean(attacked_prediction - clean_prediction[:, task_index])),
                                "parameterization": "sensor_gain_and_scaled_bias",
                                "alarm_probability": float(np.mean(alarm)),
                                "detection_rate": float(np.mean(alarm)),
                                "false_negative_rate": float(np.mean(harmful & ~alarm)),
                                "class_flip_rate": float(np.mean(clean_class != attacked_class)),
                            })
        compression_rows.append({
            "dimension": dimension, "test_fpr": float(np.mean(clean_score > threshold)),
            "ordinary_corruption_detection": float(np.mean(corruption_alarms)),
            "median_gamma": float(np.median(gammas)), "blind_fraction": blind / len(gammas),
        })

    compression = pd.DataFrame(compression_rows)
    attacks = pd.DataFrame(attack_rows)
    compression.to_csv(RAW / "compression_ablation.csv", index=False)
    attacks.to_csv(RAW / "attack_results.csv", index=False)
    grouped = attacks[attacks["attack"] == "A4_task_conditioned"].groupby(["support", "task"]).agg(gamma=("gamma", "first"), harm=("task_shift", lambda x: float(np.max(np.abs(x))))).reset_index()
    correlation = float(spearmanr(grouped["gamma"], grouped["harm"]).statistic)
    summary = {
        "gamma_harm_spearman": correlation,
        "attack_rows": len(attacks),
        "oracle_failures": oracle_failures,
        "runtime_seconds": time.perf_counter() - started,
    }
    (RAW / "attack_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].plot(compression["dimension"], compression["blind_fraction"], marker="o", color="#0072B2")
    axes[0].set(xlabel="monitor dimension", ylabel="blind task/support fraction", title="(a) Compression and blindness", xscale="log")
    scatter = axes[1].scatter(grouped["gamma"], grouped["harm"], c=grouped["gamma"], cmap="viridis", s=18)
    axes[1].set(xlabel=r"$\Gamma_M$", ylabel="maximum local task shift", title="(b) Certificate versus attackability", xscale="symlog")
    for axis in axes: axis.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_hydraulic_attacks.pdf"); fig.savefig(FIGURES / "fig_hydraulic_attacks.png", dpi=220); plt.close(fig)
    print(json.dumps({**summary, "compression": compression_rows}, indent=2))


if __name__ == "__main__": run()
