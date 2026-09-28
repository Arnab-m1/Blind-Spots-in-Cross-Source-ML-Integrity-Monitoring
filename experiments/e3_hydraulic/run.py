"""Hydraulic clean models, learned monitors, and source-block certificates."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import (
    SENSOR_SPECS,
    TARGET_NAMES,
    summary_parameter_jacobian,
)


DATA = ROOT / "data" / "hydraulic" / "processed"
RAW = ROOT / "results" / "raw" / "e3_hydraulic"
MODELS = ROOT / "results" / "models" / "e3_hydraulic"
SEEDS = (17, 29, 43, 71, 101)
TARGET_VALUES = (
    np.array([3, 20, 100]),
    np.array([73, 80, 90, 100]),
    np.array([0, 1, 2]),
    np.array([90, 100, 115, 130]),
)
# Larger value means more severe degradation for every transformed task.
SEVERITY_SIGNS = np.array([-1.0, -1.0, 1.0, -1.0])
RANK_RTOL = 1e-10


def nearest_class(values: np.ndarray, classes: np.ndarray) -> np.ndarray:
    return classes[np.argmin(np.abs(values[:, None] - classes[None, :]), axis=1)]


def parameter_columns(support: tuple[int, ...]) -> np.ndarray:
    return np.concatenate([np.arange(index * 2, (index + 1) * 2) for index in support])


def support_family(rng: np.random.Generator):
    source_count = len(SENSOR_SPECS)
    for size in (1, 2, 3):
        yield from itertools.combinations(range(source_count), size)
    size_four = list(itertools.combinations(range(source_count), 4))
    chosen = rng.choice(len(size_four), size=min(500, len(size_four)), replace=False)
    for index in chosen:
        yield size_four[int(index)]


def clean_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    balanced: list[float] = []
    macro_f1: list[float] = []
    for column, classes in enumerate(TARGET_VALUES):
        predicted_class = nearest_class(y_pred[:, column], classes)
        balanced.append(balanced_accuracy_score(y_true[:, column], predicted_class))
        macro_f1.append(f1_score(y_true[:, column], predicted_class, average="macro"))
    return {
        "balanced_accuracy_mean": float(np.mean(balanced)),
        "macro_f1_mean": float(np.mean(macro_f1)),
        **{f"{TARGET_NAMES[i]}_balanced_accuracy": float(balanced[i]) for i in range(4)},
        **{f"{TARGET_NAMES[i]}_macro_f1": float(macro_f1[i]) for i in range(4)},
    }


def masked_training_data(
    features: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    repeats = 2
    inputs = np.repeat(features, repeats, axis=0)
    targets = np.repeat(features, repeats, axis=0)
    chosen_sources = rng.integers(0, len(SENSOR_SPECS), size=inputs.shape[0])
    for row, source in enumerate(chosen_sources):
        inputs[row, source * 5 : (source + 1) * 5] = 0.0
    return inputs, targets


def run() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    bundle = np.load(DATA / "cycle_features.npz")
    splits_bundle = np.load(DATA / "grouped_splits.npz")
    features = bundle["features"]
    targets = bundle["targets"][:, :4].astype(float)
    severity_targets = targets * SEVERITY_SIGNS
    all_metrics: list[dict[str, object]] = []

    for seed in SEEDS:
        started = time.perf_counter()
        train_indices = splits_bundle[f"seed_{seed}_train"]
        validation_indices = splits_bundle[f"seed_{seed}_validation"]
        test_indices = splits_bundle[f"seed_{seed}_test"]
        scaler = StandardScaler().fit(features[train_indices])
        z_train = scaler.transform(features[train_indices])
        z_validation = scaler.transform(features[validation_indices])
        z_test = scaler.transform(features[test_indices])

        target_scaler = StandardScaler().fit(severity_targets[train_indices])
        y_train = target_scaler.transform(severity_targets[train_indices])
        y_validation = target_scaler.transform(severity_targets[validation_indices])

        linear = Ridge(alpha=10.0).fit(z_train, y_train)
        nonlinear_candidates = []
        for hidden, alpha in (((32, 16), 1e-3), ((64, 32), 1e-3), ((64, 32), 1e-2)):
            candidate = MLPRegressor(
                hidden_layer_sizes=hidden, activation="relu", alpha=alpha,
                early_stopping=True, validation_fraction=0.15,
                max_iter=1200, random_state=seed,
            ).fit(z_train, y_train)
            error = float(np.mean((candidate.predict(z_validation) - y_validation) ** 2))
            nonlinear_candidates.append((error, candidate))
        nonlinear = min(nonlinear_candidates, key=lambda item: item[0])[1]
        linear_prediction = target_scaler.inverse_transform(linear.predict(z_test)) * SEVERITY_SIGNS
        nonlinear_prediction = target_scaler.inverse_transform(nonlinear.predict(z_test)) * SEVERITY_SIGNS

        pca = PCA(random_state=seed).fit(z_train)
        monitor = pca.components_[-8:]
        validation_score = np.sum((z_validation @ monitor.T) ** 2, axis=1)
        threshold = float(np.quantile(validation_score, 0.95))
        test_score = np.sum((z_test @ monitor.T) ** 2, axis=1)

        rng = np.random.default_rng(seed)
        masked_inputs, masked_targets = masked_training_data(z_train, rng)
        reconstruction = MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            early_stopping=True,
            validation_fraction=0.15,
            max_iter=350,
            random_state=seed,
        ).fit(masked_inputs, masked_targets)

        all_metrics.extend(
            [
                {"seed": seed, "model": "ridge", **clean_metrics(targets[test_indices], linear_prediction)},
                {"seed": seed, "model": "shared_mlp", **clean_metrics(targets[test_indices], nonlinear_prediction)},
                {
                    "seed": seed,
                    "model": "monitor",
                    "validation_threshold": threshold,
                    "test_fpr": float(np.mean(test_score > threshold)),
                    "runtime_seconds": time.perf_counter() - started,
                },
            ]
        )
        joblib.dump(
            {
                "scaler": scaler,
                "target_scaler": target_scaler,
                "linear": linear,
                "nonlinear": nonlinear,
                "pca": pca,
                "monitor": monitor,
                "reconstruction": reconstruction,
                "threshold": threshold,
            },
            MODELS / f"seed_{seed}.joblib",
        )

        if seed == SEEDS[0]:
            sensor_means = features[train_indices].reshape(-1, len(SENSOR_SPECS), 5)[..., 0]
            bias_scales = np.maximum(np.std(sensor_means, axis=0), 1e-8)
            feature_jacobian = summary_parameter_jacobian(
                np.mean(features[train_indices], axis=0, keepdims=True), bias_scales
            )[0]
            standardized_jacobian = feature_jacobian / scaler.scale_[:, None]
            support_rows: list[dict[str, object]] = []
            for support in support_family(rng):
                columns = parameter_columns(support)
                restricted_monitor = (monitor @ standardized_jacobian)[:, columns]
                for task_index, task_name in enumerate(TARGET_NAMES[:4]):
                    gradient = (linear.coef_[task_index] @ standardized_jacobian)[columns]
                    certificate = task_visibility_certificate(
                        restricted_monitor, gradient, rtol=RANK_RTOL
                    )
                    support_rows.append(
                        {
                            "seed": seed,
                            "support": "+".join(SENSOR_SPECS[index].name for index in support),
                            "support_size": len(support),
                            "task": task_name,
                            "rank": certificate.rank,
                            "nullity": certificate.nullity,
                            "h": certificate.harmful_nullspace_score,
                            "gamma": certificate.gamma,
                            "is_blind": certificate.is_blind,
                            "parameterization": "sensor_gain_and_scaled_bias",
                        }
                    )
            pd.DataFrame(support_rows).to_csv(RAW / "support_certificates.csv", index=False)
            joblib.dump(bias_scales, MODELS / "seed_17_bias_scales.joblib")

    metrics_table = pd.DataFrame(all_metrics)
    metrics_table.to_csv(RAW / "clean_metrics.csv", index=False)
    summary = {
        "seeds": list(SEEDS),
        "cycles": int(features.shape[0]),
        "sources": len(SENSOR_SPECS),
        "features": int(features.shape[1]),
        "linear_balanced_accuracy_mean": float(
            metrics_table[metrics_table["model"] == "ridge"]["balanced_accuracy_mean"].mean()
        ),
        "mlp_balanced_accuracy_mean": float(
            metrics_table[metrics_table["model"] == "shared_mlp"]["balanced_accuracy_mean"].mean()
        ),
        "monitor_test_fpr_mean": float(
            metrics_table[metrics_table["model"] == "monitor"]["test_fpr"].mean()
        ),
    }
    (RAW / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
