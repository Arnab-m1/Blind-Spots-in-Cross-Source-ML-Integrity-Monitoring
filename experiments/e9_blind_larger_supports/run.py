"""Finite matched attacks on exactly blind size-three/four Hydraulic supports."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_hydraulic"))

from satml_monitoring.attacks import harmful_tangent_attack
from satml_monitoring.datasets.hydraulic import SENSOR_SPECS, TARGET_NAMES, newly_reassuring_error, summary_parameter_jacobian
from matched_security import parameter_columns, physical_classes, transformed_features, unit
from run import SEVERITY_SIGNS, nearest_class

CONFIG = ROOT / "configs/e9_blind_larger_supports.json"
DATA = ROOT / "data/hydraulic/processed"
MODEL = ROOT / "results/models/e3_hydraulic/seed_17.joblib"
CERTIFICATES = ROOT / "results/raw/e3_hydraulic/support_certificates.csv"
RAW = ROOT / "results/raw/e9_blind_larger_supports"
TABLES = ROOT / "results/tables"


def parse_support(name: str) -> tuple[int, ...]:
    lookup = {sensor.name: index for index, sensor in enumerate(SENSOR_SPECS)}
    return tuple(lookup[value] for value in name.split("+"))


def run() -> None:
    config = json.loads(CONFIG.read_text()); RAW.mkdir(parents=True, exist_ok=True); TABLES.mkdir(exist_ok=True)
    fit = joblib.load(MODEL); bundle = np.load(DATA / "cycle_features.npz"); splits = np.load(DATA / "grouped_splits.npz")
    features = bundle["features"]; targets = bundle["targets"][:, :4].astype(float) * SEVERITY_SIGNS
    train = splits["seed_17_train"]; validation = splits["seed_17_validation"]; test = splits["seed_17_test"]
    scaler = fit["scaler"]; basis = fit["pca"].components_[-config["monitor_dimension"]:]
    z_validation = scaler.transform(features[validation]); z_test = scaler.transform(features[test])
    threshold = float(np.quantile(np.sum((z_validation @ basis.T) ** 2, axis=1), .95))
    clean_score = np.sum((z_test @ basis.T) ** 2, axis=1); clean_prediction = fit["linear"].predict(z_test)
    true_z = fit["target_scaler"].transform(targets[test])
    bias_scale = np.maximum(np.std(features[train].reshape(-1, 17, 5)[..., 0], axis=0), 1e-8)
    physical_j = summary_parameter_jacobian(np.mean(features[train], axis=0, keepdims=True), bias_scale)[0]
    parameter_j = physical_j / scaler.scale_[:, None]
    monitor_j = basis @ parameter_j; gradients = fit["linear"].coef_ @ parameter_j
    broad = pd.read_csv(CERTIFICATES)
    chosen = []
    for size in config["support_sizes"]:
        group = broad[(broad.support_size == size) & broad.is_blind].sort_values(["support", "task"])
        chosen.append(group.head(config["configurations_per_support_size"]))
    selected = pd.concat(chosen, ignore_index=True)
    if any(len(selected[selected.support_size == size]) < config["configurations_per_support_size"] for size in config["support_sizes"]):
        raise AssertionError("insufficient exact-blind configurations")
    rows = []
    for case in selected.itertuples():
        support = parse_support(case.support); columns = parameter_columns(support)
        task_index = TARGET_NAMES.index(case.task); gradient = gradients[task_index, columns]
        restricted = monitor_j[:, columns]
        _, _, vh = np.linalg.svd(restricted, full_matrices=True)
        generic = -np.sign(vh[-1] @ gradient) * unit(vh[-1])
        harmful = harmful_tangent_attack(restricted, gradient, 1.0)
        if gradient @ harmful > 0: harmful = -harmful
        classes = physical_classes(fit, task_index)
        clean_class = nearest_class(clean_prediction[:, task_index], classes)
        true_class = nearest_class(true_z[:, task_index], classes)
        primary = (clean_class == true_class) & (clean_score <= threshold)
        budgets = [0.5, 0.5 * np.sqrt(len(support) / 2.0)]
        for budget_index, budget in enumerate(budgets):
            for attack, direction in (("A2_generic_stealth", generic), ("A4_harmful_nullspace", harmful)):
                theta = np.zeros(34); theta[columns] = budget * unit(direction)
                changed = transformed_features(features[test], theta, bias_scale)
                changed_z = scaler.transform(changed)
                prediction = fit["linear"].predict(changed_z)[:, task_index]
                score = np.sum((changed_z @ basis.T) ** 2, axis=1); alarm = score > threshold
                attacked_class = nearest_class(prediction, classes)
                adverse = newly_reassuring_error(clean_class, attacked_class, true_class)
                denominator = max(int(primary.sum()), 1)
                rows.append({"seed": 17, "support": case.support, "support_size": len(support),
                    "task": case.task, "certificate_gamma": case.gamma, "attack": attack,
                    "budget_design": "fixed_0.5" if budget_index == 0 else "per_source_rms_matched",
                    "budget": budget, "population": int(primary.sum()),
                    "meaningful_adverse_rate": float(np.sum(adverse & primary) / denominator),
                    "alarm_rate": float(np.sum(alarm & primary) / denominator),
                    "joint_adverse_no_alarm_rate": float(np.sum(adverse & ~alarm & primary) / denominator),
                    "median_task_movement": float(np.median((prediction - clean_prediction[:, task_index])[primary])) if np.any(primary) else np.nan,
                    "first_order_monitor_evidence": float(np.linalg.norm(restricted @ (budget * unit(direction))))})
    results = pd.DataFrame(rows); results.to_csv(RAW / "matched_attacks.csv", index=False)
    selected.to_csv(RAW / "selected_blind_configurations.csv", index=False)
    summary = results.groupby(["support_size", "budget_design", "attack"], as_index=False).agg(
        configurations=("support", "size"), median_adverse=("meaningful_adverse_rate", "median"),
        median_alarm=("alarm_rate", "median"), median_joint=("joint_adverse_no_alarm_rate", "median"),
        max_joint=("joint_adverse_no_alarm_rate", "max"), median_task_movement=("median_task_movement", "median"))
    summary.to_csv(TABLES / "table_blind_larger_supports.csv", index=False)
    metadata = {"protocol": config, "validation_threshold": threshold,
                "test_clean_alarm_rate": float(np.mean(clean_score > threshold)),
                "selected_configurations": len(selected)}
    (RAW / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
