"""Matched finite Hydraulic evaluation with joint task-and-alarm outcomes."""

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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import (
    SENSOR_SPECS,
    TARGET_NAMES,
    newly_reassuring_error,
    summary_parameter_jacobian,
    transform_summary_features,
)

from run import SEEDS, SEVERITY_SIGNS, TARGET_VALUES, nearest_class

DATA = ROOT / "data" / "hydraulic" / "processed"
MODELS = ROOT / "results" / "models" / "e3_hydraulic"
RAW = ROOT / "results" / "raw" / "e3_hydraulic"
FIGURES = ROOT / "results" / "figures"
CONFIG = ROOT / "configs" / "e3_hydraulic_matched.json"


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def parameter_columns(support: tuple[int, ...]) -> np.ndarray:
    return np.concatenate(
        [np.arange(2 * sensor, 2 * sensor + 2) for sensor in support]
    )


def support_name(support: tuple[int, ...]) -> str:
    return "+".join(SENSOR_SPECS[sensor].name for sensor in support)


def support_family(support: tuple[int, ...]) -> str:
    quantities = {SENSOR_SPECS[sensor].quantity for sensor in support}
    if len(quantities) == 1:
        return f"{next(iter(quantities))}_only"
    return "heterogeneous"


def matched_supports(pair_count: int, seed: int) -> list[tuple[int, ...]]:
    singles = list(itertools.combinations(range(len(SENSOR_SPECS)), 1))
    pairs = list(itertools.combinations(range(len(SENSOR_SPECS)), 2))
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(pairs), size=pair_count, replace=False))
    return singles + [pairs[int(index)] for index in chosen]


def orient_reassuring(direction: np.ndarray, gradient: np.ndarray) -> np.ndarray:
    direction = unit(direction)
    return -direction if float(gradient @ direction) > 0 else direction


def transformed_features(
    features: np.ndarray, theta: np.ndarray, bias_scales: np.ndarray
) -> np.ndarray:
    return transform_summary_features(
        features, theta[0::2], theta[1::2] * bias_scales
    )


def physical_classes(fit: dict[str, object], task_index: int) -> np.ndarray:
    severity = TARGET_VALUES[task_index] * SEVERITY_SIGNS[task_index]
    return (
        severity - fit["target_scaler"].mean_[task_index]
    ) / fit["target_scaler"].scale_[task_index]


def summarize_population(
    *,
    seed: int,
    support: tuple[int, ...],
    task: str,
    budget: float,
    attack: str,
    population: str,
    mask: np.ndarray,
    clean_correct: np.ndarray,
    clean_accepted: np.ndarray,
    meaningful_adverse: np.ndarray,
    alarm: np.ndarray,
    task_movement: np.ndarray,
    loss_increase: np.ndarray,
    perturbation_norm: float,
    feasible: bool,
) -> dict[str, object]:
    count = int(np.sum(mask))
    denominator = max(count, 1)
    joint = meaningful_adverse & ~alarm
    return {
        "seed": seed,
        "support": support_name(support),
        "support_family": support_family(support),
        "support_size": len(support),
        "task": task,
        "budget": budget,
        "attack": attack,
        "population": population,
        "population_size": count,
        "clean_correct_rate": float(np.sum(clean_correct & mask) / denominator),
        "clean_accept_rate": float(np.sum(clean_accepted & mask) / denominator),
        "meaningful_adverse_rate": float(np.sum(meaningful_adverse & mask) / denominator),
        "alarm_rate": float(np.sum(alarm & mask) / denominator),
        "joint_adverse_no_alarm_rate": float(np.sum(joint & mask) / denominator),
        "median_task_movement": float(np.median(task_movement[mask])) if count else np.nan,
        "median_loss_increase": float(np.median(loss_increase[mask])) if count else np.nan,
        "perturbation_norm": perturbation_norm,
        "feasible": feasible,
        "invalid_rate": float(not feasible),
    }


def aggregate_rates(
    frame: pd.DataFrame, group_columns: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    rate_columns = (
        "clean_correct_rate",
        "clean_accept_rate",
        "meaningful_adverse_rate",
        "alarm_rate",
        "joint_adverse_no_alarm_rate",
        "invalid_rate",
    )
    for keys, group in frame.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        weights = group["population_size"].to_numpy(dtype=float)
        total = float(np.sum(weights))
        row = dict(zip(group_columns, keys))
        row["evaluation_rows"] = int(total)
        for column in rate_columns:
            row[column] = float(
                np.sum(group[column].to_numpy(dtype=float) * weights) / max(total, 1.0)
            )
        row["median_task_movement"] = float(np.nanmedian(group["median_task_movement"]))
        row["median_loss_increase"] = float(np.nanmedian(group["median_loss_increase"]))
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text())
    budgets = tuple(float(value) for value in config["budgets"])
    attacks = tuple(config["attacks"])
    supports = matched_supports(
        config["support_design"]["pair_supports"],
        config["support_design"]["pair_sampling_seed"],
    )
    bundle = np.load(DATA / "cycle_features.npz")
    split_bundle = np.load(DATA / "grouped_splits.npz")
    features = bundle["features"]
    severity_targets = bundle["targets"][:, :4].astype(float) * SEVERITY_SIGNS
    max_test_rows = max(
        len(split_bundle[f"seed_{seed}_test"]) for seed in config["seeds"]
    )
    shape = (
        len(config["seeds"]), len(supports), 4, len(budgets),
        len(attacks), max_test_rows,
    )
    attacked_prediction = np.full(shape, np.nan, dtype=np.float32)
    monitor_score = np.full(shape, np.nan, dtype=np.float32)
    meaningful_event = np.zeros(shape, dtype=bool)
    alarm_event = np.zeros(shape, dtype=bool)
    joint_event = np.zeros(shape, dtype=bool)
    valid_state = np.zeros((len(config["seeds"]), max_test_rows), dtype=bool)
    clean_correct_store = np.zeros((len(config["seeds"]), 4, max_test_rows), dtype=bool)
    clean_accepted_store = np.zeros((len(config["seeds"]), max_test_rows), dtype=bool)
    test_indices_store = np.full((len(config["seeds"]), max_test_rows), -1, dtype=np.int32)
    summary_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    clean_rows: list[dict[str, object]] = []
    condition_rows: list[dict[str, object]] = []

    for seed_position, seed in enumerate(config["seeds"]):
        fit = joblib.load(MODELS / f"seed_{seed}.joblib")
        train = split_bundle[f"seed_{seed}_train"]
        validation = split_bundle[f"seed_{seed}_validation"]
        test = split_bundle[f"seed_{seed}_test"]
        row_count = len(test)
        valid_state[seed_position, :row_count] = True
        test_indices_store[seed_position, :row_count] = test
        scaler = fit["scaler"]
        z_validation = scaler.transform(features[validation])
        z_test = scaler.transform(features[test])
        monitor = fit["pca"].components_[-config["monitor_dimension"]:]
        threshold = float(np.quantile(
            np.sum((z_validation @ monitor.T) ** 2, axis=1),
            config["validation_alarm_quantile"],
        ))
        clean_scores = np.sum((z_test @ monitor.T) ** 2, axis=1)
        clean_accepted = clean_scores <= threshold
        clean_accepted_store[seed_position, :row_count] = clean_accepted
        clean_predictions = fit["linear"].predict(z_test)
        true_standardized = fit["target_scaler"].transform(severity_targets[test])
        sensor_means = features[train].reshape(-1, len(SENSOR_SPECS), 5)[..., 0]
        bias_scales = np.maximum(np.std(sensor_means, axis=0), 1e-8)
        physical_jacobian = summary_parameter_jacobian(
            np.mean(features[train], axis=0, keepdims=True), bias_scales
        )[0]
        parameter_jacobian = physical_jacobian / scaler.scale_[:, None]
        monitor_parameters = monitor @ parameter_jacobian
        gradients = fit["linear"].coef_ @ parameter_jacobian

        for task_index, task_name in enumerate(TARGET_NAMES[:4]):
            classes = physical_classes(fit, task_index)
            clean_class = nearest_class(clean_predictions[:, task_index], classes)
            true_class = nearest_class(true_standardized[:, task_index], classes)
            clean_correct = clean_class == true_class
            clean_correct_store[seed_position, task_index, :row_count] = clean_correct
            condition_labels = bundle["targets"][test, task_index]
            clean_rows.append({
                "seed": seed,
                "task": task_name,
                "test_cycles": row_count,
                "clean_task_accuracy": float(np.mean(clean_correct)),
                "clean_alarm_rate": float(np.mean(~clean_accepted)),
                "clean_correct_and_accepted": int(np.sum(clean_correct & clean_accepted)),
            })

            for support_position, support in enumerate(supports):
                cols = parameter_columns(support)
                restricted = monitor_parameters[:, cols]
                gradient = gradients[task_index, cols]
                _, _, vh = np.linalg.svd(restricted, full_matrices=True)
                rng = np.random.default_rng(
                    seed * 100_000 + support_position * 101
                )
                random_direction = unit(rng.normal(size=len(cols)))
                task_direction = -unit(gradient)
                stealth_direction = orient_reassuring(vh[-1], gradient)
                conditioned_direction = unit(
                    minimum_evidence_attack(restricted, gradient, -0.25)
                )
                directions = {
                    attacks[0]: random_direction,
                    attacks[1]: task_direction,
                    attacks[2]: stealth_direction,
                    attacks[3]: conditioned_direction,
                }
                certificate = task_visibility_certificate(restricted, gradient)

                for budget_position, budget in enumerate(budgets):
                    for attack_position, attack_name in enumerate(attacks):
                        local = budget * directions[attack_name]
                        theta = np.zeros(2 * len(SENSOR_SPECS))
                        theta[cols] = local
                        gain = theta[0::2]
                        feasible = bool(
                            np.linalg.norm(local) <= budget + config["feasibility"]["l2_budget_tolerance"]
                            and np.min(gain) >= config["feasibility"]["minimum_gain"]
                            and np.all(np.isfinite(local))
                        )
                        attacked_physical = transformed_features(
                            features[test], theta, bias_scales
                        )
                        attacked_z = scaler.transform(attacked_physical)
                        predictions = fit["linear"].predict(attacked_z)[:, task_index]
                        scores = np.sum((attacked_z @ monitor.T) ** 2, axis=1)
                        alarm = scores > threshold
                        attacked_class = nearest_class(predictions, classes)
                        # Smaller transformed-severity class means a healthier report.
                        adverse = newly_reassuring_error(
                            clean_class, attacked_class, true_class
                        )
                        joint = adverse & ~alarm
                        movement = predictions - clean_predictions[:, task_index]
                        clean_loss = np.abs(clean_predictions[:, task_index] - true_standardized[:, task_index])
                        attacked_loss = np.abs(predictions - true_standardized[:, task_index])
                        loss_increase = attacked_loss - clean_loss
                        index = (
                            seed_position, support_position, task_index,
                            budget_position, attack_position, slice(0, row_count),
                        )
                        attacked_prediction[index] = predictions.astype(np.float32)
                        monitor_score[index] = scores.astype(np.float32)
                        meaningful_event[index] = adverse
                        alarm_event[index] = alarm
                        joint_event[index] = joint
                        primary = clean_correct & clean_accepted
                        summary_rows.extend([
                            summarize_population(
                                seed=seed, support=support, task=task_name,
                                budget=budget, attack=attack_name,
                                population="all_test_cycles",
                                mask=np.ones(row_count, dtype=bool),
                                clean_correct=clean_correct,
                                clean_accepted=clean_accepted,
                                meaningful_adverse=adverse, alarm=alarm,
                                task_movement=movement,
                                loss_increase=loss_increase,
                                perturbation_norm=float(np.linalg.norm(local)),
                                feasible=feasible,
                            ),
                            summarize_population(
                                seed=seed, support=support, task=task_name,
                                budget=budget, attack=attack_name,
                                population="clean_correct_and_accepted",
                                mask=primary, clean_correct=clean_correct,
                                clean_accepted=clean_accepted,
                                meaningful_adverse=adverse, alarm=alarm,
                                task_movement=movement,
                                loss_increase=loss_increase,
                                perturbation_norm=float(np.linalg.norm(local)),
                                feasible=feasible,
                            ),
                        ])
                        for condition_label in np.unique(condition_labels):
                            condition_mask = primary & (condition_labels == condition_label)
                            if not np.any(condition_mask):
                                continue
                            condition_row = summarize_population(
                                seed=seed, support=support, task=task_name,
                                budget=budget, attack=attack_name,
                                population="clean_correct_and_accepted",
                                mask=condition_mask,
                                clean_correct=clean_correct,
                                clean_accepted=clean_accepted,
                                meaningful_adverse=adverse, alarm=alarm,
                                task_movement=movement,
                                loss_increase=loss_increase,
                                perturbation_norm=float(np.linalg.norm(local)),
                                feasible=feasible,
                            )
                            condition_row["condition_label"] = int(condition_label)
                            condition_rows.append(condition_row)
                        diagnostic_rows.append({
                            "seed": seed,
                            "support": support_name(support),
                            "support_family": support_family(support),
                            "support_size": len(support),
                            "task": task_name,
                            "budget": budget,
                            "attack": attack_name,
                            "gamma": certificate.gamma,
                            "is_blind": certificate.is_blind,
                            "first_order_task_shift": float(gradient @ local),
                            "first_order_monitor_shift": float(np.linalg.norm(restricted @ local)),
                            "threshold": threshold,
                            "constraint_residual": float(max(np.linalg.norm(local) - budget, -np.min(gain) - 1.0, 0.0)),
                            "solver_status": "analytic_feasible" if feasible else "analytic_infeasible",
                            "feasible": feasible,
                        })

    summary = pd.DataFrame(summary_rows)
    diagnostics = pd.DataFrame(diagnostic_rows)
    summary.to_csv(RAW / "matched_security_summary.csv", index=False)
    diagnostics.to_csv(RAW / "matched_security_diagnostics.csv", index=False)
    split_summary = aggregate_rates(
        summary, ["seed", "attack", "population"]
    )
    split_summary.to_csv(RAW / "matched_security_split_summary.csv", index=False)
    budget_summary = aggregate_rates(
        summary, ["seed", "budget", "attack", "population"]
    )
    budget_summary.to_csv(RAW / "matched_security_budget.csv", index=False)
    clean = pd.DataFrame(clean_rows)
    clean.to_csv(RAW / "matched_security_clean.csv", index=False)
    clean_seed = clean.groupby("seed", as_index=False).agg(
        clean_task_accuracy=("clean_task_accuracy", "mean"),
        clean_alarm_rate=("clean_alarm_rate", "first"),
        test_cycles=("test_cycles", "first"),
        conditional_task_cycles=("clean_correct_and_accepted", "sum"),
    )
    main_table = split_summary.merge(clean_seed, on="seed", how="left")
    main_table.to_csv(RAW / "matched_security_main_table.csv", index=False)
    factors = aggregate_rates(
        summary,
        ["seed", "attack", "task", "support_family", "budget", "population"],
    )
    diagnostic_factors = diagnostics.groupby(
        ["seed", "attack", "task", "support_family", "budget"],
        as_index=False,
    ).agg(
        first_order_task_shift=("first_order_task_shift", "median"),
        first_order_monitor_shift=("first_order_monitor_shift", "median"),
        max_constraint_residual=("constraint_residual", "max"),
    )
    factors = factors.merge(
        diagnostic_factors,
        on=["seed", "attack", "task", "support_family", "budget"],
        how="left",
    )
    factors["finite_nonlinearity_gap"] = (
        factors["median_task_movement"] - factors["first_order_task_shift"]
    )
    factors.to_csv(RAW / "matched_security_factor_summary.csv", index=False)
    conditions = aggregate_rates(
        pd.DataFrame(condition_rows),
        ["seed", "attack", "task", "condition_label", "support_family", "budget", "population"],
    )
    conditions.to_csv(RAW / "matched_security_condition_summary.csv", index=False)
    np.savez_compressed(
        RAW / "matched_security_events.npz",
        seeds=np.asarray(config["seeds"]),
        supports=np.asarray([support_name(value) for value in supports]),
        tasks=np.asarray(TARGET_NAMES[:4]),
        budgets=np.asarray(budgets),
        attacks=np.asarray(attacks),
        test_indices=test_indices_store,
        valid_state=valid_state,
        clean_correct=clean_correct_store,
        clean_accepted=clean_accepted_store,
        attacked_prediction=attacked_prediction,
        monitor_score=monitor_score,
        meaningful_adverse=meaningful_event,
        alarm=alarm_event,
        joint_adverse_no_alarm=joint_event,
    )
    primary_budget = budget_summary[
        budget_summary["population"] == "clean_correct_and_accepted"
    ]
    fig, axis = plt.subplots(figsize=(4.8, 3.6))
    for attack_name, group in primary_budget.groupby("attack"):
        aggregate = group.groupby("budget")["joint_adverse_no_alarm_rate"]
        median = aggregate.median()
        low = aggregate.min()
        high = aggregate.max()
        axis.plot(median.index, median.values, marker="o", label=attack_name.split("_", 1)[0])
        axis.fill_between(median.index, low.values, high.values, alpha=0.14)
    axis.set(
        xlabel="L2 parameter budget",
        ylabel="adverse outcome without alarm",
    )
    axis.grid(alpha=0.2)
    axis.legend(ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_hydraulic_matched_security.pdf")
    fig.savefig(FIGURES / "fig_hydraulic_matched_security.png", dpi=220)
    plt.close(fig)
    metadata = {
        "protocol": config,
        "event_array_shape": list(shape),
        "supports": [support_name(value) for value in supports],
        "test_rows_by_seed": {
            str(seed): int(np.sum(valid_state[position]))
            for position, seed in enumerate(config["seeds"])
        },
        "summary_rows": len(summary),
        "diagnostic_rows": len(diagnostics),
        "runtime_seconds": time.perf_counter() - started,
    }
    (RAW / "matched_security_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    print(split_summary.to_string(index=False))


if __name__ == "__main__":
    run()
