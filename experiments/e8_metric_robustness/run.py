"""Whitening and scalar-alarm robustness on frozen held-out finite labels."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_hydraulic"))

from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import SENSOR_SPECS, TARGET_NAMES, summary_parameter_jacobian
from matched_security import parameter_columns

DATA = ROOT / "data/hydraulic/processed"
MODELS = ROOT / "results/models/e3_hydraulic"
CASES = ROOT / "results/raw/e3_hydraulic/prioritization_cases.csv"
RAW = ROOT / "results/raw/e8_metric_robustness"
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
CONFIG = ROOT / "configs/e8_metric_robustness.json"


def parse_support(name: str) -> tuple[int, ...]:
    lookup = {sensor.name: index for index, sensor in enumerate(SENSOR_SPECS)}
    return tuple(lookup[value] for value in name.split("+"))


def rank_metrics(frame: pd.DataFrame, metrics: dict[str, bool], budgets: list[int]) -> pd.DataFrame:
    rows = []
    for seed, group in frame.groupby("seed"):
        total = int(group.verified_failure.sum())
        for metric, descending in metrics.items():
            ordered = group.sort_values(metric, ascending=not descending, kind="stable")
            for budget in budgets:
                k = min(budget, len(ordered)); found = int(ordered.head(k).verified_failure.sum())
                rows.append({"seed": seed, "metric": metric, "budget": budget,
                             "failures_found": found, "precision": found / k,
                             "recall": found / max(total, 1)})
    return pd.DataFrame(rows)


def paired_intervals(metrics: pd.DataFrame, repetitions: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed); rows = []
    reference = "gamma_standardized_identity"
    for budget in sorted(metrics.budget.unique()):
        pivot = metrics[metrics.budget == budget].pivot(index="seed", columns="metric", values="failures_found")
        for comparator in pivot.columns:
            if comparator == reference:
                continue
            differences = (pivot[reference] - pivot[comparator]).to_numpy(float)
            draws = rng.choice(differences, size=(repetitions, len(differences)), replace=True).mean(axis=1)
            rows.append({"budget": budget, "reference": reference, "comparator": comparator,
                         "paired_mean_difference": float(differences.mean()),
                         "ci_low": float(np.quantile(draws, .025)),
                         "ci_high": float(np.quantile(draws, .975))})
    return pd.DataFrame(rows)


def run() -> None:
    config = json.loads(CONFIG.read_text())
    RAW.mkdir(parents=True, exist_ok=True); TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    bundle = np.load(DATA / "cycle_features.npz"); splits = np.load(DATA / "grouped_splits.npz")
    features = bundle["features"]; cases = pd.read_csv(CASES)
    rows = []
    for seed in config["seeds"]:
        fit = joblib.load(MODELS / f"seed_{seed}.joblib")
        train = splits[f"seed_{seed}_train"]; validation = splits[f"seed_{seed}_validation"]
        z_train = fit["scaler"].transform(features[train])
        z_validation = fit["scaler"].transform(features[validation])
        basis = fit["pca"].components_[-config["monitor_dimension"]:]
        train_residual = z_train @ basis.T
        validation_score = np.sum((z_validation @ basis.T) ** 2, axis=1)
        threshold = float(np.quantile(validation_score, .95))
        diagonal = np.maximum(np.var(train_residual, axis=0, ddof=1), config["covariance_floor"])
        full = LedoitWolf().fit(train_residual).covariance_
        full += config["covariance_floor"] * np.eye(len(full))
        sensor_means = features[train].reshape(-1, 17, 5)[..., 0]
        bias_scale = np.maximum(np.std(sensor_means, axis=0), 1e-8)
        physical_j = summary_parameter_jacobian(
            np.mean(features[train], axis=0, keepdims=True), bias_scale
        )[0]
        parameter_j = physical_j / fit["scaler"].scale_[:, None]
        monitor_j = basis @ parameter_j
        gradients = fit["linear"].coef_ @ parameter_j
        for case in cases[(cases.seed == seed) & (cases.partition == "evaluation")].itertuples():
            support = parse_support(case.support); columns = parameter_columns(support)
            task_index = TARGET_NAMES.index(case.task); gradient = gradients[task_index, columns]
            restricted = monitor_j[:, columns]
            standardized = restricted / np.sqrt(diagonal)[:, None]
            gamma_standardized = task_visibility_certificate(standardized, gradient).gamma
            gamma_diagonal = task_visibility_certificate(restricted, gradient, np.diag(diagonal)).gamma
            gamma_full = task_visibility_certificate(restricted, gradient, full).gamma
            z_state = fit["scaler"].transform(features[[int(case.state)]])[0]
            residual = basis @ z_state
            scalar_j = (2.0 * residual @ restricted)[None, :]
            scalar_gamma = task_visibility_certificate(scalar_j, gradient).gamma
            slack = max(threshold - float(residual @ residual), config["covariance_floor"])
            rows.append({
                "case_id": case.case_id, "seed": seed, "support": case.support,
                "task": case.task, "verified_failure": bool(case.verified_failure),
                "gamma_standardized_identity": gamma_standardized,
                "gamma_diagonal": gamma_diagonal,
                "gamma_shrinkage_full": gamma_full,
                "task_gradient": float(np.linalg.norm(gradient)),
                "alarm_slack": slack, "scalar_score_gamma": scalar_gamma,
                "scalar_alarm_margin": scalar_gamma / slack,
                "full_covariance_condition": float(np.linalg.cond(full)),
            })
    table = pd.DataFrame(rows); table.to_csv(RAW / "case_metrics.csv", index=False)
    metric_directions = {
        "gamma_standardized_identity": False, "gamma_diagonal": False,
        "gamma_shrinkage_full": False, "task_gradient": True,
        "alarm_slack": False, "scalar_alarm_margin": False,
    }
    rankings = rank_metrics(table, metric_directions, config["ranking_budgets"])
    intervals = paired_intervals(rankings, config["bootstrap_repetitions"], config["bootstrap_seed"])
    rankings.to_csv(RAW / "ranking_results.csv", index=False)
    intervals.to_csv(RAW / "paired_intervals.csv", index=False)
    summary = rankings.groupby(["metric", "budget"], as_index=False).agg(
        failures_median=("failures_found", "median"), failures_min=("failures_found", "min"),
        failures_max=("failures_found", "max"), precision_median=("precision", "median"),
    )
    summary.to_csv(TABLES / "table_metric_robustness.csv", index=False)
    equivalence = float(np.max(np.abs(table.gamma_standardized_identity - table.gamma_diagonal)))
    metadata = {"protocol": config, "cases": len(table),
                "standardized_identity_diagonal_max_difference": equivalence,
                "scalar_first_order_blind_fraction": float((table.scalar_score_gamma <= 1e-12).mean()),
                "full_covariance_condition_range": [float(table.full_covariance_condition.min()), float(table.full_covariance_condition.max())]}
    (RAW / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    selected = summary[summary.metric.isin(["gamma_standardized_identity", "gamma_shrinkage_full",
                                             "task_gradient", "alarm_slack", "scalar_alarm_margin"])]
    fig, axis = plt.subplots(figsize=(3.45, 2.45))
    for metric, group in selected.groupby("metric"):
        axis.plot(group.budget, group.failures_median, marker="o", label=metric.replace("gamma_", "").replace("_", " "))
    axis.set(xlabel="finite-audit budget", ylabel="median failures found")
    axis.grid(alpha=.2); axis.legend(frameon=False, fontsize=6)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_metric_robustness.pdf")
    fig.savefig(FIGURES / "fig_metric_robustness.png", dpi=220); plt.close(fig)
    print(summary.to_string(index=False)); print(intervals.to_string(index=False)); print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
