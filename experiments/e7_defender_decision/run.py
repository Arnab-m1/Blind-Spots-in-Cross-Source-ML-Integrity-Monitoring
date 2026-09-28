"""Select integrity anchors on development cases and test held-out finite risk."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments/e3_hydraulic"))

from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import SENSOR_SPECS, TARGET_NAMES, summary_parameter_jacobian
from matched_security import parameter_columns
from prioritization import finite_verify
from run import SEVERITY_SIGNS, TARGET_VALUES, nearest_class

DATA = ROOT / "data/hydraulic/processed"
MODELS = ROOT / "results/models/e3_hydraulic"
P0 = ROOT / "results/raw/e3_hydraulic"
RAW = ROOT / "results/raw/e7_defender_decision"
TABLES = ROOT / "results/tables"
FIGURES = ROOT / "results/figures"
CONFIG = ROOT / "configs/e7_defender_decision.json"
P0_CONFIG = ROOT / "configs/e3_hydraulic_prioritization.json"


def parse_support(name: str) -> tuple[int, ...]:
    lookup = {sensor.name: index for index, sensor in enumerate(SENSOR_SPECS)}
    return tuple(lookup[value] for value in name.split("+"))


def source_orders(seed: int, fit: dict, features: np.ndarray, train: np.ndarray,
                  cases: pd.DataFrame, monitor: np.ndarray, parameter_j: np.ndarray,
                  gradients: np.ndarray, max_budget: int, random_seed: int) -> dict[str, list[int]]:
    rng = np.random.default_rng(random_seed + seed)
    random_order = rng.permutation(len(SENSOR_SPECS)).tolist()
    conditioning = np.argsort(-np.linalg.norm(monitor.reshape(8, 17, 2), axis=(0, 2))).tolist()
    task_gradient = np.argsort(-np.max(np.abs(gradients.reshape(4, 17, 2)), axis=(0, 2))).tolist()

    def margin_metrics(protected: set[int]) -> tuple[int, float, float]:
        values = []
        for support_name, task in cases[["support", "task"]].drop_duplicates().itertuples(index=False):
            support = parse_support(support_name)
            residual = tuple(source for source in support if source not in protected)
            if not residual:
                continue
            cols = parameter_columns(residual)
            task_index = TARGET_NAMES.index(task)
            cert = task_visibility_certificate(monitor[:, cols], gradients[task_index, cols])
            values.append(cert.gamma)
        if not values:
            return (0, float("inf"), float("inf"))
        positive = [value for value in values if value > 1e-12]
        return (sum(value <= 1e-12 for value in values), min(positive, default=0.0), float(np.median(values)))

    local_order, selected = [], set()
    for _ in range(max_budget):
        candidates = []
        for source in set(range(17)) - selected:
            blind, minimum, median = margin_metrics(selected | {source})
            candidates.append(((-blind, minimum, median), -source, source))
        source = max(candidates)[2]
        selected.add(source); local_order.append(source)

    failures = cases[(cases.partition == "development") & cases.verified_failure].copy()
    finite_order, covered = [], set()
    for _ in range(max_budget):
        candidates = []
        for source in set(range(17)) - set(finite_order):
            newly = {
                row.case_id for row in failures.itertuples()
                if row.case_id not in covered and source in parse_support(row.support)
            }
            # Stable tie-break: prefer larger task gradient, then sensor index.
            utility = (len(newly), float(np.max(np.abs(gradients[:, 2 * source:2 * source + 2]))), -source)
            candidates.append((utility, source, newly))
        _, source, newly = max(candidates)
        finite_order.append(source); covered.update(newly)
    return {
        "random": random_order, "conditioning": conditioning,
        "task_gradient": task_gradient, "local_margin": local_order,
        "finite_audit_guided": finite_order,
    }


def build_context(seed: int, case: pd.Series, residual_support: tuple[int, ...],
                  fit: dict, features: np.ndarray, targets: np.ndarray,
                  splits, p0_config: dict) -> dict:
    train = splits[f"seed_{seed}_train"]
    validation = splits[f"seed_{seed}_validation"]
    scaler = fit["scaler"]
    monitor_basis = fit["pca"].components_[-p0_config["monitor_dimension"]:]
    threshold = float(np.quantile(
        np.sum((scaler.transform(features[validation]) @ monitor_basis.T) ** 2, axis=1),
        p0_config["validation_alarm_quantile"],
    ))
    bias_scales = np.maximum(np.std(features[train].reshape(-1, 17, 5)[..., 0], axis=0), 1e-8)
    physical_j = summary_parameter_jacobian(
        np.mean(features[train], axis=0, keepdims=True), bias_scales
    )[0]
    parameter_j = physical_j / scaler.scale_[:, None]
    monitor = monitor_basis @ parameter_j
    task_index = TARGET_NAMES.index(case.task)
    cols = parameter_columns(residual_support)
    severity = TARGET_VALUES[task_index] * SEVERITY_SIGNS[task_index]
    classes = (severity - fit["target_scaler"].mean_[task_index]) / fit["target_scaler"].scale_[task_index]
    return {
        "cols": cols, "gradient": (fit["linear"].coef_ @ parameter_j)[task_index, cols],
        "restricted": monitor[:, cols], "physical": features[int(case.state)],
        "bias_scales": bias_scales, "scaler": scaler, "model": fit["linear"],
        "monitor": monitor_basis, "task_index": task_index, "classes": classes,
        "threshold": threshold,
    }


def run() -> None:
    started = time.perf_counter()
    config = json.loads(CONFIG.read_text()); p0_config = json.loads(P0_CONFIG.read_text())
    RAW.mkdir(parents=True, exist_ok=True); TABLES.mkdir(exist_ok=True); FIGURES.mkdir(exist_ok=True)
    data = np.load(DATA / "cycle_features.npz"); splits = np.load(DATA / "grouped_splits.npz")
    features = data["features"]; targets = data["targets"][:, :4].astype(float) * SEVERITY_SIGNS
    cases = pd.read_csv(P0 / "prioritization_cases.csv")
    rows, selections, reruns = [], [], []
    max_budget = max(config["protection_budgets"])

    for seed, seed_cases in cases.groupby("seed"):
        seed = int(seed); fit = joblib.load(MODELS / f"seed_{seed}.joblib")
        train = splits[f"seed_{seed}_train"]
        bias = np.maximum(np.std(features[train].reshape(-1, 17, 5)[..., 0], axis=0), 1e-8)
        physical_j = summary_parameter_jacobian(np.mean(features[train], axis=0, keepdims=True), bias)[0]
        parameter_j = physical_j / fit["scaler"].scale_[:, None]
        monitor = fit["pca"].components_[-8:] @ parameter_j
        gradients = fit["linear"].coef_ @ parameter_j
        orders = source_orders(seed, fit, features, train, seed_cases, monitor,
                               parameter_j, gradients, max_budget, config["random_seed"])
        evaluation = seed_cases[seed_cases.partition == "evaluation"]
        original_failures = int(evaluation.verified_failure.sum())
        for method, order in orders.items():
            for budget in config["protection_budgets"]:
                protected = set(order[:budget])
                remaining = 0; optimized = 0
                for _, case in evaluation.iterrows():
                    if not bool(case.verified_failure):
                        continue
                    support = parse_support(case.support)
                    residual = tuple(source for source in support if source not in protected)
                    if residual == support:
                        remaining += 1
                        continue
                    if not residual:
                        reruns.append({"case_id": case.case_id, "seed": seed, "method": method,
                                      "protection_budget": budget, "residual_support": "",
                                      "verified_failure": False, "feasible": True,
                                      "status": "empty_residual_support"})
                        continue
                    context = build_context(seed, case, residual, fit, features, targets, splits, p0_config)
                    result = finite_verify(context, config["attack_budget"], p0_config)
                    clean_prediction = fit["linear"].predict(
                        fit["scaler"].transform(features[[int(case.state)]])
                    )[0, context["task_index"]]
                    true_prediction = fit["target_scaler"].transform(targets[[int(case.state)]])[0, context["task_index"]]
                    clean_class = nearest_class(np.asarray([clean_prediction]), context["classes"])[0]
                    true_class = nearest_class(np.asarray([true_prediction]), context["classes"])[0]
                    verified = bool(result["feasible"] and clean_class == true_class and
                                    result["attacked_class"] < true_class and
                                    result["monitor_score"] <= context["threshold"] + p0_config["optimizer"]["constraint_tolerance"])
                    remaining += int(verified); optimized += 1
                    reruns.append({"case_id": case.case_id, "seed": seed, "method": method,
                                  "protection_budget": budget,
                                  "residual_support": "+".join(SENSOR_SPECS[i].name for i in residual),
                                  "verified_failure": verified, "feasible": result["feasible"],
                                  "status": result["solver_message"]})
                names = "+".join(SENSOR_SPECS[index].name for index in sorted(protected))
                selections.append({"seed": seed, "method": method, "budget": budget,
                                   "protected_sources": names})
                rows.append({"seed": seed, "method": method, "budget": budget,
                             "protected_sources": names, "evaluation_cases": len(evaluation),
                             "original_failures": original_failures, "residual_failures": remaining,
                             "residual_joint_risk": remaining / len(evaluation),
                             "failure_reduction": original_failures - remaining,
                             "rerun_optimizations": optimized})
    results = pd.DataFrame(rows); reruns_frame = pd.DataFrame(reruns)
    results.to_csv(RAW / "heldout_hardening.csv", index=False)
    pd.DataFrame(selections).to_csv(RAW / "selected_sources.csv", index=False)
    reruns_frame.to_csv(RAW / "finite_reruns.csv", index=False)
    summary = results.groupby(["method", "budget"], as_index=False).agg(
        residual_failures_median=("residual_failures", "median"),
        residual_failures_min=("residual_failures", "min"),
        residual_failures_max=("residual_failures", "max"),
        residual_joint_risk_median=("residual_joint_risk", "median"),
        failure_reduction_median=("failure_reduction", "median"),
    )
    summary.to_csv(TABLES / "table_heldout_defender_decision.csv", index=False)
    fig, axis = plt.subplots(figsize=(3.45, 2.4))
    labels = {"random": "random", "conditioning": "conditioning",
              "task_gradient": "task gradient", "local_margin": "local margin",
              "finite_audit_guided": "finite-audit guided"}
    for method, group in results.groupby("method"):
        aggregate = group.groupby("budget").residual_failures.agg(["min", "median", "max"])
        axis.plot(aggregate.index, aggregate["median"], marker="o", label=labels[method])
        axis.fill_between(aggregate.index, aggregate["min"], aggregate["max"], alpha=.1)
    axis.set(xlabel="protected-source budget", ylabel="held-out verified failures",
             xticks=config["protection_budgets"])
    axis.grid(alpha=.2); axis.legend(frameon=False, fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_heldout_defender_decision.pdf")
    fig.savefig(FIGURES / "fig_heldout_defender_decision.png", dpi=220); plt.close(fig)
    metadata = {"protocol": config, "p0_label_protocol": p0_config["protocol_version"],
                "runtime_seconds": time.perf_counter() - started,
                "finite_reruns": len(reruns_frame),
                "invalid_rerun_rate": float(1.0 - reruns_frame.feasible.mean()) if len(reruns_frame) else 0.0}
    (RAW / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(summary.to_string(index=False)); print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
