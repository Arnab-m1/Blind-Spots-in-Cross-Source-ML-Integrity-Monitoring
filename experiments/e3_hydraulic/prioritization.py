"""Held-out fixed-budget prioritization of finite Hydraulic failures."""

from __future__ import annotations

import hashlib
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
from joblib import Parallel, delayed
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import (
    SENSOR_SPECS, TARGET_NAMES, newly_reassuring_error,
    summary_parameter_jacobian, transform_summary_features,
)
from run import SEVERITY_SIGNS, TARGET_VALUES, nearest_class
from matched_security import matched_supports, parameter_columns, support_name, unit

DATA = ROOT / "data" / "hydraulic" / "processed"
MODELS = ROOT / "results" / "models" / "e3_hydraulic"
RAW = ROOT / "results" / "raw" / "e3_hydraulic"
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
CONFIG = ROOT / "configs" / "e3_hydraulic_prioritization.json"


def stable_partition(name: str, modulus: int, remainder: int) -> str:
    value = int(hashlib.sha256(name.encode("ascii")).hexdigest()[:8], 16)
    return "development" if value % modulus == remainder else "evaluation"


def ranks(values: np.ndarray, *, descending: bool) -> np.ndarray:
    order = np.argsort(-values if descending else values, kind="stable")
    result = np.empty(len(values), dtype=float)
    result[order] = np.linspace(1.0, 0.0, len(values), endpoint=True)
    return result


def finite_verify(context: dict, budget: float, config: dict) -> dict[str, object]:
    cols = context["cols"]
    gradient = context["gradient"]
    restricted = context["restricted"]
    starts = [np.zeros(len(cols)), -budget * unit(gradient)]
    try:
        starts.append(minimum_evidence_attack(restricted, gradient, -0.25))
    except (ArithmeticError, ValueError):
        starts.append(-unit(gradient))
    starts[-1] = budget * unit(starts[-1])

    def evaluate(local: np.ndarray) -> tuple[float, float, int]:
        theta = np.zeros(2 * len(SENSOR_SPECS))
        theta[cols] = local
        changed = transform_summary_features(
            context["physical"], theta[0::2], theta[1::2] * context["bias_scales"]
        )
        z = context["scaler"].transform(changed[None, :])
        prediction = float(context["model"].predict(z)[0, context["task_index"]])
        score = float(np.sum((z @ context["monitor"].T) ** 2))
        attacked_class = int(nearest_class(np.asarray([prediction]), context["classes"])[0])
        return prediction, score, attacked_class

    best = None
    for start_name, start in zip(config["optimizer"]["starts"], starts):
        result = minimize(
            lambda local: evaluate(local)[0],
            np.clip(start, -budget, budget),
            method=config["optimizer"]["method"],
            bounds=[(-min(1.0, budget), budget) if index % 2 == 0 else (-budget, budget)
                    for index in range(len(cols))],
            constraints=[
                {"type": "ineq", "fun": lambda local: budget**2 - float(local @ local)},
                {"type": "ineq", "fun": lambda local: context["threshold"] - evaluate(local)[1]},
                {"type": "ineq", "fun": lambda local: 1.0 + local[0::2]},
            ],
            options={"maxiter": config["optimizer"]["maxiter"], "ftol": config["optimizer"]["ftol"]},
        )
        prediction, score, attacked_class = evaluate(result.x)
        residual = max(
            float(result.x @ result.x - budget**2), score - context["threshold"],
            float(-np.min(1.0 + result.x[0::2])), 0.0,
        )
        feasible = bool(np.all(np.isfinite(result.x)) and residual <= config["optimizer"]["constraint_tolerance"])
        candidate = {
            "start": start_name, "solver_success": bool(result.success),
            "solver_status": int(result.status), "solver_message": str(result.message),
            "feasible": feasible, "constraint_residual": residual,
            "prediction": prediction, "monitor_score": score,
            "attacked_class": attacked_class, "perturbation_norm": float(np.linalg.norm(result.x)),
            "objective": prediction,
        }
        if feasible and (best is None or prediction < best["prediction"]):
            best = candidate
    if best is None:
        return {"start": "none", "solver_success": False, "solver_status": -1,
                "solver_message": "no feasible return", "feasible": False,
                "constraint_residual": np.nan, "prediction": np.nan,
                "monitor_score": np.nan, "attacked_class": -1,
                "perturbation_norm": np.nan, "objective": np.nan}
    return best


def timed_verify(case_id: str, seed: int, context: dict, budget: float, config: dict) -> tuple[dict, dict]:
    started = time.perf_counter()
    result = finite_verify(context, budget, config)
    runtime = {"seed": seed, "stage": "finite_verification", "seconds": time.perf_counter() - started}
    result.update({"case_id": case_id, "budget": budget})
    return result, runtime


def run() -> None:
    config = json.loads(CONFIG.read_text())
    bundle = np.load(DATA / "cycle_features.npz")
    splits = np.load(DATA / "grouped_splits.npz")
    features = bundle["features"]
    targets = bundle["targets"][:, :4].astype(float) * SEVERITY_SIGNS
    supports = matched_supports(config["support_pair_count"], config["support_sampling_seed"])
    case_rows, verification_rows, runtime_rows, pending = [], [], [], []
    rng = np.random.default_rng(config["state_sampling_seed"])

    for seed in config["seeds"]:
        fit = joblib.load(MODELS / f"seed_{seed}.joblib")
        train, validation, test = (splits[f"seed_{seed}_{part}"] for part in ("train", "validation", "test"))
        stage = time.perf_counter()
        scaler = fit["scaler"]
        z_validation, z_test = scaler.transform(features[validation]), scaler.transform(features[test])
        monitor = fit["pca"].components_[-config["monitor_dimension"]:]
        threshold = float(np.quantile(np.sum((z_validation @ monitor.T) ** 2, axis=1), config["validation_alarm_quantile"]))
        clean_scores = np.sum((z_test @ monitor.T) ** 2, axis=1)
        clean_predictions = fit["linear"].predict(z_test)
        true_z = fit["target_scaler"].transform(targets[test])
        bias_scales = np.maximum(np.std(features[train].reshape(-1, len(SENSOR_SPECS), 5)[..., 0], axis=0), 1e-8)
        parameter_jacobian = summary_parameter_jacobian(np.mean(features[train], axis=0, keepdims=True), bias_scales)[0] / scaler.scale_[:, None]
        monitor_parameters = monitor @ parameter_jacobian
        gradients = fit["linear"].coef_ @ parameter_jacobian
        runtime_rows.append({"seed": seed, "stage": "features_jacobians_gradients_whitening", "seconds": time.perf_counter() - stage})

        for support_index, support in enumerate(supports):
            name = support_name(support)
            partition = stable_partition(name, config["development_support_modulus"], config["development_support_remainder"])
            cols = parameter_columns(support)
            restricted = monitor_parameters[:, cols]
            singular = np.linalg.svd(restricted, compute_uv=False)
            sigma_min = float(singular[-1]) if len(singular) else 0.0
            for task_index, task in enumerate(TARGET_NAMES[:4]):
                severity = TARGET_VALUES[task_index] * SEVERITY_SIGNS[task_index]
                classes = (severity - fit["target_scaler"].mean_[task_index]) / fit["target_scaler"].scale_[task_index]
                clean_class = nearest_class(clean_predictions[:, task_index], classes)
                true_class = nearest_class(true_z[:, task_index], classes)
                eligible = np.flatnonzero((clean_class == true_class) & (clean_scores <= threshold))
                if not len(eligible):
                    continue
                pick_rng = np.random.default_rng(rng.integers(0, 2**32) + support_index * 101 + task_index)
                local_row = int(pick_rng.choice(eligible))
                gradient = gradients[task_index, cols]
                cert = task_visibility_certificate(restricted, gradient)
                case_id = f"{seed}:{int(test[local_row])}:{name}:{task}"
                case_rows.append({
                    "case_id": case_id, "seed": seed, "state": int(test[local_row]),
                    "support": name, "support_size": len(support), "task": task,
                    "partition": partition, "gamma": cert.gamma, "rank": cert.rank,
                    "nullity": cert.nullity, "h": cert.harmful_nullspace_score,
                    "gradient_norm": float(np.linalg.norm(gradient)), "sigma_min": sigma_min,
                    "clean_alarm_slack": threshold - float(clean_scores[local_row]),
                })
                context = {
                    "cols": cols, "gradient": gradient, "restricted": restricted,
                    "physical": features[test[local_row]], "bias_scales": bias_scales,
                    "scaler": scaler, "model": fit["linear"], "monitor": monitor,
                    "task_index": task_index, "classes": classes, "threshold": threshold,
                }
                for budget in config["verification_budgets"]:
                    pending.append((case_id, seed, context, float(budget),
                                    int(clean_class[local_row]), int(true_class[local_row]), threshold))

    completed = Parallel(n_jobs=-1, verbose=5)(
        delayed(timed_verify)(case_id, seed, context, budget, config)
        for case_id, seed, context, budget, _, _, _ in pending
    )
    for ((result, runtime), (_, _, _, _, clean_class, true_class, threshold)) in zip(completed, pending):
        runtime_rows.append(runtime)
        verified = bool(result["feasible"] and newly_reassuring_error(
            np.asarray([clean_class]), np.asarray([result["attacked_class"]]),
            np.asarray([true_class]),
        )[0] and result["monitor_score"] <= threshold + config["optimizer"]["constraint_tolerance"])
        verification_rows.append({"verified_failure": verified, **result})

    cases = pd.DataFrame(case_rows)
    verification = pd.DataFrame(verification_rows)
    labels = verification.groupby("case_id", as_index=False).agg(
        verified_failure=("verified_failure", "max"),
        feasible=("feasible", "max"), invalid_rate=("feasible", lambda x: 1.0 - float(np.mean(x))),
        verification_seconds=("case_id", lambda x: 0.0),
    )
    finite_times = pd.DataFrame(runtime_rows)
    per_case_seconds = finite_times.loc[finite_times.stage == "finite_verification", "seconds"].to_numpy()
    labels["verification_seconds"] = per_case_seconds.reshape(len(labels), -1).sum(axis=1)
    cases = cases.merge(labels, on="case_id")
    cases.to_csv(RAW / "prioritization_cases.csv", index=False)
    verification.to_csv(RAW / "prioritization_verification.csv", index=False)

    methods = {
        "random": ("random_score", True), "rank_nullity": ("rank_nullity_score", True),
        "harmful_kernel": ("h", True), "task_gradient": ("gradient_norm", True),
        "minimum_singular": ("sigma_min", False), "alarm_slack": ("clean_alarm_slack", False),
        "task_conditioned_margin": ("gamma", False),
    }
    cases["random_score"] = np.random.default_rng(2027).random(len(cases))
    cases["rank_nullity_score"] = cases["nullity"] * 100.0 - cases["rank"]
    feature_specs = [("gamma", False), ("gradient_norm", True), ("sigma_min", False), ("clean_alarm_slack", False)]
    for column, descending in feature_specs:
        cases[f"rank_{column}"] = cases.groupby("seed")[column].transform(lambda x: ranks(x.to_numpy(float), descending=descending))
    dev = cases[cases.partition == "development"]
    best_weights, best_ap = None, -1.0
    grid = config["combined_weight_grid"]
    for weights in itertools.product(grid, repeat=len(feature_specs)):
        if not any(weights):
            continue
        score = sum(weight * dev[f"rank_{column}"] for weight, (column, _) in zip(weights, feature_specs))
        order = np.argsort(-score.to_numpy(), kind="stable")
        y = dev["verified_failure"].to_numpy(float)[order]
        ap = float(np.sum(np.cumsum(y) / np.arange(1, len(y) + 1) * y) / max(np.sum(y), 1.0))
        if ap > best_ap:
            best_ap, best_weights = ap, weights
    cases["combined_score"] = sum(weight * cases[f"rank_{column}"] for weight, (column, _) in zip(best_weights, feature_specs))
    methods["combined_validation_selected"] = ("combined_score", True)

    evaluation = cases[cases.partition == "evaluation"].copy()
    curve_rows, metric_rows = [], []
    screening_seconds = float(finite_times.loc[finite_times.stage != "finite_verification", "seconds"].sum()) / max(len(cases), 1)
    for seed, seed_frame in evaluation.groupby("seed"):
        total_failures = int(seed_frame.verified_failure.sum())
        for method, (column, descending) in methods.items():
            ordered = seed_frame.sort_values(column, ascending=not descending, kind="stable").reset_index(drop=True)
            cumulative = ordered.verified_failure.astype(int).cumsum()
            wall = screening_seconds * np.arange(1, len(ordered) + 1) + ordered.verification_seconds.cumsum()
            for index, row in ordered.iterrows():
                curve_rows.append({"seed": seed, "method": method, "verification_count": index + 1,
                                   "cumulative_failures": int(cumulative.iloc[index]), "wall_seconds": float(wall.iloc[index])})
            for budget in config["ranking_budgets"]:
                k = min(int(budget), len(ordered))
                found = int(cumulative.iloc[k - 1])
                metric_rows.append({"seed": seed, "method": method, "budget": budget,
                                    "precision": found / k, "recall": found / max(total_failures, 1),
                                    "audc": float(np.trapezoid(cumulative.iloc[:k], dx=1) / max(k * max(total_failures, 1), 1)),
                                    "failures_found": found, "total_failures": total_failures,
                                    "wall_seconds": float(wall.iloc[k - 1])})
    curves, metrics = pd.DataFrame(curve_rows), pd.DataFrame(metric_rows)
    cases.to_csv(RAW / "prioritization_ranked_cases.csv", index=False)
    curves.to_csv(RAW / "prioritization_curves.csv", index=False)
    metrics.to_csv(RAW / "prioritization_metrics.csv", index=False)
    finite_times.to_csv(RAW / "prioritization_runtime.csv", index=False)
    summary = metrics.groupby(["method", "budget"], as_index=False).agg(
        precision_median=("precision", "median"), recall_median=("recall", "median"),
        audc_median=("audc", "median"), failures_median=("failures_found", "median"),
        wall_seconds_median=("wall_seconds", "median"),
    )
    summary.to_csv(TABLES / "table_hydraulic_prioritization.csv", index=False)
    selected = curves[curves.method.isin([
        "random", "task_gradient", "task_conditioned_margin",
        "combined_validation_selected",
    ])]
    plot_labels = {
        "random": "random", "task_gradient": "task gradient",
        "task_conditioned_margin": r"$\Gamma_M$",
        "combined_validation_selected": "combined",
    }
    figure, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for method, group in selected.groupby("method"):
        count_summary = group.groupby("verification_count").cumulative_failures.agg(["min", "median", "max"])
        axes[0].plot(count_summary.index, count_summary["median"], label=plot_labels[method])
        axes[0].fill_between(count_summary.index, count_summary["min"], count_summary["max"], alpha=.12)
        time_grid = np.linspace(0, selected.wall_seconds.max(), 100)
        traces = np.asarray([
            np.interp(time_grid, trace.wall_seconds, trace.cumulative_failures,
                      left=0, right=trace.cumulative_failures.iloc[-1])
            for _, trace in group.groupby("seed")
        ])
        axes[1].plot(time_grid, np.median(traces, axis=0), label=plot_labels[method])
        axes[1].fill_between(time_grid, np.min(traces, axis=0), np.max(traces, axis=0), alpha=.12)
    axes[0].set(xlabel="finite verifications", ylabel="verified joint failures", xlim=(0, 100))
    axes[1].set(xlabel="screening + verification time (s)", ylabel="verified joint failures", xlim=(0, 150))
    axes[0].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig_hydraulic_prioritization.pdf", bbox_inches="tight")
    figure.savefig(FIGURES / "fig_hydraulic_prioritization.png", dpi=220, bbox_inches="tight")
    plt.close(figure)
    metadata = {"config": config, "development_average_precision": best_ap,
                "combined_weights": dict((column, weight) for weight, (column, _) in zip(best_weights, feature_specs)),
                "cases": len(cases), "evaluation_cases": len(evaluation),
                "verified_failures": int(evaluation.verified_failure.sum()),
                "invalid_rate": float(1.0 - verification.feasible.mean())}
    (RAW / "prioritization_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    run()
