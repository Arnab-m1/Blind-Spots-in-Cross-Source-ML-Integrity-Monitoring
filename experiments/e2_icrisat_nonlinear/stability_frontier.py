"""Five-seed nonlinear stability and local operational-frontier calibration."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.model_adapters import mlp_regressor_value_and_jacobian

from run import FEATURES, PANEL, TARGET, local_geometry, model_functions

RAW = ROOT / "results" / "raw" / "e2_icrisat_nonlinear"
FIGURES = ROOT / "results" / "figures"
SEEDS = (17, 29, 43, 71, 101)
BASE_SEED = 43
DELTAS = (0.01, 0.02, 0.05, 0.1)
MAX_RADIUS = 0.5


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def fit_endpoint(seed: int, train, validation, test):
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["district_id"]])
    scaler = StandardScaler().fit(train[FEATURES])
    target_scaler = StandardScaler().fit(train[[TARGET]])
    z_train = scaler.transform(train[FEATURES])
    z_validation = scaler.transform(validation[FEATURES])
    z_test = scaler.transform(test[FEATURES])
    y_train = target_scaler.transform(train[[TARGET]]).ravel()
    y_validation = target_scaler.transform(validation[[TARGET]]).ravel()
    y_test = target_scaler.transform(test[[TARGET]]).ravel()

    def context(rows):
        year = ((rows["year"].to_numpy(dtype=float) - 2010.0) / 10.0)[:, None]
        return np.column_stack([year, encoder.transform(rows[["district_id"]])])

    c_train, c_validation, c_test = context(train), context(validation), context(test)
    x_train = np.column_stack([z_train, c_train])
    x_validation = np.column_stack([z_validation, c_validation])
    x_test = np.column_stack([z_test, c_test])
    linear = Ridge(alpha=10.0).fit(x_train, y_train)
    predictor = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="tanh", alpha=1e-2,
        early_stopping=True, max_iter=1200, random_state=seed,
    ).fit(x_train, y_train - linear.predict(x_train))
    residual_validation = predictor.predict(x_validation)
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    residual_weight = min(
        weights,
        key=lambda weight: np.mean(
            (linear.predict(x_validation) + weight * residual_validation - y_validation) ** 2
        ),
    )
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(8, 3, 8), activation="tanh", alpha=1e-3,
        early_stopping=True, max_iter=1000, random_state=seed,
    ).fit(z_train, z_train)
    prediction = linear.predict(x_test) + residual_weight * predictor.predict(x_test)
    validation_residual = z_validation - autoencoder.predict(z_validation)
    threshold = float(np.quantile(np.sum(validation_residual**2, axis=1), 0.95))
    return {
        "linear": linear, "predictor": predictor, "autoencoder": autoencoder,
        "residual_weight": residual_weight, "z_test": z_test, "c_test": c_test,
        "y_test": y_test, "r2": float(r2_score(y_test, prediction)),
        "threshold": threshold,
    }


def frontier_value(task, monitor, x, initial, delta):
    clean_task = task(x)
    clean_monitor = np.asarray(monitor(x), dtype=float)

    def evidence(attack):
        return float(np.linalg.norm(np.asarray(monitor(x + attack)) - clean_monitor))

    result = minimize(
        evidence,
        np.asarray(initial, dtype=float),
        method="SLSQP",
        constraints=[
            {"type": "eq", "fun": lambda attack: task(x + attack) - clean_task - delta},
            {"type": "ineq", "fun": lambda attack: MAX_RADIUS - np.linalg.norm(attack)},
        ],
        options={"maxiter": 500, "ftol": 1e-12},
    )
    task_error = abs(task(x + result.x) - clean_task - delta)
    if not result.success or task_error > 1e-6:
        return None
    return evidence(result.x), float(np.linalg.norm(result.x))


def run() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL).dropna(subset=["year", TARGET, *FEATURES])
    train = panel[panel["year"] <= 2008]
    validation = panel[(panel["year"] >= 2009) & (panel["year"] <= 2010)]
    test = panel[panel["year"] >= 2011]
    audit_rows = np.linspace(0, len(test) - 1, 40, dtype=int)
    seed_rows: list[dict[str, object]] = []
    attack_rows: list[dict[str, object]] = []
    frontier_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        fit = fit_endpoint(seed, train, validation, test)
        seed_attacks = []
        for row_index in audit_rows:
            x = fit["z_test"][row_index]
            context = fit["c_test"][row_index]
            task, monitor = model_functions(
                fit["linear"], fit["predictor"], fit["residual_weight"], context,
                fit["autoencoder"],
            )
            jacobian, gradient = local_geometry(
                fit["linear"], fit["predictor"], fit["residual_weight"],
                fit["autoencoder"], x, context,
            )
            certificate = task_visibility_certificate(jacobian, gradient)
            _, _, vh = np.linalg.svd(jacobian, full_matrices=True)
            directions = {
                "A2_generic_stealth": 0.5 * unit(vh[-1]),
                "A4_task_conditioned": 0.5 * unit(
                    minimum_evidence_attack(jacobian, gradient, task_shift=0.25)
                ),
            }
            for attack_name, attack in directions.items():
                shift = float(task(x + attack) - task(x))
                evidence = float(np.linalg.norm(monitor(x + attack) - monitor(x)))
                detected = bool(np.sum(np.asarray(monitor(x + attack)) ** 2) > fit["threshold"])
                row = {
                    "seed": seed, "row": int(row_index), "attack": attack_name,
                    "gamma": certificate.gamma, "task_shift": shift,
                    "monitor_shift": evidence, "detected": detected,
                }
                attack_rows.append(row)
                seed_attacks.append(row)

            if seed == BASE_SEED:
                for delta in DELTAS:
                    initial = minimum_evidence_attack(jacobian, gradient, task_shift=delta)
                    value = frontier_value(task, monitor, x, initial, delta)
                    frontier_rows.append({
                        "seed": seed, "row": int(row_index), "delta": delta,
                        "gamma": certificate.gamma,
                        "frontier_evidence": np.nan if value is None else value[0],
                        "attack_norm": np.nan if value is None else value[1],
                        "ratio": np.nan if value is None else value[0] / abs(delta),
                        "success": value is not None,
                    })

        seed_frame = pd.DataFrame(seed_attacks)
        a4 = seed_frame[seed_frame.attack == "A4_task_conditioned"]
        correlation = float(spearmanr(a4.gamma, np.abs(a4.task_shift)).statistic)
        seed_rows.append({
            "seed": seed, "r2": fit["r2"], "residual_weight": fit["residual_weight"],
            "gamma_a4_harm_spearman": correlation,
            "a2_median_abs_harm": float(np.median(np.abs(seed_frame[seed_frame.attack == "A2_generic_stealth"].task_shift))),
            "a4_median_abs_harm": float(np.median(np.abs(a4.task_shift))),
            "a2_detection": float(seed_frame[seed_frame.attack == "A2_generic_stealth"].detected.mean()),
            "a4_detection": float(a4.detected.mean()),
        })

    stability = pd.DataFrame(seed_rows)
    attacks = pd.DataFrame(attack_rows)
    frontier = pd.DataFrame(frontier_rows)
    stability.to_csv(RAW / "seed_stability.csv", index=False)
    attacks.to_csv(RAW / "seed_stability_attacks.csv", index=False)
    frontier.to_csv(RAW / "frontier_calibration.csv", index=False)
    summary = {
        column: {"median": float(stability[column].median()),
                 "min": float(stability[column].min()),
                 "max": float(stability[column].max())}
        for column in stability.columns if column != "seed"
    }
    summary["frontier_success_rate"] = float(frontier.success.mean())
    (RAW / "stability_summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    successful = frontier[frontier.success]
    fig, axis = plt.subplots(figsize=(4.8, 3.8))
    for delta, group in successful.groupby("delta"):
        axis.scatter(group.gamma, group.ratio, s=24, alpha=.75, label=fr"$\Delta={delta:g}$")
    low = min(successful.gamma.min(), successful.ratio.min())
    high = max(successful.gamma.max(), successful.ratio.max())
    axis.plot([low, high], [low, high], "--", color="#c23b22", label="local prediction")
    axis.set(xlabel=r"certificate $\Gamma_M$", ylabel=r"nonlinear frontier $V_M(\Delta)/|\Delta|$", xscale="log", yscale="log")
    axis.grid(alpha=.2); axis.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_frontier_calibration.pdf")
    fig.savefig(FIGURES / "fig_frontier_calibration.png", dpi=220)
    plt.close(fig)
    print(stability.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    run()
