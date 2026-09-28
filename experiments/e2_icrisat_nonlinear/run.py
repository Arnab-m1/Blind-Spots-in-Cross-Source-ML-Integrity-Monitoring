"""Nonlinear ICRISAT certificates, attacks, and continuation audit."""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import (
    constrained_oracle_attack,
    continuation_attack,
    harmful_tangent_attack,
    minimum_evidence_attack,
)
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.model_adapters import mlp_regressor_value_and_jacobian

PANEL = ROOT / "data" / "icrisat" / "panel.csv"
RAW = ROOT / "results" / "raw" / "e2_icrisat_nonlinear"
FIGURES = ROOT / "results" / "figures"
FEATURES = ["rainfall", "temperature", "irrigated_area_1kha", "n_fertilizer_tons", "acreage"]
TARGET = "reported_yield"
SEED = 43
RTOL = 1e-10


def supports():
    for size in range(1, len(FEATURES) + 1):
        yield from itertools.combinations(range(len(FEATURES)), size)


def model_functions(linear, predictor, residual_weight, context, autoencoder):
    def task(x):
        design = np.concatenate([x, context])
        residual = mlp_regressor_value_and_jacobian(predictor, design)[0][0]
        return float(linear.predict(design[None, :])[0] + residual_weight * residual)

    def monitor(x):
        reconstruction = mlp_regressor_value_and_jacobian(autoencoder, x)[0]
        return x - reconstruction

    return task, monitor


def local_geometry(linear, predictor, residual_weight, autoencoder, x, context):
    _, task_jacobian = mlp_regressor_value_and_jacobian(
        predictor, np.concatenate([x, context])
    )
    _, reconstruction_jacobian = mlp_regressor_value_and_jacobian(autoencoder, x)
    gradient = linear.coef_[:x.size] + residual_weight * task_jacobian[0, :x.size]
    return np.eye(x.size) - reconstruction_jacobian, gradient


def run() -> None:
    started = time.perf_counter()
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL).dropna(subset=["year", TARGET, *FEATURES])
    train = panel[panel["year"] <= 2008]
    validation = panel[(panel["year"] >= 2009) & (panel["year"] <= 2010)]
    test = panel[panel["year"] >= 2011]
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
    residual_train = y_train - linear.predict(x_train)
    predictor = MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="tanh", alpha=1e-2,
        early_stopping=True, max_iter=1200, random_state=SEED,
    ).fit(x_train, residual_train)
    residual_validation = predictor.predict(x_validation)
    weights = (0.0, 0.25, 0.5, 0.75, 1.0)
    residual_weight = min(
        weights,
        key=lambda weight: np.mean((linear.predict(x_validation) + weight * residual_validation - y_validation) ** 2),
    )
    autoencoder = MLPRegressor(
        hidden_layer_sizes=(8, 3, 8), activation="tanh", alpha=1e-3,
        early_stopping=True, max_iter=1000, random_state=SEED,
    ).fit(z_train, z_train)
    prediction = linear.predict(x_test) + residual_weight * predictor.predict(x_test)
    validation_residual = z_validation - autoencoder.predict(z_validation)
    validation_score = np.sum(validation_residual**2, axis=1)
    threshold = float(np.quantile(validation_score, 0.95))
    clean_residual = z_test - autoencoder.predict(z_test)
    clean_score = np.sum(clean_residual**2, axis=1)

    certificate_rows = []
    for row_index, x in enumerate(z_test):
        jacobian, gradient = local_geometry(
            linear, predictor, residual_weight, autoencoder, x, c_test[row_index]
        )
        for support in supports():
            columns = np.asarray(support)
            certificate = task_visibility_certificate(
                jacobian[:, columns], gradient[columns], rtol=RTOL
            )
            certificate_rows.append({
                "row": row_index, "support": "+".join(FEATURES[i] for i in support),
                "support_size": len(support), "gamma": certificate.gamma,
                "h": certificate.harmful_nullspace_score, "rank": certificate.rank,
                "nullity": certificate.nullity, "is_blind": certificate.is_blind,
            })
    certificate_table = pd.DataFrame(certificate_rows)
    certificate_table.to_csv(RAW / "certificates.csv", index=False)

    # Fixed, predeclared audit subset for expensive nonlinear attacks.
    audit_rows = np.linspace(0, len(z_test) - 1, 40, dtype=int)
    attack_rows = []
    support = tuple(range(len(FEATURES)))
    for row_index in audit_rows:
        x = z_test[row_index]
        task, monitor = model_functions(
            linear, predictor, residual_weight, c_test[row_index], autoencoder
        )
        jacobian, gradient = local_geometry(
            linear, predictor, residual_weight, autoencoder, x, c_test[row_index]
        )
        certificate = task_visibility_certificate(jacobian, gradient, rtol=RTOL)
        directions = {}
        if certificate.is_blind:
            directions["tangent"] = harmful_tangent_attack(jacobian, gradient, 0.5)
        else:
            directions["task_conditioned"] = minimum_evidence_attack(
                jacobian, gradient, task_shift=0.25
            )
            norm = np.linalg.norm(directions["task_conditioned"])
            if norm > 0.5:
                directions["task_conditioned"] *= 0.5 / norm
        directions["continuation"] = continuation_attack(
            monitor, task, x, support, budget=0.5, steps=12
        )
        try:
            directions["oracle"] = constrained_oracle_attack(
                monitor, task, x, support, budget=0.5, monitor_budget=0.05
            )
        except RuntimeError:
            pass
        for name, attack in directions.items():
            attacked = x + attack
            attack_rows.append({
                "row": int(row_index), "attack": name, "gamma": certificate.gamma,
                "budget": float(np.linalg.norm(attack)),
                "task_shift": task(attacked) - task(x),
                "monitor_shift": float(np.linalg.norm(monitor(attacked) - monitor(x))),
                "detected": bool(np.sum(monitor(attacked) ** 2) > threshold),
            })
    attack_table = pd.DataFrame(attack_rows)
    attack_table.to_csv(RAW / "attacks.csv", index=False)
    best_harm = attack_table.groupby("row")["task_shift"].max()
    row_gamma = attack_table.groupby("row")["gamma"].first()
    correlation = spearmanr(row_gamma, best_harm).statistic

    metrics = {
        "seed": SEED, "residual_weight": residual_weight,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "clean": {"r2": float(r2_score(y_test, prediction)),
                  "mae": float(mean_absolute_error(y_test, prediction)),
                  "rmse": float(np.sqrt(mean_squared_error(y_test, prediction))),
                  "monitor_fpr": float(np.mean(clean_score > threshold))},
        "certificates": len(certificate_table),
        "blind_fraction": float(certificate_table["is_blind"].mean()),
        "gamma_harm_spearman": float(correlation),
        "runtime_seconds": time.perf_counter() - started,
    }
    (RAW / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    summary = certificate_table.groupby("support").agg(gamma=("gamma", "median"), blind=("is_blind", "mean")).reset_index()
    axes[0].scatter(summary["gamma"], summary["blind"], c=summary["gamma"], cmap="viridis")
    axes[0].set(xlabel=r"median $\Gamma_M$", ylabel="blind-row fraction", title="(a) Nonlinear support audit", xscale="symlog")
    for name, group in attack_table.groupby("attack"):
        axes[1].scatter(group["monitor_shift"], group["task_shift"], s=18, label=name, alpha=0.75)
    axes[1].set(xlabel="monitor displacement", ylabel="task displacement", title="(b) Nonlinear attack frontier")
    axes[1].legend(frameon=False, fontsize=7)
    for axis in axes: axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_icrisat_nonlinear.pdf")
    fig.savefig(FIGURES / "fig_icrisat_nonlinear.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
