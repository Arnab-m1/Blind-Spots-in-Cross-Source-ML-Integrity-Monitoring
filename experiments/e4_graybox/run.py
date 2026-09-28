"""Hydraulic white-box and surrogate-transfer evaluation."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from satml_monitoring.attacks import harmful_tangent_attack, minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate
from satml_monitoring.datasets.hydraulic import SENSOR_SPECS, TARGET_NAMES, summary_parameter_jacobian

DATA = ROOT / "data" / "hydraulic" / "processed"
MODELS = ROOT / "results" / "models" / "e3_hydraulic"
RAW = ROOT / "results" / "raw" / "e4_graybox"
FIGURES = ROOT / "results" / "figures"
SEED = 1701
BUDGET = 0.5
FRACTIONS = (0.10, 0.25, 0.50, 1.00)


def block_columns(support):
    return np.concatenate([np.arange(i * 2, (i + 1) * 2) for i in support])


def direction(jacobian, gradient):
    cert = task_visibility_certificate(jacobian, gradient)
    if cert.is_blind:
        return harmful_tangent_attack(jacobian, gradient, BUDGET), cert
    attack = minimum_evidence_attack(jacobian, gradient, task_shift=0.25)
    norm = np.linalg.norm(attack)
    return (attack if norm <= BUDGET else attack * BUDGET / norm), cert


def run():
    RAW.mkdir(parents=True, exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA / "cycle_features.npz"); splits = np.load(DATA / "grouped_splits.npz")
    features = data["features"]; targets = data["targets"][:, :4].astype(float)
    signs = np.array([-1.0, -1.0, 1.0, -1.0]); severity = targets * signs
    defender = joblib.load(MODELS / "seed_17.joblib")
    train_idx = splits["seed_17_train"]; test_idx = splits["seed_17_test"]
    z_train = defender["scaler"].transform(features[train_idx])
    sensor_means = features[train_idx].reshape(-1, len(SENSOR_SPECS), 5)[..., 0]
    bias_scales = np.maximum(np.std(sensor_means, axis=0), 1e-8)
    physical_j = summary_parameter_jacobian(
        np.mean(features[train_idx], axis=0, keepdims=True), bias_scales
    )[0]
    defender_feature_j = physical_j / defender["scaler"].scale_[:, None]
    defender_j = defender["monitor"] @ defender_feature_j
    defender_u = defender["linear"].coef_ @ defender_feature_j
    rng = np.random.default_rng(SEED)
    support_pool = list(itertools.combinations(range(len(SENSOR_SPECS)), 2))
    support_pool = [support_pool[i] for i in rng.choice(len(support_pool), 60, replace=False)]

    surrogates = [("WB", defender_j, defender_u)]
    for fraction in FRACTIONS:
        count = max(100, int(fraction * len(train_idx)))
        chosen = rng.choice(len(train_idx), count, replace=False)
        y_train = defender["target_scaler"].transform(severity[train_idx])
        ridge = Ridge(alpha=10.0).fit(z_train[chosen], y_train[chosen])
        pca = PCA().fit(z_train[chosen])
        surrogates.append((f"GB_data_{fraction:.2f}", pca.components_[-8:] @ defender_feature_j, ridge.coef_ @ defender_feature_j))
    independent = joblib.load(MODELS / "seed_29.joblib")
    independent_feature_j = physical_j / independent["scaler"].scale_[:, None]
    surrogates.append(("GB_independent", independent["monitor"] @ independent_feature_j, independent["linear"].coef_ @ independent_feature_j))
    # Architecture mismatch: full-coordinate residual monitor and shared MLP's
    # first-order average gradient approximated by finite differences.
    mismatch_feature_gradient = np.zeros((4, 85))
    audit_points = defender["scaler"].transform(features[test_idx[:40]])
    for task in range(4):
        for column in range(85):
            offset = np.zeros(85); offset[column] = 1e-4
            plus = independent["nonlinear"].predict(audit_points + offset)[:, task]
            minus = independent["nonlinear"].predict(audit_points - offset)[:, task]
            mismatch_feature_gradient[task, column] = np.mean((plus - minus) / 2e-4)
    mismatch_u = mismatch_feature_gradient @ independent_feature_j
    surrogates.append(("GB_arch_mismatch", independent_feature_j, mismatch_u))

    rows = []
    for setting, surrogate_j, surrogate_u in surrogates:
        for support in support_pool:
            cols = block_columns(support)
            j_hat = surrogate_j[:, cols]; j_true = defender_j[:, cols]
            # Compare row spaces through projectors when monitor output sizes differ.
            p_hat = np.linalg.pinv(j_hat) @ j_hat
            p_true = np.linalg.pinv(j_true) @ j_true
            epsilon_j = float(np.linalg.norm(p_true - p_hat, 2))
            for task_index, task_name in enumerate(TARGET_NAMES[:4]):
                u_hat = surrogate_u[task_index, cols]; u_true = defender_u[task_index, cols]
                epsilon_f = float(np.linalg.norm(u_true - u_hat))
                proposed, cert = direction(j_hat, u_hat)
                _, _, vh = np.linalg.svd(j_hat, full_matrices=True)
                baselines = {
                    "proposed": proposed,
                    "random": BUDGET * rng.normal(size=len(cols)),
                    "task_only": BUDGET * u_hat / max(np.linalg.norm(u_hat), 1e-12),
                    "generic_stealth": BUDGET * vh[-1] / max(np.linalg.norm(vh[-1]), 1e-12),
                }
                baselines["random"] *= 1 / max(np.linalg.norm(baselines["random"]), 1e-12)
                for attack_name, attack in baselines.items():
                    leakage = float(np.linalg.norm(j_true @ attack))
                    harm = float(u_true @ attack)
                    rows.append({
                        "setting": setting, "support": "+".join(SENSOR_SPECS[i].name for i in support),
                        "task": task_name, "attack": attack_name, "epsilon_j": epsilon_j,
                        "epsilon_f": epsilon_f, "surrogate_gamma": cert.gamma,
                        "budget": float(np.linalg.norm(attack)), "defender_leakage": leakage,
                        "defender_task_harm": harm, "linear_bound": BUDGET * epsilon_j,
                        "bound_holds": leakage <= BUDGET * epsilon_j + 1e-8 if cert.is_blind and attack_name == "proposed" else None,
                    })
    table = pd.DataFrame(rows); table.to_csv(RAW / "graybox_results.csv", index=False)
    proposed = table[table["attack"] == "proposed"]
    summary = proposed.groupby("setting").agg(
        median_leakage=("defender_leakage", "median"), median_harm=("defender_task_harm", lambda x: float(np.median(np.abs(x)))),
        median_epsilon_j=("epsilon_j", "median"), median_epsilon_f=("epsilon_f", "median"),
        transfer_failure_rate=("defender_task_harm", lambda x: float(np.mean(np.abs(x) < 1e-3))),
    ).reset_index()
    summary.to_csv(RAW / "graybox_summary.csv", index=False)
    (RAW / "metadata.json").write_text(json.dumps({"seed": SEED, "budget": BUDGET, "supports": len(support_pool)}, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    axes[0].scatter(proposed["epsilon_j"], proposed["defender_leakage"], s=10, alpha=.4)
    limit = max(proposed["epsilon_j"].max(), proposed["defender_leakage"].max() / BUDGET)
    axes[0].plot([0, limit], [0, BUDGET * limit], "--", color="#c23b22", label=r"first-order mismatch term $\rho\epsilon_J$")
    axes[0].set(xlabel=r"surrogate mismatch $\epsilon_J$", ylabel="defender monitor leakage", title="(a) Transfer leakage"); axes[0].legend(frameon=False)
    order = summary.sort_values("median_epsilon_j")
    axes[1].plot(range(len(order)), order["median_harm"], marker="o", color="#0072B2")
    axes[1].set_xticks(range(len(order)), order["setting"], rotation=35, ha="right", fontsize=7)
    axes[1].set(ylabel="median absolute task harm", title="(b) Surrogate transfer")
    for axis in axes: axis.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(FIGURES / "fig_graybox_transfer.pdf"); fig.savefig(FIGURES / "fig_graybox_transfer.png", dpi=220); plt.close(fig)
    print(summary.to_string(index=False))


if __name__ == "__main__": run()
