"""Exhaustive partial-support audit for the linear ICRISAT endpoint."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import (
    harmful_tangent_attack,
    minimum_evidence_attack,
    monitor_evidence,
    task_gradient_attack,
)
from satml_monitoring.certificates import task_visibility_certificate


PANEL = ROOT / "data" / "icrisat" / "panel.csv"
RAW = ROOT / "results" / "raw" / "e1_icrisat_linear"
FIGURES = ROOT / "results" / "figures"
FEATURES = [
    "rainfall",
    "temperature",
    "irrigated_area_1kha",
    "n_fertilizer_tons",
    "acreage",
]
TARGET = "reported_yield"
BUDGETS = (0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
SEED = 42
RANK_RTOL = 1e-10


def all_supports(dimension: int):
    for size in range(1, dimension + 1):
        yield from itertools.combinations(range(dimension), size)


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def run() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    panel = pd.read_csv(PANEL).dropna(subset=["district_id", "year", TARGET, *FEATURES])
    panel["district_id"] = panel["district_id"].astype(str).str.zfill(3)
    train = panel[panel["year"] <= 2008].copy()
    validation = panel[(panel["year"] >= 2009) & (panel["year"] <= 2010)].copy()
    test = panel[panel["year"] >= 2011].copy()

    scaler = StandardScaler().fit(train[FEATURES])
    target_scaler = StandardScaler().fit(train[[TARGET]])
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["district_id"]])

    def design(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        z = scaler.transform(rows[FEATURES])
        district = encoder.transform(rows[["district_id"]])
        year = ((rows["year"].to_numpy(dtype=float) - 2010.0) / 10.0)[:, None]
        return z, np.column_stack([z, year, district])

    z_train, x_train = design(train)
    z_validation, _ = design(validation)
    z_test, x_test = design(test)
    y_train = target_scaler.transform(train[[TARGET]]).ravel()
    y_test = target_scaler.transform(test[[TARGET]]).ravel()
    model = Ridge(alpha=10.0).fit(x_train, y_train)
    predictions = model.predict(x_test)
    task_gradient = np.asarray(model.coef_[: len(FEATURES)], dtype=float)

    covariance = np.cov(z_train, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    selected = np.argsort(eigenvalues)[:3]
    monitor = eigenvectors[:, selected].T
    residual_variances = np.maximum(eigenvalues[selected], 1e-12)
    validation_scores = np.sum((z_validation @ monitor.T) ** 2 / residual_variances, axis=1)
    threshold = float(np.quantile(validation_scores, 0.95))

    clean_scores = np.sum((z_test @ monitor.T) ** 2 / residual_variances, axis=1)
    clean_fpr = float(np.mean(clean_scores > threshold))
    rng = np.random.default_rng(SEED)
    nonnegative_columns = np.asarray([0, 2, 3, 4], dtype=int)
    robust_min = train[FEATURES].quantile(.01).to_numpy(dtype=float)
    robust_max = train[FEATURES].quantile(.99).to_numpy(dtype=float)
    support_rows: list[dict[str, object]] = []
    attack_rows: list[dict[str, object]] = []

    for support in all_supports(len(FEATURES)):
        indices = np.asarray(support, dtype=int)
        names = "+".join(FEATURES[index] for index in support)
        restricted_monitor = monitor[:, indices]
        restricted_gradient = task_gradient[indices]
        certificate = task_visibility_certificate(
            restricted_monitor,
            restricted_gradient,
            np.diag(residual_variances),
            rtol=RANK_RTOL,
        )
        support_rows.append(
            {
                "support": names,
                "support_size": len(support),
                "rank": certificate.rank,
                "nullity": certificate.nullity,
                "h": certificate.harmful_nullspace_score,
                "gamma": certificate.gamma,
                "is_blind": certificate.is_blind,
            }
        )

        _, _, vh = np.linalg.svd(
            np.diag(1.0 / np.sqrt(residual_variances)) @ restricted_monitor,
            full_matrices=True,
        )
        generic_stealth = vh[-1]
        directions: dict[str, np.ndarray] = {
            "A0_random": unit(rng.normal(size=len(support))),
            "A1_task_gradient": unit(restricted_gradient),
            "A2_generic_stealth": unit(generic_stealth),
        }
        if certificate.is_blind:
            directions["A3_harmful_tangent"] = harmful_tangent_attack(
                restricted_monitor, restricted_gradient, 1.0
            )
            directions["A5_continuation"] = directions["A3_harmful_tangent"]
        else:
            local = minimum_evidence_attack(
                restricted_monitor,
                restricted_gradient,
                task_shift=1.0,
                covariance=np.diag(residual_variances),
            )
            directions["A4_task_conditioned"] = unit(local)

        for attack_name, direction in directions.items():
            for budget in BUDGETS:
                local_shift = budget * direction
                shift = np.zeros(len(FEATURES))
                shift[indices] = local_shift
                attacked_scores = np.sum(
                    ((z_test + shift) @ monitor.T) ** 2 / residual_variances,
                    axis=1,
                )
                physical_shift = shift * scaler.scale_
                attacked_physical = test[FEATURES].to_numpy(dtype=float) + physical_shift
                train_min = train[FEATURES].min().to_numpy(dtype=float)
                train_max = train[FEATURES].max().to_numpy(dtype=float)
                inside = (attacked_physical >= train_min) & (attacked_physical <= train_max)
                inside_robust = (attacked_physical >= robust_min) & (attacked_physical <= robust_max)
                nonnegative = attacked_physical[:, nonnegative_columns] >= 0.0
                attack_rows.append(
                    {
                        "support": names,
                        "support_size": len(support),
                        "attack": attack_name,
                        "budget": budget,
                        "task_shift_standardized": float(task_gradient @ shift),
                        "monitor_evidence": monitor_evidence(
                            restricted_monitor,
                            local_shift,
                            np.diag(residual_variances),
                        ),
                        "detection_rate": float(np.mean(attacked_scores > threshold)),
                        "clean_fpr": clean_fpr,
                        "fraction_rows_in_training_range": float(np.mean(np.all(inside, axis=1))),
                        "fraction_rows_in_robust_1_99_range": float(np.mean(np.all(inside_robust, axis=1))),
                        "fraction_rows_nonnegative_physical_fields": float(np.mean(np.all(nonnegative, axis=1))),
                        "max_absolute_standardized_shift": float(np.max(np.abs(shift))),
                    }
                )

    support_table = pd.DataFrame(support_rows)
    attack_table = pd.DataFrame(attack_rows)
    support_table.to_csv(RAW / "support_certificates.csv", index=False)
    attack_table.to_csv(RAW / "attack_results.csv", index=False)

    variant_rows = []
    scalar_points = z_test[np.linspace(0, len(z_test) - 1, 100, dtype=int)]
    for dimension in (1, 2, 3, 5):
        selected = np.argsort(eigenvalues)[:dimension]
        variant_monitor = eigenvectors[:, selected].T
        variances = np.maximum(eigenvalues[selected], 1e-12)
        val_scores = np.sum((z_validation @ variant_monitor.T) ** 2 / variances, axis=1)
        test_scores = np.sum((z_test @ variant_monitor.T) ** 2 / variances, axis=1)
        variant_threshold = float(np.quantile(val_scores, 0.95))
        for support in all_supports(len(FEATURES)):
            indices = np.asarray(support, dtype=int)
            vector_certificate = task_visibility_certificate(
                variant_monitor[:, indices], task_gradient[indices], np.diag(variances)
            )
            scalar_blind = []
            scalar_gamma = []
            whitening = np.diag(1.0 / variances)
            for point in scalar_points:
                score_gradient = 2.0 * variant_monitor.T @ whitening @ variant_monitor @ point
                certificate = task_visibility_certificate(
                    score_gradient[indices][None, :], task_gradient[indices]
                )
                scalar_blind.append(certificate.is_blind)
                scalar_gamma.append(certificate.gamma)
            variant_rows.append({
                "dimension": dimension,
                "support": "+".join(FEATURES[index] for index in support),
                "support_size": len(support),
                "test_fpr": float(np.mean(test_scores > variant_threshold)),
                "vector_gamma": vector_certificate.gamma,
                "vector_is_blind": vector_certificate.is_blind,
                "scalar_blind_fraction": float(np.mean(scalar_blind)),
                "scalar_median_gamma": float(np.median(scalar_gamma)),
            })
    pd.DataFrame(variant_rows).to_csv(RAW / "monitor_variants.csv", index=False)
    sensitivity_rows = []
    for shrinkage in (0.0, 1e-6, 1e-4, 1e-2):
        covariance_variant = np.diag(residual_variances + shrinkage)
        for rtol in (1e-12, 1e-10, 1e-8, 1e-6):
            for support in all_supports(len(FEATURES)):
                indices = np.asarray(support, dtype=int)
                certificate = task_visibility_certificate(
                    monitor[:, indices], task_gradient[indices],
                    covariance_variant, rtol=rtol,
                )
                sensitivity_rows.append({
                    "shrinkage": shrinkage, "rank_rtol": rtol,
                    "support": "+".join(FEATURES[index] for index in support),
                    "gamma": certificate.gamma, "is_blind": certificate.is_blind,
                    "covariance_condition_number": float(np.linalg.cond(covariance_variant)),
                })
    pd.DataFrame(sensitivity_rows).to_csv(RAW / "numerical_sensitivity.csv", index=False)
    np.savez_compressed(
        RAW / "linear_system.npz",
        monitor=monitor,
        residual_variances=residual_variances,
        task_gradient=task_gradient,
        feature_scale=scaler.scale_,
        feature_mean=scaler.mean_,
        threshold=np.asarray([threshold]),
    )

    metrics = {
        "seed": SEED,
        "rank_rtol": RANK_RTOL,
        "features": FEATURES,
        "rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "years": {"train_end": 2008, "validation": [2009, 2010], "test_start": 2011},
        "clean": {
            "r2": float(r2_score(y_test, predictions)),
            "mae_standardized": float(mean_absolute_error(y_test, predictions)),
            "rmse_standardized": float(np.sqrt(mean_squared_error(y_test, predictions))),
            "monitor_fpr": clean_fpr,
            "validation_threshold": threshold,
        },
        "supports": len(support_table),
        "blind_supports": int(support_table["is_blind"].sum()),
    }
    (RAW / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ordered = support_table.sort_values(["support_size", "gamma"]).reset_index(drop=True)
    axes[0].scatter(
        np.arange(len(ordered)),
        ordered["gamma"],
        c=ordered["support_size"],
        cmap="viridis",
        s=32,
    )
    axes[0].set(
        xlabel="compromised support (ordered)",
        ylabel=r"task-conditioned margin $\Gamma_M$",
        title="(a) All 31 ICRISAT supports",
    )
    axes[0].set_yscale("symlog", linthresh=1e-4)
    axes[0].grid(alpha=0.2)

    merged = attack_table.merge(support_table[["support", "gamma"]], on="support")
    selected_attacks = merged[
        merged["attack"].isin(["A3_harmful_tangent", "A4_task_conditioned"])
        & np.isclose(merged["budget"], 1.0)
    ]
    axes[1].scatter(
        selected_attacks["gamma"],
        np.abs(selected_attacks["task_shift_standardized"]),
        c=selected_attacks["support_size"],
        cmap="viridis",
        s=36,
    )
    axes[1].set(
        xlabel=r"$\Gamma_M$",
        ylabel="absolute task shift at unit budget",
        title="(b) Certificate and attack headroom",
    )
    axes[1].set_xscale("symlog", linthresh=1e-4)
    axes[1].grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig_icrisat_supports.pdf")
    fig.savefig(FIGURES / "fig_icrisat_supports.png", dpi=220)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()
