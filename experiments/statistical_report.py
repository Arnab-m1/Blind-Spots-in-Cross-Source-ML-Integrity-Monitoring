"""Bootstrap intervals and paper-facing summary tables."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"
SEED = 2027
BOOTSTRAPS = 2000


def cluster_bootstrap_spearman(frame, cluster, rng):
    values = []
    clusters = frame[cluster].drop_duplicates().to_numpy()
    for _ in range(BOOTSTRAPS):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        pieces = [frame[frame[cluster] == value] for value in sampled]
        bootstrap = pd.concat(pieces, ignore_index=True)
        value = spearmanr(bootstrap.gamma, bootstrap.harm).statistic
        if np.isfinite(value): values.append(value)
    return float(spearmanr(frame.gamma, frame.harm).statistic), [float(v) for v in np.quantile(values, [0.025, 0.975])]


def run():
    TABLES.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(SEED)
    nonlinear = pd.read_csv(RAW / "e2_icrisat_nonlinear" / "attacks.csv")
    n_group = nonlinear.groupby("row").agg(gamma=("gamma", "first"), harm=("task_shift", lambda x: float(np.max(np.abs(x))))).reset_index()
    hydraulic = pd.read_csv(RAW / "e3_hydraulic" / "attack_results.csv")
    h_conditioned = hydraulic[hydraulic["attack"] == "A4_task_conditioned"]
    h_group = h_conditioned.groupby(["support", "task"]).agg(gamma=("gamma", "first"), harm=("task_shift", lambda x: float(np.max(np.abs(x))))).reset_index()
    intervals = {
        "seed": SEED, "bootstrap_repetitions": BOOTSTRAPS,
        "resampling_unit": {"icrisat": "test row", "hydraulic": "sensor support"},
        "interpretation": "design-based audit uncertainty for fixed fitted models; not population confidence intervals",
        "icrisat_gamma_harm_spearman": cluster_bootstrap_spearman(n_group, "row", rng),
        "hydraulic_gamma_harm_spearman": cluster_bootstrap_spearman(h_group, "support", rng),
    }
    (TABLES / "confidence_intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")

    i_metrics = json.loads((RAW / "e1_icrisat_linear" / "metrics.json").read_text())
    h_metrics = json.loads((RAW / "e3_hydraulic" / "summary.json").read_text())
    nonlinear_stability = pd.read_csv(RAW / "e2_icrisat_nonlinear" / "seed_stability.csv")
    clean = pd.DataFrame([
        {"dataset": "ICRISAT", "sources": 5, "task_model": "ridge", "clean_quality": i_metrics["clean"]["r2"], "metric": "R2", "test_fpr": i_metrics["clean"]["monitor_fpr"]},
        {"dataset": "ICRISAT", "sources": 5, "task_model": "contextual nonlinear", "clean_quality": nonlinear_stability["r2"].median(), "metric": "R2", "test_fpr": np.nan},
        {"dataset": "Hydraulic", "sources": 17, "task_model": "ridge multi-output", "clean_quality": h_metrics["linear_balanced_accuracy_mean"], "metric": "balanced accuracy", "test_fpr": h_metrics["monitor_test_fpr_mean"]},
        {"dataset": "Hydraulic", "sources": 17, "task_model": "shared MLP", "clean_quality": h_metrics["mlp_balanced_accuracy_mean"], "metric": "balanced accuracy", "test_fpr": np.nan},
    ])
    clean.to_csv(TABLES / "table_clean_models.csv", index=False)
    clean.drop(columns=["test_fpr"]).to_csv(TABLES / "table_task_models.csv", index=False)
    compression = pd.read_csv(RAW / "e3_hydraulic" / "compression_ablation.csv")
    monitors = pd.DataFrame([
        {"monitor": "ICRISAT PCA-3", "test_fpr": i_metrics["clean"]["monitor_fpr"], "ordinary_corruption_detection": np.nan},
        {"monitor": "Hydraulic PCA-8", **compression[compression.dimension == 8][["test_fpr", "ordinary_corruption_detection"]].iloc[0].to_dict()},
        {"monitor": "Hydraulic PCA-85", **compression[compression.dimension == 85][["test_fpr", "ordinary_corruption_detection"]].iloc[0].to_dict()},
    ])
    monitors.to_csv(TABLES / "table_monitor_quality.csv", index=False)
    icrisat_stability = pd.read_csv(RAW / "e2_icrisat_nonlinear" / "seed_stability.csv")
    icrisat_stability.to_csv(TABLES / "table_icrisat_seed_stability.csv", index=False)
    hydraulic_stability = pd.read_csv(RAW / "e3_hydraulic" / "seed_stability_summary.csv")
    hydraulic_stability.to_csv(TABLES / "table_hydraulic_seed_stability.csv", index=False)
    task_comparison = pd.read_csv(RAW / "e3_hydraulic" / "task_metric_example.csv")
    task_comparison.to_csv(TABLES / "table_task_metric_comparison.csv", index=False)
    hardening = pd.read_csv(RAW / "e5_hardening" / "hardening_results.csv")
    hardening[hardening.budget.isin([0,1,2])].to_csv(TABLES / "table_hardening.csv", index=False)
    hydraulic_hardening = pd.read_csv(RAW / "e5_hardening" / "hydraulic_hardening_results.csv")
    hydraulic_hardening.to_csv(TABLES / "table_hydraulic_hardening.csv", index=False)
    gray = pd.read_csv(RAW / "e4_graybox" / "graybox_summary.csv")
    gray.to_csv(TABLES / "table_graybox.csv", index=False)

    distributions = []
    for dataset, frame, columns in (
        ("ICRISAT nonlinear", n_group, ("gamma", "harm")),
        ("Hydraulic", h_group, ("gamma", "harm")),
    ):
        for metric in columns:
            values = frame[metric].to_numpy(dtype=float)
            q1, median, q3 = np.quantile(values, [0.25, 0.5, 0.75])
            distributions.append({
                "dataset": dataset,
                "metric": metric,
                "n": len(values),
                "q1": float(q1),
                "median": float(median),
                "q3": float(q3),
                "iqr": float(q3 - q1),
            })
    pd.DataFrame(distributions).to_csv(
        TABLES / "table_vulnerability_distributions.csv", index=False
    )
    print(json.dumps(intervals, indent=2))


if __name__ == "__main__": run()
