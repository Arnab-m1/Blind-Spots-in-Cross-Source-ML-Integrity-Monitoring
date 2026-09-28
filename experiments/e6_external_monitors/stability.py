"""Four-split stability evaluation for faithful GDN and TranAD monitors."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from satml_monitoring.datasets.hydraulic import SENSOR_SPECS
from satml_monitoring.temporal_monitors import GDN, TranAD
from run import SEVERITY_SIGNS, evaluate_monitor, train_gdn, train_tranad

CONFIG = ROOT / "configs/e6_external_monitors.json"
DATA = ROOT / "data/hydraulic/processed"
MODELS = ROOT / "results/models/e6_external_monitors"
RAW = ROOT / "results/raw/e6_external_monitors"
TABLES = ROOT / "results/tables"
SEEDS = (17, 29, 43, 71)


def run() -> None:
    started = time.perf_counter(); config = json.loads(CONFIG.read_text())
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    sequence = np.load(DATA / "cycle_sequences.npz")["sequences"].astype(np.float32)
    split_data = np.load(DATA / "grouped_splits.npz")
    feature_data = np.load(DATA / "cycle_features.npz")
    summaries = feature_data["features"].astype(float); targets = feature_data["targets"].astype(float)
    severity_targets = targets[:, :4] * SEVERITY_SIGNS
    all_certificates, all_events, metadata = [], [], {}
    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        splits = {part: split_data[f"seed_{seed}_{part}"] for part in ("train", "validation", "test")}
        task_scaler = StandardScaler().fit(summaries[splits["train"]])
        target_scaler = StandardScaler().fit(severity_targets[splits["train"]])
        linear = Ridge(alpha=10.0).fit(task_scaler.transform(summaries[splits["train"]]),
                                       target_scaler.transform(severity_targets[splits["train"]]))
        fit = {"scaler": task_scaler, "target_scaler": target_scaler, "linear": linear}
        sequence_mean = sequence[splits["train"]].mean(axis=(0, 1), keepdims=True)
        sequence_scale = np.maximum(sequence[splits["train"]].std(axis=(0, 1), keepdims=True), 1e-6)
        z_train = (sequence[splits["train"]] - sequence_mean) / sequence_scale
        bias_scale = np.maximum(sequence[splits["train"]].mean(axis=1).std(axis=0), 1e-8)
        gcfg, tcfg = config["models"]["gdn"], config["models"]["tranad"]
        models = {
            "gdn": GDN(17, config["window"], gcfg["embedding_dim"], gcfg["hidden_dim"], gcfg["top_k"]),
            "tranad": TranAD(17, config["window"], tcfg["model_dim"], tcfg["heads"], tcfg["layers"]),
        }
        checkpoint_paths = {name: MODELS / f"{name}_seed_{seed}.pt" for name in models}
        if all(path.exists() for path in checkpoint_paths.values()):
            for name, model in models.items():
                model.load_state_dict(torch.load(checkpoint_paths[name], map_location="cpu"))
            losses = {name: "loaded_checkpoint" for name in models}
        else:
            losses = {"gdn": train_gdn(models["gdn"], z_train, config),
                      "tranad": train_tranad(models["tranad"], z_train, config)}
        metadata[str(seed)] = {"training_loss": losses}
        for name, model in models.items():
            torch.save(model.state_dict(), MODELS / f"{name}_seed_{seed}.pt")
            certificates, events, monitor_meta = evaluate_monitor(
                name, model, sequence, splits, config, sequence_mean, sequence_scale,
                bias_scale, fit, targets,
            )
            for row in certificates: row["seed"] = seed
            for row in events: row["seed"] = seed
            all_certificates.extend(certificates); all_events.extend(events)
            metadata[str(seed)][name] = monitor_meta
        pd.DataFrame([row for row in all_certificates if row["seed"] == seed]).to_csv(
            RAW / f"stability_certificates_seed_{seed}.csv", index=False)
        pd.DataFrame([row for row in all_events if row["seed"] == seed]).to_csv(
            RAW / f"stability_matched_security_seed_{seed}.csv", index=False)
    certificates = pd.DataFrame(all_certificates); events = pd.DataFrame(all_events)
    certificates.to_csv(RAW / "stability_certificates.csv", index=False)
    events.to_csv(RAW / "stability_matched_security.csv", index=False)
    selected = events[(events.attack == "A4_task_conditioned") & (events.budget == .5)].copy()
    rows = []
    for (seed, monitor), group in selected.groupby(["seed", "monitor"]):
        rows.append({"seed": seed, "monitor": monitor,
                     "gamma_joint_spearman": group.gamma.corr(group.joint_adverse_no_alarm_rate, method="spearman"),
                     "vulnerable_configurations": int((group.joint_adverse_no_alarm_rate > 0).sum()),
                     "test_alarm_rate": metadata[str(seed)][monitor]["test_alarm_rate"],
                     "validation_alarm_rate": metadata[str(seed)][monitor]["validation_alarm_rate"],
                     "blind_configurations": int(certificates[(certificates.seed == seed) & (certificates.monitor == monitor)].is_blind.sum())})
    summary = pd.DataFrame(rows); summary.to_csv(TABLES / "table_external_monitor_stability.csv", index=False)
    aggregate = summary.groupby("monitor", as_index=False).agg(
        spearman_min=("gamma_joint_spearman", "min"), spearman_median=("gamma_joint_spearman", "median"),
        spearman_max=("gamma_joint_spearman", "max"), vulnerable_min=("vulnerable_configurations", "min"),
        vulnerable_median=("vulnerable_configurations", "median"), vulnerable_max=("vulnerable_configurations", "max"),
        blind_total=("blind_configurations", "sum"))
    aggregate.to_csv(TABLES / "table_external_monitor_stability_aggregate.csv", index=False)
    (RAW / "stability_metadata.json").write_text(json.dumps({"seeds": list(SEEDS),
        "runtime_seconds": time.perf_counter() - started, "details": metadata}, indent=2) + "\n")
    print(summary.to_string(index=False)); print(aggregate.to_string(index=False))


if __name__ == "__main__":
    run()
