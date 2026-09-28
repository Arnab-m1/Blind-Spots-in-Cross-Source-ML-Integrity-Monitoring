"""Aggregate independently checkpointed external-monitor split results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/e6_external_monitors"
TABLES = ROOT / "results/tables"
SEEDS = (17, 29, 43, 71)


def run() -> None:
    certificate_frames = []
    event_frames = []
    for seed in SEEDS:
        certificate_frames.append(pd.read_csv(RAW / f"stability_certificates_seed_{seed}.csv"))
        event_frames.append(pd.read_csv(RAW / f"stability_matched_security_seed_{seed}.csv"))
    certificates = pd.concat(certificate_frames, ignore_index=True)
    events = pd.concat(event_frames, ignore_index=True)
    certificates.to_csv(RAW / "stability_certificates.csv", index=False)
    events.to_csv(RAW / "stability_matched_security.csv", index=False)

    selected = events[(events.attack == "A4_task_conditioned") & (events.budget == .5)]
    rows = []
    for (seed, monitor), group in selected.groupby(["seed", "monitor"]):
        rows.append({
            "seed": seed,
            "monitor": monitor,
            "gamma_joint_spearman": group.gamma.corr(
                group.joint_adverse_no_alarm_rate, method="spearman"
            ),
            "vulnerable_configurations": int(
                (group.joint_adverse_no_alarm_rate > 0).sum()
            ),
            "blind_configurations": int(
                certificates[
                    (certificates.seed == seed) & (certificates.monitor == monitor)
                ].is_blind.sum()
            ),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES / "table_external_monitor_stability.csv", index=False)
    aggregate = summary.groupby("monitor", as_index=False).agg(
        spearman_min=("gamma_joint_spearman", "min"),
        spearman_median=("gamma_joint_spearman", "median"),
        spearman_max=("gamma_joint_spearman", "max"),
        vulnerable_min=("vulnerable_configurations", "min"),
        vulnerable_median=("vulnerable_configurations", "median"),
        vulnerable_max=("vulnerable_configurations", "max"),
        blind_total=("blind_configurations", "sum"),
    )
    aggregate.to_csv(TABLES / "table_external_monitor_stability_aggregate.csv", index=False)
    (RAW / "stability_metadata.json").write_text(
        json.dumps({"seeds": list(SEEDS), "independent_split_count": len(SEEDS)}, indent=2)
        + "\n"
    )
    print(summary.to_string(index=False))
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    run()
