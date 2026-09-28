"""Create deterministic cycle features and leakage-audited grouped splits."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.datasets.hydraulic import (
    FEATURE_NAMES,
    SENSOR_SPECS,
    TARGET_NAMES,
    extract_cycle_features,
    grouped_splits,
    split_audit,
)


ARCHIVE = ROOT / "data" / "hydraulic" / "raw.zip"
PROCESSED = ROOT / "data" / "hydraulic" / "processed"
SEEDS = (17, 29, 43, 71, 101)


def run() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    features, means, targets = extract_cycle_features(str(ARCHIVE))
    np.savez_compressed(
        PROCESSED / "cycle_features.npz",
        features=features,
        sensor_means=means,
        targets=targets,
        sensor_names=np.asarray([sensor.name for sensor in SENSOR_SPECS]),
        feature_names=np.asarray(FEATURE_NAMES),
        target_names=np.asarray(TARGET_NAMES),
    )

    audits: dict[str, object] = {}
    split_arrays: dict[str, np.ndarray] = {}
    for seed in SEEDS:
        splits = grouped_splits(targets, seed=seed)
        audits[str(seed)] = split_audit(targets, splits)
        for name, indices in splits.items():
            split_arrays[f"seed_{seed}_{name}"] = indices
    np.savez_compressed(PROCESSED / "grouped_splits.npz", **split_arrays)

    inventory = {
        "cycles": int(features.shape[0]),
        "physical_sources": len(SENSOR_SPECS),
        "features_per_source": len(FEATURE_NAMES),
        "feature_columns": int(features.shape[1]),
        "mean_only_columns": int(means.shape[1]),
        "targets": list(TARGET_NAMES),
        "sensors": [asdict(sensor) for sensor in SENSOR_SPECS],
        "split_seeds": list(SEEDS),
        "split_audits": audits,
    }
    (PROCESSED / "inventory.json").write_text(
        json.dumps(inventory, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"wrote {features.shape[0]} cycles, {len(SENSOR_SPECS)} sources, "
        f"and {features.shape[1]} summary features to {PROCESSED}"
    )


if __name__ == "__main__":
    run()
