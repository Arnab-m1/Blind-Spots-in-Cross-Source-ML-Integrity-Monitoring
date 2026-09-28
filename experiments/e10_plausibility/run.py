"""Summarize physical plausibility and record concrete attacker actors."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/raw/e1_icrisat_linear/attack_results.csv"
RAW = ROOT / "results/raw/e10_plausibility"
TABLES = ROOT / "results/tables"


def run() -> None:
    RAW.mkdir(parents=True, exist_ok=True); TABLES.mkdir(exist_ok=True)
    attacks = pd.read_csv(SOURCE)
    selected = attacks[attacks.attack.isin(["A3_harmful_tangent", "A4_task_conditioned"])]
    summary = selected.groupby(["attack", "budget"], as_index=False).agg(
        configurations=("support", "size"),
        nonnegative_median=("fraction_rows_nonnegative_physical_fields", "median"),
        nonnegative_min=("fraction_rows_nonnegative_physical_fields", "min"),
        training_range_median=("fraction_rows_in_training_range", "median"),
        training_range_min=("fraction_rows_in_training_range", "min"),
        robust_range_median=("fraction_rows_in_robust_1_99_range", "median"),
        robust_range_min=("fraction_rows_in_robust_1_99_range", "min"),
    )
    summary.to_csv(TABLES / "table_icrisat_plausibility.csv", index=False)
    selected.to_csv(RAW / "support_budget_plausibility.csv", index=False)
    actors = {
        "icrisat": {
            "actor": "a compromised data integration or reporting path controlling a declared subset of district-level source fields before model ingestion",
            "scope": "record manipulation benchmark; no claim of physical control over weather or agricultural processes",
            "constraints": "non-negativity for rainfall, irrigated area, fertilizer, and acreage; training min/max and 1st--99th percentile diagnostics",
        },
        "hydraulic": {
            "actor": "a compromised telemetry acquisition, calibration, or gateway path applying gain/bias changes to selected channels before monitoring and prediction",
            "scope": "sensor-report manipulation; no claim of changing the physical rig state",
            "derived_channels": "CE, CP, and SE are separate telemetry and change only when explicitly compromised",
        },
    }
    (RAW / "actor_and_constraints.json").write_text(json.dumps(actors, indent=2) + "\n")
    print(summary.to_string(index=False)); print(json.dumps(actors, indent=2))


if __name__ == "__main__":
    run()
