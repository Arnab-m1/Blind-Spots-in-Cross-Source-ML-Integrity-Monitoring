"""Generate exact and higher-order synthetic verification results."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from satml_monitoring.attacks import minimum_evidence_attack
from satml_monitoring.certificates import task_visibility_certificate


SEED = 20270916
ALPHAS = (0.20, 0.25, 0.30, 0.40, 0.55)
HORIZONS = (32, 64, 128, 256, 512, 1024, 2048, 4096)
REPETITIONS = 1000
OUTPUT = ROOT / "results" / "raw" / "e0_synthetic"


def exact_error_sum(separation: float) -> float:
    return 2.0 * NormalDist().cdf(-abs(separation) / 2.0)


def empirical_error_sum(
    rng: np.random.Generator,
    mean_shift: float,
    horizon: int,
    repetitions: int,
) -> tuple[float, float]:
    standard_error = 1.0 / math.sqrt(horizon)
    threshold = mean_shift / 2.0
    clean_means = rng.normal(0.0, standard_error, size=repetitions)
    attacked_means = rng.normal(mean_shift, standard_error, size=repetitions)
    if mean_shift >= 0:
        type_one = float(np.mean(clean_means > threshold))
        type_two = float(np.mean(attacked_means <= threshold))
    else:
        type_one = float(np.mean(clean_means < threshold))
        type_two = float(np.mean(attacked_means >= threshold))
    return type_one, type_two


def blind_manifold_record() -> dict[str, float | bool]:
    jacobian = np.array([[1.0, -1.0]])
    gradient = np.array([3.0, 2.0])
    certificate = task_visibility_certificate(jacobian, gradient)
    attack = minimum_evidence_attack(jacobian, gradient, task_shift=5.0)
    return {
        "gamma": certificate.gamma,
        "harmful_nullspace_score": certificate.harmful_nullspace_score,
        "attack_x1": float(attack[0]),
        "attack_x2": float(attack[1]),
        "monitor_shift": float(np.linalg.norm(jacobian @ attack)),
        "task_shift": float(gradient @ attack),
        "exactly_blind": bool(np.linalg.norm(jacobian @ attack) <= 1e-12),
    }


def run() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []
    for alpha in ALPHAS:
        for horizon in HORIZONS:
            delta = horizon ** (-alpha)
            for observer, mean_shift in (
                ("raw", delta),
                ("quadratic_monitor", delta**2),
            ):
                separation = math.sqrt(horizon) * mean_shift
                type_one, type_two = empirical_error_sum(
                    rng, mean_shift, horizon, REPETITIONS
                )
                standard_error = math.sqrt(
                    type_one * (1.0 - type_one) / REPETITIONS
                    + type_two * (1.0 - type_two) / REPETITIONS
                )
                rows.append(
                    {
                        "alpha": alpha,
                        "horizon": horizon,
                        "delta": delta,
                        "observer": observer,
                        "mean_shift": mean_shift,
                        "separation": separation,
                        "theoretical_error_sum": exact_error_sum(separation),
                        "empirical_type_one": type_one,
                        "empirical_type_two": type_two,
                        "empirical_error_sum": type_one + type_two,
                        "mc_ci95_low": max(0.0, type_one + type_two - 1.96 * standard_error),
                        "mc_ci95_high": min(2.0, type_one + type_two + 1.96 * standard_error),
                    }
                )

    with (OUTPUT / "phase_diagram.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "seed": SEED,
        "repetitions": REPETITIONS,
        "alphas": ALPHAS,
        "horizons": HORIZONS,
        "blind_manifold": blind_manifold_record(),
        "model": {
            "raw_mean_shift": "delta",
            "monitor_mean_shift": "delta^2",
            "variance": 1.0,
        },
    }
    (OUTPUT / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(rows)} phase-diagram rows to {OUTPUT}")


if __name__ == "__main__":
    run()
