"""Preprocessing for the UCI hydraulic condition-monitoring dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class SensorSpec:
    name: str
    quantity: str
    unit: str
    sampling_hz: int


SENSOR_SPECS = (
    *(SensorSpec(f"PS{i}", "pressure", "bar", 100) for i in range(1, 7)),
    SensorSpec("EPS1", "motor_power", "W", 100),
    SensorSpec("FS1", "volume_flow", "l/min", 10),
    SensorSpec("FS2", "volume_flow", "l/min", 10),
    *(SensorSpec(f"TS{i}", "temperature", "degC", 1) for i in range(1, 5)),
    SensorSpec("VS1", "vibration", "mm/s", 1),
    SensorSpec("CE", "cooling_efficiency", "percent", 1),
    SensorSpec("CP", "cooling_power", "kW", 1),
    SensorSpec("SE", "efficiency_factor", "percent", 1),
)

FEATURE_NAMES = ("mean", "std", "rms", "min", "max")
TARGET_NAMES = ("cooler", "valve", "pump", "accumulator", "stable_flag")


def summarize_cycles(values: NDArray[np.float64]) -> NDArray[np.float64]:
    if values.ndim != 2:
        raise ValueError("sensor values must be a cycle-by-time matrix")
    return np.column_stack(
        [
            np.mean(values, axis=1),
            np.std(values, axis=1),
            np.sqrt(np.mean(np.square(values), axis=1)),
            np.min(values, axis=1),
            np.max(values, axis=1),
        ]
    )


def transform_summary_features(
    features: NDArray[np.float64],
    gain: NDArray[np.float64],
    bias: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Exactly recompute summaries after ``trace' = (1 + gain) trace + bias``.

    ``features`` has five consecutive summaries per sensor. ``gain`` and
    ``bias`` may be one-dimensional or have the same leading dimension as
    ``features``. Gains must preserve trace order, which is the relevant
    operating regime for the bounded attacks used in the experiments.
    """

    values = np.asarray(features, dtype=float)
    if values.shape[-1] != len(SENSOR_SPECS) * len(FEATURE_NAMES):
        raise ValueError("features must contain five summaries per sensor")
    blocks = values.reshape(*values.shape[:-1], len(SENSOR_SPECS), 5)
    gain = np.asarray(gain, dtype=float)
    bias = np.asarray(bias, dtype=float)
    scale = 1.0 + gain
    if np.any(scale < 0):
        raise ValueError("gain must be at least -1")
    mean, std, rms, minimum, maximum = np.moveaxis(blocks, -1, 0)
    transformed = np.stack(
        [
            scale * mean + bias,
            scale * std,
            np.sqrt(np.maximum(scale**2 * rms**2 + 2 * scale * bias * mean + bias**2, 0.0)),
            scale * minimum + bias,
            scale * maximum + bias,
        ],
        axis=-1,
    )
    return transformed.reshape(values.shape)


def summary_parameter_jacobian(
    features: NDArray[np.float64], bias_scales: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Jacobian from dimensionless sensor gain/bias controls to summaries.

    The two controls for each sensor are fractional gain and bias divided by
    ``bias_scales``. The returned shape is ``(..., 85, 34)``.
    """

    values = np.asarray(features, dtype=float)
    blocks = values.reshape(*values.shape[:-1], len(SENSOR_SPECS), 5)
    bias_scales = np.asarray(bias_scales, dtype=float)
    if bias_scales.shape != (len(SENSOR_SPECS),):
        raise ValueError("bias_scales must contain one value per sensor")
    jacobian = np.zeros((*values.shape[:-1], values.shape[-1], 2 * len(SENSOR_SPECS)))
    for sensor in range(len(SENSOR_SPECS)):
        mean, std, rms, minimum, maximum = np.moveaxis(blocks[..., sensor, :], -1, 0)
        gain_derivative = np.stack([mean, std, rms, minimum, maximum], axis=-1)
        safe_rms = np.maximum(rms, np.finfo(float).eps)
        bias_derivative = np.stack(
            [np.ones_like(mean), np.zeros_like(mean), mean / safe_rms,
             np.ones_like(mean), np.ones_like(mean)], axis=-1
        ) * bias_scales[sensor]
        rows = slice(sensor * 5, (sensor + 1) * 5)
        jacobian[..., rows, sensor * 2] = gain_derivative
        jacobian[..., rows, sensor * 2 + 1] = bias_derivative
    return jacobian


def extract_cycle_features(
    archive_path: str,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Read immutable raw traces from ZIP and return features, means, labels."""

    feature_blocks: list[NDArray[np.float64]] = []
    mean_blocks: list[NDArray[np.float64]] = []
    expected_cycles: int | None = None
    with ZipFile(archive_path) as archive:
        for sensor in SENSOR_SPECS:
            with archive.open(f"{sensor.name}.txt") as handle:
                values = np.loadtxt(handle, dtype=float)
            if expected_cycles is None:
                expected_cycles = values.shape[0]
            if values.shape[0] != expected_cycles:
                raise ValueError(f"cycle mismatch for sensor {sensor.name}")
            expected_samples = sensor.sampling_hz * 60
            if values.shape[1] != expected_samples:
                raise ValueError(
                    f"{sensor.name} has {values.shape[1]} samples; "
                    f"expected {expected_samples}"
                )
            summary = summarize_cycles(values)
            feature_blocks.append(summary)
            mean_blocks.append(summary[:, :1])

        with archive.open("profile.txt") as handle:
            targets = np.loadtxt(handle, dtype=int)

    features = np.hstack(feature_blocks)
    means = np.hstack(mean_blocks)
    if targets.shape != (features.shape[0], len(TARGET_NAMES)):
        raise ValueError("target matrix does not align with sensor cycles")
    if not np.all(np.isfinite(features)):
        raise ValueError("derived features contain non-finite values")
    return features, means, targets


def grouped_splits(
    targets: NDArray[np.int64],
    *,
    seed: int,
    block_size: int = 10,
    fractions: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> Mapping[str, NDArray[np.int64]]:
    """Split contiguous cycle blocks without leaking cycles across partitions."""

    if targets.ndim != 2:
        raise ValueError("targets must be a cycle-by-task matrix")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if len(fractions) != 3 or not np.isclose(sum(fractions), 1.0):
        raise ValueError("split fractions must contain three values summing to one")

    cycle_count = targets.shape[0]
    groups = np.arange(cycle_count) // block_size
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)

    train_end = int(round(fractions[0] * unique_groups.size))
    validation_end = train_end + int(round(fractions[1] * unique_groups.size))
    group_partitions = {
        "train": unique_groups[:train_end],
        "validation": unique_groups[train_end:validation_end],
        "test": unique_groups[validation_end:],
    }
    return {
        name: np.flatnonzero(np.isin(groups, partition)).astype(np.int64)
        for name, partition in group_partitions.items()
    }


def split_audit(
    targets: NDArray[np.int64], splits: Mapping[str, NDArray[np.int64]]
) -> dict[str, object]:
    global_classes = [set(values.tolist()) for values in targets.T]
    report: dict[str, object] = {}
    all_indices: list[int] = []
    for name, indices in splits.items():
        all_indices.extend(indices.tolist())
        present = [set(targets[indices, column].tolist()) for column in range(targets.shape[1])]
        report[name] = {
            "rows": int(indices.size),
            "class_values": [sorted(values) for values in present],
            "covers_all_global_classes": [
                present[column] == global_classes[column]
                for column in range(targets.shape[1])
            ],
        }
    report["disjoint_and_complete"] = (
        len(all_indices) == targets.shape[0]
        and len(set(all_indices)) == targets.shape[0]
    )
    return report
